//
// Created by zerenluo on 21.10.22.
//
// replay the recorded manual actuation history in here, and then record the q, dq data for training

#include <Eigen/Dense>

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include<fstream>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace UNITREE_LEGGED_SDK;

#define INIT_COUNT 200 // for smooth the initial position
#define N_JOINTS 12

#define ACTION_FREQUENCY_TIMES 1 // regulate the velocity, the smaller, the slower; Range: 0.75 ~ 2
#define HARDWARE_FEEDBACK_FREQUENCY 0.0025 // sample frequency 400 Hz; fixed

class Custom
{
 public:
  Custom(uint8_t level, char* signal_path, char* mode_path):
      safe(LeggedType::Go1),
      udp(level, 8090, "192.168.123.10", 8007) {
    udp.InitCmdData(cmd);
    signal_path_ = signal_path; mode_path_ = mode_path;

    readSignalFile();

    // init kp, kd vector
    Kp.setZero(N_JOINTS); Kd.setZero(N_JOINTS);

  }
  void UDPRecv();
  void UDPSend();
  void RobotControl();
  void readSignalFile();
  void saveResponseAsFile();


    Safety safe;
  UDP udp;
  LowCmd cmd = {0};
  LowState state = {0};
  float qInit[N_JOINTS]={0};
  float qDes[N_JOINTS]={0};
  float Kp_hip, Kp_thigh, Kp_calf;
  float Kd_hip, Kd_thigh, Kd_calf;
  Eigen::VectorXf Kp;
  Eigen::VectorXf Kd;

  int rate_count = 0;
  int sin_count = 0;
  int motiontime = 0;
  float dt = 0.028;     // Base sampling frequency/ highlevel sine signal frequency

 private:
  string signal_path_, mode_path_;
  int signal_length_;
  vector<float> qSignal_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> qSignalMat_;

  // response matrix
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> actMat_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> qReMat_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dqReMat_;

  vector<float> act_;
  vector<float> qRe_;
  vector<float> dqRe_;

};

void Custom::UDPRecv()
{
  udp.Recv();
  udp.GetRecv(state);

  // save the history of q, dq
  if(motiontime >= INIT_COUNT && motiontime < signal_length_ + INIT_COUNT -1) { // stop accumulate once reach boundary
    for (int i = 0; i < N_JOINTS; i++) {
      act_.push_back(qDes[i]);
      qRe_.push_back(state.motorState[i].q);
      dqRe_.push_back(state.motorState[i].dq);
    }
  }
}

void Custom::UDPSend()
{
  for (int i = 0; i < N_JOINTS; i++) {
    cmd.motorCmd[i].q = qDes[i];
    cmd.motorCmd[i].dq = 0;
    cmd.motorCmd[i].Kp = Kp(i);
    cmd.motorCmd[i].Kd = Kd(i);
    cmd.motorCmd[i].tau = 0.0f;
  }

  safe.PositionLimit(cmd);
  int res1 = safe.PowerProtect(cmd, state, 6);
  if(res1 < 0) exit(-1);
  udp.SetSend(cmd);
  udp.Send();
}

double jointLinearInterpolation(double initPos, double targetPos, double rate)
{
  double p;
  rate = std::min(std::max(rate, 0.0), 1.0);
  p = initPos*(1-rate) + targetPos*rate;
  return p;
}

void Custom::readSignalFile() {
  ifstream fileStream;
  std::string line_info, input_result;
  fileStream.open("/home/zerenluo/unitree_ws/unitree_legged_sdk/actuator_network/data/" + signal_path_ + "/" + mode_path_ + ".txt", ios::in);

  int i = 0; // row count

  if (fileStream.fail()) { // no such file
    throw std::logic_error("read file fail");
  } else { // success
    while (getline(fileStream, line_info, '\n')) { // read a row
      stringstream input(line_info);

      for (int j = 0; input >> input_result; ++j) {
        string::size_type size;
//        qSignalMat_(i, j) = stof(input_result, &size);
        qSignal_.push_back(stof(input_result, &size));
      }
      i += 1; // row count
    }
  }

  signal_length_ = i;
  qSignalMat_ = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(qSignal_.data(), signal_length_, qSignal_.size()/signal_length_);

}

void Custom::RobotControl()
{
  motiontime++;
   printf("%d  %f\n", motiontime, state.motorState[FR_2].q);
//  printf("%d  %f  %f\n", motiontime, state.motorState[FR_1].q, state.motorState[FR_1].dq);


  if( motiontime >= 0){
    // first, get record initial position
    if( motiontime >= 0 && motiontime < 20){
      for (int i = 0; i < N_JOINTS; i++) {
        qInit[i] = state.motorState[i].q;
      }
    }
    // second, move to the first qSignalMat state
    if( motiontime >= 20 && motiontime < INIT_COUNT){
      rate_count++;
      double rate = rate_count/double(INIT_COUNT);                       // needs count to 100
      Kp_hip = 20.0; Kd_hip = 0.5;
      Kp_thigh = 20.0; Kd_thigh = 0.5;
      Kp_calf = 20.0; Kd_calf = 0.5;
      Kp << Kp_hip, Kp_thigh, Kp_calf,
            Kp_hip, Kp_thigh, Kp_calf,
            Kp_hip, Kp_thigh*2, Kp_calf*2,
            Kp_hip, Kp_thigh*2, Kp_calf*2;
      Kd << Kd_hip, Kd_thigh, Kd_calf,
            Kd_hip, Kd_thigh, Kd_calf,
            Kd_hip, Kd_thigh, Kd_calf,
            Kd_hip, Kd_thigh, Kd_calf;

      for (int i = 0; i < N_JOINTS; i++) {
        qDes[i] = jointLinearInterpolation(qInit[i], qSignalMat_(1, i), rate);
      }
    }
    // third, execute the signal trajectory
    if( motiontime >= INIT_COUNT && motiontime < signal_length_ + INIT_COUNT){
      Kp_hip = 20.0; Kd_hip = 0.5;
      Kp_thigh = 20.0; Kd_thigh = 0.5;
      Kp_calf = 20.0; Kd_calf = 0.5;
      Kp << Kp_hip, Kp_thigh, Kp_calf,
          Kp_hip, Kp_thigh, Kp_calf,
          Kp_hip, Kp_thigh*2, Kp_calf*2,
          Kp_hip, Kp_thigh*2, Kp_calf*2;
      Kd << Kd_hip, Kd_thigh, Kd_calf,
          Kd_hip, Kd_thigh, Kd_calf,
          Kd_hip, Kd_thigh, Kd_calf,
          Kd_hip, Kd_thigh, Kd_calf;

      std::cout << "********************" << Kp(1) << Kp(4) << Kp(7) << std::endl;
      for (int i = 0; i < N_JOINTS; i++) { // first try with front legs
        qDes[i] = qSignalMat_(motiontime-INIT_COUNT, i);
      }
    }
  }

  if(motiontime >= signal_length_ + INIT_COUNT -1) {
    saveResponseAsFile();
    exit(-1);
  }

}

void Custom::saveResponseAsFile() {
  string folderPath = "/home/zerenluo/unitree_ws/unitree_legged_sdk/actuator_network/data/training/"
                      + signal_path_ + "_f" + to_string(dt * ACTION_FREQUENCY_TIMES) + "/" + mode_path_;
  string command;
  command = "mkdir -p " + folderPath;
  system(command.c_str());

  std::cout << "act size: " << act_.size();
  std::cout << "q size: " << qRe_.size();
  std::cout << "dq size: " << dqRe_.size();

  Eigen::VectorXf qVec = Eigen::Map<Eigen::VectorXf>(qRe_.data(), qRe_.size());
  Eigen::VectorXf actVec = Eigen::Map<Eigen::VectorXf>(act_.data(), act_.size());
  Eigen::VectorXf dqVec = Eigen::Map<Eigen::VectorXf>(dqRe_.data(), dqRe_.size());
  std::cout << "q vec: " << qVec.size();
  std::cout << "act vec: " << actVec.size();
  std::cout << "dq vec: " << dqVec.size();

  // transfer vector to mat
  actMat_ = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, N_JOINTS, Eigen::RowMajor>>(act_.data(), act_.size()/N_JOINTS, N_JOINTS);
  qReMat_ = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, N_JOINTS, Eigen::RowMajor>>(qRe_.data(), qRe_.size()/N_JOINTS, N_JOINTS);
  dqReMat_ = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, N_JOINTS, Eigen::RowMajor>>(dqRe_.data(), dqRe_.size()/N_JOINTS, N_JOINTS);

  std::cout << "act mat: " << actMat_.size();
  std::cout << "q mat: " << qReMat_.size();
  std::cout << "dq mat: " << dqReMat_.size();

  ofstream qOutFile;
  qOutFile.open(folderPath + "/qResponse.txt");
  qOutFile << qReMat_;
  qOutFile.close();

  ofstream dqOutFile;
  dqOutFile.open(folderPath + "/dqResponse.txt");
  dqOutFile << dqReMat_;
  dqOutFile.close();

  ofstream actOutFile;
  actOutFile.open(folderPath + "/actResponse.txt");
  actOutFile << actMat_;
  actOutFile.close();

}

int main(int argc, char **argv)
{
  std::cout << "Communication level is set to LOW-level." << std::endl
            << "WARNING: Make sure the robot is hung up." << std::endl
            << "Press Enter to continue..." << std::endl;
  std::cin.ignore();

  char* signal_path = argv[1];
  char* mode_path = argv[2];
  Custom custom(LOWLEVEL, signal_path, mode_path);
  // InitEnvironment();
  LoopFunc loop_control("control_loop", custom.dt * ACTION_FREQUENCY_TIMES,    boost::bind(&Custom::RobotControl, &custom));
  LoopFunc loop_udpSend("udp_send",     custom.dt * ACTION_FREQUENCY_TIMES, 3, boost::bind(&Custom::UDPSend,      &custom));
  LoopFunc loop_udpRecv("udp_recv",     HARDWARE_FEEDBACK_FREQUENCY, 3, boost::bind(&Custom::UDPRecv,      &custom));

  loop_udpSend.start();
  loop_udpRecv.start();
  loop_control.start();

  while(1){
    std::cout << "sleep" << std::endl;
    sleep(10);
  };

  return 0;
}

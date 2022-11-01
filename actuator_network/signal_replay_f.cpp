//
// Created by zerenluo on 01.11.22.
//

// TODO: unitree low-level controller cannot control the sampling frequency, need to implement by ROS

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

class Custom
{
 public:
  Custom(uint8_t level, char* signal_path, char* mode_path):
      safe(LeggedType::Go1),
      udp(level, 8090, "192.168.123.10", 8007) {
    udp.InitCmdData(cmd);
    signal_path_ = signal_path; mode_path_ = mode_path;

    readSignalFile();
    qReMat_ = Eigen::MatrixXf::Zero(signal_length_, N_JOINTS);
    dqReMat_ = Eigen::MatrixXf::Zero(signal_length_, N_JOINTS);
    ddqReMat_ = Eigen::MatrixXf::Zero(signal_length_, N_JOINTS);
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
  float Kp;
  float Kd;
  int rate_count = 0;
  int sin_count = 0;
  int motiontime = 0;
  float dt = 0.016;     // 0.001~0.01; 0.016, 0.024, 0.012

 private:
  string signal_path_, mode_path_;
  int signal_length_;
  vector<float> qSignal_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> qSignalMat_;

  // response matrix
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> qReMat_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dqReMat_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> ddqReMat_;

};

void Custom::UDPRecv()
{
  udp.GetRecv(state);
  udp.Recv();
}

void Custom::UDPSend()
{
  for (int i = 0; i < N_JOINTS; i++) {
    cmd.motorCmd[i].q = qDes[i];
    cmd.motorCmd[i].dq = 0;
    cmd.motorCmd[i].Kp = Kp;
    cmd.motorCmd[i].Kd = Kd;
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
  // printf("%d  %f\n", motiontime, state.motorState[FR_2].q);
//  printf("%d  %f  %f\n", motiontime, state.motorState[FR_1].q, state.motorState[FR_1].dq);


  if( motiontime >= 0){
    // first, get record initial position
    if( motiontime >= 0 && motiontime < 10){
      for (int i = 0; i < N_JOINTS; i++) {
        qInit[i] = state.motorState[i].q;
      }
    }
    // second, move to the first qSignalMat state
    if( motiontime >= 10 && motiontime < INIT_COUNT){
      rate_count++;
      double rate = rate_count/double(INIT_COUNT);                       // needs count to 100
      Kp = 20.0;
      Kd = 0.5;

      for (int i = 0; i < N_JOINTS; i++) {
        qDes[i] = jointLinearInterpolation(qInit[i], qSignalMat_(1, i), rate);
      }
    }

    if( motiontime >= INIT_COUNT && motiontime < signal_length_ + INIT_COUNT){
      Kp = 20.0;
      Kd = 0.5;
      for (int i = 0; i < N_JOINTS; i++) { // first try with front legs
        qDes[i] = qSignalMat_(motiontime-INIT_COUNT, i);
      }

      // save the history of q, dq, ddq
      for (int i = 0; i < N_JOINTS; i++) {
        qReMat_(motiontime-INIT_COUNT, i) = state.motorState[i].q;
        dqReMat_(motiontime-INIT_COUNT, i) = state.motorState[i].dq;
        ddqReMat_(motiontime-INIT_COUNT, i) = state.motorState[i].ddq;
      }
    }
  }

  if(motiontime >= signal_length_ + INIT_COUNT -1) {
    saveResponseAsFile();
    exit(-1);
  }

}

void Custom::saveResponseAsFile() {
  string folderPath = "/home/zerenluo/unitree_ws/unitree_legged_sdk/actuator_network/data/training/" + signal_path_ + "/" + mode_path_;
  string command;
  command = "mkdir -p " + folderPath;
  system(command.c_str());

  ofstream qOutFile;
  qOutFile.open(folderPath + "/qResponse.txt");
  qOutFile << qReMat_;
  qOutFile.close();

  ofstream dqOutFile;
  dqOutFile.open(folderPath + "/dqResponse.txt");
  dqOutFile << dqReMat_;
  dqOutFile.close();

  ofstream ddqOutFile;
  ddqOutFile.open(folderPath + "/ddqResponse.txt");
  ddqOutFile << ddqReMat_;
  ddqOutFile.close();
}

//int main(int argc, char* argv[])
//{
//  std::cout << "Communication level is set to LOW-level." << std::endl
//            << "WARNING: Make sure the robot is hung up." << std::endl
//            << "Press Enter to continue..." << std::endl;
//  std::cin.ignore();
//
//  char* signal_path = argv[1];
//  char* mode_path = argv[2];
//  Custom custom(LOWLEVEL, signal_path, mode_path);
//  // InitEnvironment();
//  LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
//  LoopFunc loop_udpSend("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
//  LoopFunc loop_udpRecv("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));
//
//  loop_udpSend.start();
//  loop_udpRecv.start();
//  loop_control.start();
//
//  while(1){
//    sleep(10);
//  };
//
//  return 0;
//}

int main(int argc, char **argv) {
  std::cout << "Communication level is set to LOW-level." << std::endl
            << "WARNING: Make sure the robot is hung up." << std::endl
            << "Press Enter to continue..." << std::endl;
  std::cin.ignore();

  ros::init(argc, argv, "go1_collect_data");
  ros::NodeHandle nh;


  // create go1 controllers
  char* signal_path = argv[1];
  char* mode_path = argv[2];
  Custom custom(LOWLEVEL, signal_path, mode_path);


  std::atomic<bool> control_execute{};
  control_execute.store(true, std::memory_order_release);

  // Thread 1: compute command, forward the NN policy network
  std::cout << "Enter thread 1: compute desired action command" << std::endl;
  std::thread compute_action_thread([&]() {
    // prepare variables to monitor time and control the while loop
    ros::Time start = ros::Time::now();
    ros::Time prev = ros::Time::now();
    ros::Time now = ros::Time::now();  // bool res = app.exec();
    ros::Duration dt(0);

    while (control_execute.load(std::memory_order_acquire) && ros::ok()) {

      ros::Duration(ACTION_UPDATE_FREQUENCY / 1000).sleep();

      // get t and dt
      now = ros::Time::now();
      dt = now - prev;
      prev = now;

      auto t1 = std::chrono::high_resolution_clock::now();

      bool running = go1_rl->advance(dt.toSec());

      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> ms_double = t2 - t1;

      if (!running) {
        std::cout << "Thread 1 loop is terminated because of errors." << std::endl;
        ros::shutdown();
        std::terminate();
        break;
      }
    }

  });

  // Thread 2: depolyment on the real system; send commands
  std::cout << "Enter thread 2: Deployment on the real robot, sending command" << std::endl;
  std::thread deploy_thread([&]() {
    // prepare variables to monitor time and control the while loop
    ros::Time start = ros::Time::now();
    ros::Time prev = ros::Time::now();
    ros::Time now = ros::Time::now();  // bool res = app.exec();
    ros::Duration dt(0);

    while (control_execute.load(std::memory_order_acquire) && ros::ok()) {
      auto t3 = std::chrono::high_resolution_clock::now();

      ros::Duration(DEPLOYMENT_FREQUENCY / 1000).sleep();

      // get t and dt
      now = ros::Time::now();
      dt = now - prev;
      prev = now;

      bool send_cmd_running  = go1_rl->send_cmd();

      auto t4 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> ms_double = t4 - t3;

      if (!send_cmd_running) {
        std::cout << "Thread 2 loop is terminated because of errors." << std::endl;
        ros::shutdown();
        std::terminate();
        break;
      }

    }
  });

  ros::AsyncSpinner spinner(12);
  spinner.start();

  compute_action_thread.join();
  deploy_thread.join();

  return 0;

}





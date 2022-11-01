//
// Created by zerenluo on 20.10.22.
//

// record the manual actuation history in here
// we can manually control the actuation frequency
#include <Eigen/Dense>

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <fstream>

using namespace std;
using namespace UNITREE_LEGGED_SDK;

#define SIGNAL_LENGTH 20000
#define N_JOINTS 12

class Custom
{
 public:
  Custom(uint8_t level):
      safe(LeggedType::Go1),
      udp(level, 8090, "192.168.123.10", 8007) {
    udp.InitCmdData(cmd);
  }
  void UDPRecv();
  void UDPSend();
  void RobotControl();
  void saveSignalAsFile();

  Safety safe;
  UDP udp;
  LowCmd cmd = {0};
  LowState state = {0};
  float qInit[3]={0};
  float qDes[3]={0};
  float sin_mid_q[3] = {0.0, 1.2, -2.0};
  float Kp[3] = {0};
  float Kd[3] = {0};
  double time_consume = 0;
  int rate_count = 0;
  int sin_count = 0;
  int motiontime = 0;
  float dt = 0.002;     // 0.001~0.01
  int signalLength = 5000;

 private:
  int totalTime_;
  vector<float> qSignal_;

};

void Custom::UDPRecv()
{
  udp.Recv();
}

void Custom::UDPSend()
{
  udp.Send();
}

void Custom::RobotControl()
{
  motiontime++;
  udp.GetRecv(state);
  // printf("%d  %f\n", motiontime, state.motorState[FR_2].q);
  printf("%d  %f  %f\n", motiontime, state.motorState[FR_1].q, state.motorState[FR_1].dq);

  // make sure this are zero and send this cmd, so there is no torque in the joint
  Kp[0] = 0.0; Kp[1] = 0.0; Kp[2] = 0.0;
  Kd[0] = 0.0; Kd[1] = 0.0; Kd[2] = 0.0;
  for (int i = 0; i < 12; i++) {
    cmd.motorCmd[i].q = qDes[0];
    cmd.motorCmd[i].dq = 0;
    cmd.motorCmd[i].Kp = Kp[0];
    cmd.motorCmd[i].Kd = Kd[0];
    cmd.motorCmd[i].tau = 0.0f;
  }
  udp.SetSend(cmd);


  // TODO: record and save the actuation signal
  for (int i = 0; i < N_JOINTS; i++) { // first try with front legs
    qSignal_.push_back(state.motorState[i].q);
  }

  if (motiontime >= SIGNAL_LENGTH) {
    totalTime_ = motiontime;
    saveSignalAsFile();
    exit(-1);
  }
}


void Custom::saveSignalAsFile() {
  // change list into eigen matrix
  Eigen::MatrixXf qSignalMat = Eigen::Map<Eigen::Matrix<float, SIGNAL_LENGTH, N_JOINTS, Eigen::RowMajor>>(qSignal_.data());
  ofstream outFile;
  outFile.open("/home/zerenluo/unitree_ws/unitree_legged_sdk/actuator_network/data/qSignal.txt");
  outFile << qSignalMat;
  outFile.close();
}


int main(void)
{
  std::cout << "Communication level is set to LOW-level." << std::endl
            << "WARNING: Make sure the robot is hung up." << std::endl
            << "Press Enter to continue..." << std::endl;
  std::cin.ignore();

  Custom custom(LOWLEVEL);
  // InitEnvironment();
  LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
  LoopFunc loop_udpSend("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
  LoopFunc loop_udpRecv("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));

  loop_udpSend.start();
  loop_udpRecv.start();
  loop_control.start();

  while(1){
    sleep(10);
  };

  return 0;
}

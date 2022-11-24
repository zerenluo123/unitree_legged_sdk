import torch
from torch.utils.data import Dataset
import os
import numpy as np

LEG_NUM = 4
LEG_DOF = 3
HARDWARE_FEEDBACK_FREQUENCY = 0.0025 # consistent with signal_replay
SAMPLING_FREQUENCY = 0.005

class MotionData(Dataset):
    def __init__(self, root_dir):
        # ! define the training symbol
        self.symbol = ['q_err', 'dq'] # ['q', 'dq', 'act']

        # ! define the history length h
        self.len_hist = 50
        self.interval = int(SAMPLING_FREQUENCY / HARDWARE_FEEDBACK_FREQUENCY)
        self.model_in_size = len(self.symbol) * LEG_DOF * self.len_hist

        self.q_err = []
        self.dq = []
        self.label = []

        self.root_dir = root_dir

        # ! open the data document and read the data
        response_path = os.path.join(self.root_dir, "training")
        self.frequency_names = os.listdir(response_path)

        print("Loading motion data from directory")
        for frequency_name in self.frequency_names:
            frequency_path = os.path.join(response_path, frequency_name)
            mode_names = os.listdir(frequency_path)
            for mode_name in mode_names:
                mode_path = os.path.join(frequency_path, mode_name)
                print(mode_path) # for checking the mode order
                q_temp = np.loadtxt(mode_path + "/qResponse.txt")
                dq_temp = np.loadtxt(mode_path + "/dqResponse.txt")
                act_temp = np.loadtxt(mode_path + "/actResponse.txt")

                # fixed pos, 0 vel, 0 setpt, then calculate q_err
                zero_init_buffer = np.zeros(( self.interval*self.len_hist-1, 12 ))
                act_temp = np.concatenate((zero_init_buffer, act_temp), axis=0)
                dq_temp = np.concatenate((zero_init_buffer, dq_temp), axis=0)
                # fixed pos
                q_init_buffer = np.tile(q_temp[0,:], (self.interval*self.len_hist-1, 1))
                q_temp = np.concatenate((q_init_buffer, q_temp), axis=0)
                q_err_temp = act_temp - q_temp # calculate the joint position error; q_des - q_curr

                self.make_dataset(q_err_temp, dq_temp)

        print("data volume", len(self.q_err))
        self.normalization()


    def normalization(self):
        # normalization/Standardization. need to know the mean and variance of each group of data
        self.q_err_mean, self.q_err_std = np.mean(np.array(self.q_err).reshape(-1, LEG_DOF), axis=0), \
                                          np.std(np.array(self.q_err).reshape(-1, LEG_DOF), axis=0)
        self.dq_mean, self.dq_std = np.mean(np.array(self.dq).reshape(-1, LEG_DOF), axis=0), \
                                    np.std(np.array(self.dq).reshape(-1, LEG_DOF), axis=0)
        self.dVel_mean, self.dVel_std = np.mean(np.array(self.label), axis=0), np.std(np.array(self.label), axis=0)
        print("q_err_mean: ", self.q_err_mean, "\n", "q_err_std: ", self.q_err_std)
        print("dq_mean: ", self.dq_mean, "\n", "dq_std: ", self.dq_std)
        print("dVel_mean: ", self.dVel_mean, "\n", "dVel_std: ", self.dVel_std)
        print("*******************************************************")

        for i in range(len(self.q_err)):
            self.q_err[i] = (self.q_err[i] - self.q_err_mean) / self.q_err_std
            self.dq[i] = (self.dq[i] - self.dq_mean) / self.dq_std
            self.label[i] = (self.label[i] - self.dVel_mean) / self.dVel_std

        # # Debugging check
        # self.q_err_mean, self.q_err_std = np.mean(np.array(self.q_err).reshape(-1, LEG_DOF), axis=0), \
        #                                   np.std(np.array(self.q_err).reshape(-1, LEG_DOF), axis=0)
        # self.dq_mean, self.dq_std = np.mean(np.array(self.dq).reshape(-1, LEG_DOF), axis=0), \
        #                             np.std(np.array(self.dq).reshape(-1, LEG_DOF), axis=0)
        # self.dVel_mean, self.dVel_std = np.mean(np.array(self.label), axis=0), np.std(np.array(self.label), axis=0)
        # print("AFTER q_err_mean: ", self.q_err_mean, " AFTER q_err_std: ", self.q_err_std)
        # print("AFTER dq_mean: ", self.dq_mean, " AFTER dq_std: ", self.dq_std)
        # print("AFTER dVel_mean: ", self.dVel_mean, " AFTER dVel_std: ", self.dVel_std)

    def make_dataset(self, q_err_temp, dq_temp):
        # return 1. concatenation of history state(action/q_err(t, t-0.005, t-0.01, t-0.015, t-0.02), dq(t, t-0.005, t-0.01, t-0.015, t-0.02); 2. dq(t+0.005) - dq(t)
        for i in range(q_err_temp.shape[0] - self.interval * self.len_hist):
            q_err_hist = q_err_temp[i: i + self.interval * self.len_hist: self.interval, :]  # dim: [hist, 12]
            dq_hist = dq_temp[i: i + self.interval * self.len_hist: self.interval, :]  # dim: [hist, 12]
            label = dq_temp[i + self.interval * self.len_hist, :] - dq_temp[i + self.interval * (self.len_hist - 1), :]  # dim: [12, ]
            for i in range(LEG_NUM):  # expand the dataset by integrating 4 legs'data into 1 leg's data
                self.q_err.append(q_err_hist[:, i * LEG_DOF:(i + 1) * LEG_DOF])
                self.dq.append(dq_hist[:, i * LEG_DOF:(i + 1) * LEG_DOF])
                self.label.append(label[i * LEG_DOF:(i + 1) * LEG_DOF])


    def __getitem__(self, index):
        # return 1. concatenation of history state(action/q_err(t, t-0.005, t-0.01, t-0.015, t-0.02), dq(t, t-0.005, t-0.01, t-0.015, t-0.02); 2. dq(t+0.005) - dq(t)
        model_in = np.array([])
        for dof in range(LEG_DOF):
            model_in = np.concatenate( (model_in, self.q_err[index][:, dof]), axis=0 )
            model_in = np.concatenate( (model_in, self.dq[index][:, dof]), axis=0 )

        label = self.label[index]
        return model_in, label

    def __len__(self):
        # return the total length
        return len(self.q_err)

# custom_dataset = MotionData(root_dir='../data')
# print(custom_dataset[0])

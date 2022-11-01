import torch
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn import preprocessing

LEG_NUM = 4
LEG_DOF = 3

class MotionData(Dataset):
    def __init__(self, root_dir):

        # ! define the history length h
        self.len_hist = 10
        self.model_in_size = 3 * LEG_DOF * self.len_hist

        # open the data document and read the data
        # self.q_data = np.array([[0 for i in range(LEG_DOF)]])
        # self.dq_data = np.array([[0 for i in range(LEG_DOF)]])
        # self.act_data = np.array([[0 for i in range(LEG_DOF)]])
        self.q_data = np.zeros((self.len_hist - 1, LEG_DOF))
        self.dq_data = np.zeros((self.len_hist - 1, LEG_DOF))
        self.act_data = np.zeros((self.len_hist - 1, LEG_DOF))

        self.root_dir = root_dir

        # load q and dq data
        response_path = os.path.join(self.root_dir, "training")
        self.frequency_names = os.listdir(response_path)

        for frequency_name in self.frequency_names:
            frequency_path = os.path.join(response_path, frequency_name)
            mode_names = os.listdir(frequency_path)
            for mode_name in mode_names:
                mode_path = os.path.join(frequency_path, mode_name)
                # print(mode_path) # for checking the mode order
                q_temp = np.loadtxt(mode_path + "/qResponse.txt")
                dq_temp = np.loadtxt(mode_path + "/dqResponse.txt")
                act_temp = np.loadtxt(mode_path + "/" + mode_name + ".txt")

                for i in range(LEG_NUM): # expand the dataset by integrating 4 legs'data into 1 leg's data
                    self.q_data = np.concatenate((self.q_data, q_temp[:, i*LEG_DOF:(i+1)*LEG_DOF]), axis=0)
                    self.dq_data = np.concatenate((self.dq_data, dq_temp[:, i*LEG_DOF:(i+1)*LEG_DOF]), axis=0)
                    self.act_data = np.concatenate((self.act_data, act_temp[:, i*LEG_DOF:(i+1)*LEG_DOF]), axis=0)

        # delete the first 0 row
        # self.q_data = np.delete(self.q_data, 0, axis=0)
        # self.dq_data = np.delete(self.dq_data, 0, axis=0)
        # self.act_data = np.delete(self.act_data, 0, axis=0)
        # print(self.q_data.shape)

        # normalization/Standardization. need to know the mean and variance of each group of data
        self.q_data_mean, self.q_data_std = np.mean(self.q_data, axis=0), np.std(self.q_data, axis=0)
        self.dq_data_mean, self.dq_data_std = np.mean(self.dq_data, axis=0), np.std(self.dq_data, axis=0)
        self.act_data_mean, self.act_data_std = np.mean(self.act_data, axis=0), np.std(self.act_data, axis=0)

        self.q_data = (self.q_data - self.q_data_mean) / self.q_data_std
        self.dq_data = (self.dq_data - self.dq_data_mean) / self.dq_data_std
        self.act_data = (self.act_data - self.act_data_mean) / self.act_data_std
        # print(np.mean(self.q_data, axis=0), np.std(self.q_data, axis=0))


    def __getitem__(self, index):
        # return 1. concatenation of history state(action/qDes_(t, ..., t-h), q_(t, ..., t-h), dq_(t, ..., t-h)); 2. dq_(t+1)
        model_in = np.array([])
        q_hist = self.q_data[index:index + self.len_hist, :]
        dq_hist = self.dq_data[index:index + self.len_hist, :]
        act_hist = self.act_data[index:index + self.len_hist, :]

        for dof in range(LEG_DOF):
            model_in = np.concatenate( (model_in, q_hist[: , dof]), axis=0 )
            model_in = np.concatenate( (model_in, dq_hist[:, dof]), axis=0 )
            model_in = np.concatenate( (model_in, act_hist[:, dof]), axis=0 )

        label = self.dq_data[index + self.len_hist + 1, :] # dq at t+1
        return model_in, label

    def __len__(self):
        # return the total length
        return self.q_data.shape[0] - self.len_hist - 1
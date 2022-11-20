# TODO: test the model here
# TODO: pick one of the episode from the training set, roll forward using the prediction

# given the init dq(0), q(0) and act(0)
# --> input1: 0, ..., 0, q_err(0); input2: 0, ..., 0, dq(0)
# --> model(input, output)
# --> dq(1) = output + dq(0)
# --> q(1) = q(0) + dt * dq(1), or q(1) = q(0) + dt * dq(0)
# --> q_err(1) = act(1) - q(1)
# --> input1: 0, ...,q_err(0), q_err(1); input2: 0, ..., dq(0), dq(1)
# --> ...
import os
import numpy as np
import argparse

import torch
from torch import nn

from mlp import MLP

# !!!! be consistent with motion_data.py
SAMPLING_FREQUENCY = 0.005
LEG_NUM = 4
LEG_DOF = 3
SYMBOL = ['q_err', 'dq']
LEN_HIST = 5
MODEL_IN_SIZE = len(SYMBOL) * LEG_DOF * LEN_HIST

class UniNet(nn.Module):
    def __init__(self, model):
        super(UniNet, self).__init__()
        self.core_model = model

    def forward(self, x): # x: 4 * MODEL_IN_SIZE
        out = torch.tensor(()).to(x.device)
        for i in range(LEG_NUM):
            sub_in = x[:, MODEL_IN_SIZE*i:MODEL_IN_SIZE*(i+1)]
            sub_out = self.core_model(sub_in)
            out = torch.cat((out, sub_out), 1)
        return out



class TestModel():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-path', '--model_path', type=str, default='None',
                            help='load model path from which directory')
        args = parser.parse_args()

        # load previously trained model
        log_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs')
        load_run = args.model_path  # which time
        checkpoint = -1  # which model
        self.load_run, resume_path = self.get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)
        print(f"Loading model from: {resume_path}")

        # ! create an actuator network model
        hidden_layers = [128, 128, 128]
        self.device = torch.device('cuda')
        self.model = MLP(hidden_layers,
                             nn.Tanh,
                             MODEL_IN_SIZE,
                             output_size=3,
                             init_scale=1.4).to(self.device)
        self.model.load_state_dict(torch.load(resume_path))
        self.model.eval() # switch to evaluation mode (dropout for example)

        # get mean and std of input and output from data
        self.pos_err_mean = np.array([  0.00036437 , 0.01540757, -0.00972657])
        self.pos_err_std = np.array([0.11722939, 0.19275887, 0.28700321])
        self.vel_mean = np.array([-0.00017714, -0.00024455,  0.0005956 ])
        self.vel_std = np.array([2.31517027, 3.84613839, 5.52599008])

        # init pos err and vel buffer
        self.pos_err_buffs = np.zeros((LEG_DOF, LEN_HIST))
        self.vel_buffs = np.zeros((LEG_DOF, LEN_HIST))

        # TODO: init pos_err and vel with the data[0]
        self.pos = 0.
        self.vel = 0.


    def get_load_path(self, root, load_run, checkpoint=-1):
        try:
            runs = os.listdir(root)
            runs.sort()
            last_run = os.path.join(root, runs[-1])
        except:
            raise ValueError("No runs in this directory: " + root)

        load_run = os.path.join(root, load_run)

        if checkpoint == -1:
            models = [file for file in os.listdir(load_run) if 'snapshot' in file]
            # print(len(models))
            models.sort(key=lambda m: '{0:0>20}'.format(m))
            model = models[-1]
        else:
            model = "snapshot_{}.pt".format(checkpoint)

        load_path = os.path.join(load_run, model)
        return load_run, load_path

    # def integrate_net(self):
    #     # TODO: integrate the input and output


    # TODO: organize the series of input
    def advance(self, action, pos, vel):  # action is passed-in value, pos and vel are both return value
        # scale pos_err and vel TODO: clip
        pos_err = action - self.pos
        pos_err_s = (pos_err, - self.pos_err_mean) / self.pos_err_std
        vel_s = (self.vel - self.vel_mean) / self.vel_std

        model_in = np.array([])
        for i in range(LEG_DOF):
            # fill buffers with scaled data [t-h, ... , t-0]
            # hist can be different for each joint
            self.pos_err_buffs[i, :] = np.delete(self.pos_err_buffs[i, :], 0)
            self.pos_err_buffs[i, :] = np.append(self.pos_err_buffs[i, :], pos_err_s)

            self.vel_buffs[i, :] = np.delete(self.vel_buffs[i, :], 0)
            self.vel_buffs[i, :] = np.append(self.vel_buffs[i, :], vel_s)

            # fill actuator model input vector
            model_in = np.concatenate((model_in, self.pos_err_buffs[i, :]), axis=0)
            model_in = np.concatenate((model_in, self.vel_buffs[i, :]), axis=0)

        # advance actuator mlp
        dVel = self.model(model_in.float())

        # upscale mlp output
        dVel *= self.vel_std

        # integrate vel in time
        #  x_t+1 = x_t + (xdot_t + xdot_t+1)/2 * dt
        #  x_t+1 = x_t + xdot_t*dt
        self.pos += SAMPLING_FREQUENCY  * self.vel
        self.vel += dVel

        # return
        pos = self.pos
        vel = self.vel

def main():
    test_model = TestModel()

    traced_script_model = torch.jit.script(test_model.model)
    traced_script_model.save(os.path.join(test_model.load_run, "go1_net.pt"))

    test_in = torch.cat((torch.ones(1, 30), 2*torch.ones(1, 30), 3*torch.ones(1, 30), 4*torch.ones(1, 30)), 0).to(test_model.device)
    test_out = test_model.model(test_in)
    print(test_out)

    uni_model = UniNet(test_model.model)
    test_in_uni = torch.cat((torch.ones(1, 30), 2*torch.ones(1, 30), 3*torch.ones(1, 30), 4*torch.ones(1, 30)), 1).to(test_model.device)
    test_out_uni = uni_model(test_in_uni)
    print(test_out_uni)



if __name__ == "__main__":
    main()





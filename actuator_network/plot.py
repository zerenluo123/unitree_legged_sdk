import matplotlib.pyplot as plt
import numpy as np
import csv


in_file = "/home/zerenluo/unitree_ws/unitree_legged_sdk/actuator_network/data/qDes.txt"
in_base_file = "/home/zerenluo/unitree_ws/unitree_legged_sdk/actuator_network/data/qDesBase.txt"
in_mod_file = "/home/zerenluo/unitree_ws/unitree_legged_sdk/actuator_network/data/qDesMod.txt"

q_des = np.loadtxt(in_file)
q_des_base = np.loadtxt(in_base_file)
q_des_mod = np.loadtxt(in_mod_file)

x = np.arange(q_des_base.shape[0])

# pos limit
go1_Hip_max   = 1.047    # unit:radian ( = 60   degree)
go1_Hip_min   = -1.047   # unit:radian ( = -60  degree)
go1_Thigh_max = 2.966    # unit:radian ( = 170  degree)
go1_Thigh_min = -0.663   # unit:radian ( = -38  degree)
go1_Calf_max  = -0.837   # unit:radian ( = -48  degree)
go1_Calf_min  = -2.721   # unit:radian ( = -156 degree)

# fig, axs = plt.subplots(1, 3)

plt.plot(x,go1_Hip_max * np.ones(q_des_base.shape[0]))
plt.plot(x,go1_Hip_min * np.ones(q_des_base.shape[0]))
# plt.plot(x,q_des[:,0])
plt.plot(x,q_des_base[:,0])
# plt.plot(x,q_des_mod[:,0])


# axs[1].plot(x,go1_Thigh_max * np.ones(q_des.shape[0]))
# axs[1].plot(x,go1_Thigh_min * np.ones(q_des.shape[0]))
# axs[1].plot(x,q_des[:,1])
# axs[1].plot(x,q_des_base[:,1])
# axs[1].plot(x,q_des_mod[:,1])
#
# axs[2].plot(x,go1_Calf_max * np.ones(q_des.shape[0]))
# axs[2].plot(x,go1_Calf_min * np.ones(q_des.shape[0]))
# axs[2].plot(x,q_des[:,2])
# axs[2].plot(x,q_des_base[:,2])
# axs[2].plot(x,q_des_mod[:,2])


# plt.plot(x,q_des[:,1])
# plt.plot(x,q_des[:,2])
plt.show()




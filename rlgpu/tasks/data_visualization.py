from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns
import numpy as np

sim_force = np.loadtxt("tasks/sim_force_record.txt")
real_force = np.loadtxt('/home/lohse/下载/force_record.txt')
print(sim_force.shape)
print(real_force.shape)

sim_x_list = []
sim_y_list = []
for i in range(sim_force.shape[0]):
    sim_x_list.append(i)
    sim_y_list.append(sim_force[i, 0])

real_x_list = []
real_y_list = []
for i in range(real_force.shape[0]):
    real_x_list.append(i)
    real_y_list.append(real_force[i, 0])

sns.relplot(x=sim_x_list, y=sim_y_list, kind="line", ci="sd")
sns.relplot(x=real_x_list, y=real_y_list, kind="line", ci="sd")

plt.show()
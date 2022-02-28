from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns
import numpy as np

sim_force = np.loadtxt("/home/lohse/isaac_ws/src/isaac-gym/scripts/Isaac-drlgrasp/rlgpu/tasks/sim_force_record.txt")
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
for j in range(real_force.shape[1]):
    real_x_tem = []
    real_y_tem = []
    for i in range(real_force.shape[0]):
        real_x_tem.append(i)
        real_y_tem.append(real_force[i, j])

    real_x_list.append(real_x_tem)
    real_y_list.append(real_y_tem)

# sns.relplot(x=sim_x_list, y=sim_y_list, kind="line", ci="sd")

plt.plot(real_x_list[0], real_y_list[0], color='blue', label='$x$', linewidth=0.8)
plt.plot(real_x_list[1], real_y_list[1], color='red', label='$y$', linewidth=0.8)

plt.plot(real_x_list[2], real_y_list[2], color='green', label='$z$', linewidth=0.8)

plt.legend(loc='upper left')                                # 绘制图例，指定图例位置
leg = plt.gca().get_legend()
ltext = leg.get_texts()

plt.savefig('./force.jpg', format='jpg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()

plt.show()
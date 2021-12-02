#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt





fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
#ax4 = fig.add_subplot(224)
#ax5 = fig.add_subplot(225)
#ax6 = fig.add_subplot(226)

ax = [ax1,ax2,ax3]

fname1 = ('x', 'y', 'z', 'rx', 'ry', 'rz')
for j in range(3, 6):
    fname = '/home/cobot/Desktop/assemble_project/track_data/assemble_0.01s/dataFile_%s.txt' % (fname1[j])
    dmp_fname = '/home/cobot/Desktop/assemble_project/track_data/assemble_0.01s/dmp_dataFile_%s.txt' % (fname1[j])
    X1, Y1 = [], []
    X2, Y2 = [], []

    with open(fname, 'r') as f1:#1
        lines1 = f1.readlines()#2
        for line1 in lines1:#3
            value1 = [float(s1) for s1 in line1.split()]#4
            X1.append(line1)
            Y1.append(value1[0])

    with open(dmp_fname, 'r') as f2:#1
        lines2 = f2.readlines()#2
        for line2 in lines2:#3
            value2 = [float(s2) for s2 in line2.split()]#4
            X2.append(line2)
            Y2.append(value2[0])

    ax[j-3].plot(Y1)
    ax[j-3].axis('tight') # 坐标轴适应数据量 axis 设置坐标轴
#    ax[j-3].plot(Y2)
#    ax[j-3].axis('tight') # 坐标轴适应数据量 axis 设置坐标轴



plt.show()


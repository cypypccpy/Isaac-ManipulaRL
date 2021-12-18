#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt





fig1 = plt.figure()
ax11 = fig1.add_subplot(131)
ax12 = fig1.add_subplot(132)
ax13 = fig1.add_subplot(133)
AX1 = [ax11,ax12,ax13]

fig2 = plt.figure()
ax21 = fig2.add_subplot(131)
ax22 = fig2.add_subplot(132)
ax23 = fig2.add_subplot(133)


AX2 = [ax21,ax22,ax23]

fname1 = ('x', 'y', 'z', 'rx', 'ry', 'rz', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
for j in range(9, 12):
    fname = '/home/xrh/Desktop/ur_assemble/data/track_data/assemble_joints_1s/dataFile_%s.txt' % (fname1[j])
    dmp_fname = '/home/xrh/Desktop/ur_assemble/data/track_data/assemble_joints_1s/dmp_dataFile_%s.txt' % (fname1[j])
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

    AX1[j-9].plot(Y1)
    AX1[j-9].axis('tight') 
    AX2[j-9].plot(Y2)
    AX2[j-9].axis('tight')



plt.show()


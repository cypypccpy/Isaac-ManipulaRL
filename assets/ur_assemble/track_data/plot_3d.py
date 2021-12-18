#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x,y,z = [], [], []
xd,yd,zd = [], [], []

x_fname = '/home/cobot/Desktop/assemble_project/track_data/assemble_joints_1s/dataFile_x.txt'
x_dmp_fname = '/home/cobot/Desktop/assemble_project/track_data/assemble_joints_1s/dmp_dataFile_x.txt'
with open(x_fname, 'r') as fx:#1
    linesx = fx.readlines()#2
    for linex in linesx:#3
        valuex = [float(sx) for sx in linex.split()]#4
        x.append(valuex[0])
with open(x_dmp_fname, 'r') as fxd:#1
    linesxd = fxd.readlines()#2
    for linexd in linesxd:#3
        valuexd = [float(sxd) for sxd in linexd.split()]#4
        xd.append(valuexd[0])

y_fname = '/home/cobot/Desktop/assemble_project/track_data/assemble_joints_1s/dataFile_y.txt'
y_dmp_fname = '/home/cobot/Desktop/assemble_project/track_data/assemble_joints_1s/dmp_dataFile_y.txt'
with open(y_fname, 'r') as fy:#1
    linesy = fy.readlines()#2
    for liney in linesy:#3
        valuey = [float(sy) for sy in liney.split()]#4
        y.append(valuey[0])
with open(y_dmp_fname, 'r') as fyd:#1
    linesyd = fyd.readlines()#2
    for lineyd in linesyd:#3
        valueyd = [float(syd) for syd in lineyd.split()]#4
        yd.append(valueyd[0])

z_fname = '/home/cobot/Desktop/assemble_project/track_data/assemble_joints_1s/dataFile_z.txt'
z_dmp_fname = '/home/cobot/Desktop/assemble_project/track_data/assemble_joints_1s/dmp_dataFile_z.txt'
with open(z_fname, 'r') as fz:#1
    linesz = fz.readlines()#2
    for linez in linesz:#3
        valuez = [float(sz) for sz in linez.split()]#4
        z.append(valuez[0])
with open(z_dmp_fname, 'r') as fzd:#1
    lineszd = fzd.readlines()#2
    for linezd in lineszd:#3
        valuezd = [float(szd) for szd in linezd.split()]#4
        zd.append(valuezd[0])

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_title("3D")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

figure = ax.plot(x,y,z,c='b')
figure = ax.plot(xd,yd,zd,c='g')

plt.show()

#!/usr/bin/env python
# Baxter Teleoperation & Obstacle Avoidance
# By Xinyu Wang
# 14 Mar 2015
# wangxinyu@bit.edu.cn

"""
Baxter Teleoperation & Obstacle Avoidance
"""
import argparse
import sys

from copy import copy

from numpy.lib.function_base import angle

import rospy

import cv2
import cv_bridge

import numpy as np
import struct
import baxter_interface
import baxter_external_devices
import socket
import threading
import time

import actionlib

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)

from baxter_interface import CHECK_VERSION
from baxter_pykdl import baxter_kinematics
import PyKDL

from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from sensor_msgs.msg import (
    Image,
)
# from force_receiver import Force_Receiver 
# my_force_receiver = Force_Receiver()
# my_force_receiver.start()

####################### Parallel System ########################
class CParaArmController:
    def __init__(self, ArmName, initq, _qmax, _qmin):
        self.name = ArmName
        self.kin = baxter_kinematics(ArmName)
        self.q = PyKDL.JntArray(self.kin._num_jnts)
        for i in range(0, 7):
            self.q[i] = initq[i]
        self.posd = np.mat(self.kin.forward_position_kinematics()).T
        self.rpyd = np.mat([3.14, 0, 3.14]).T
        #self.thread = threading.Thread(target=self.Calculate)
        #self.thread_stop = False
        self.qmax = _qmax
        self.qmin = _qmin
        self.Ie = np.mat([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
        self.vxyzlmt = 0.5
        self.vrpylmt = 0.5
        self.pxyz = 0
        self.prpy = 0
        self.kprate = 1.0
        #self.thread.start()
        
    def GetAllJointPos(self):
        pos = np.mat(np.zeros((12, 3)))
        for i in range(1,12):
            posraw = self.GetJointPos(i)
            pos[i,0] = posraw[0]
            pos[i,1] = posraw[1]
            pos[i,2] = posraw[2]
        return pos
        
    def GetJointPos(self,idx):
        return self.kin.forward_position_kinematicspang_chain(idx,self.q)
        
    def Calculate1(self):
        (xyz,rot) = self.kin.forward_position_kinematics()
        errxyz = self.posd - np.mat(xyz).T
        
        errrpyv = PyKDL.diff(PyKDL.Rotation.RPY(self.rpyd[0,0],self.rpyd[1,0],self.rpyd[2,0]), rot)
        
        errrpy = np.mat(-self.rpyd)
        errrpy[0,0] = -errrpyv[0]
        errrpy[1,0] = -errrpyv[1]
        errrpy[2,0] = -errrpyv[2]
        
        dx = self.pxyz*self.kprate*errxyz
        drpy = self.prpy*self.kprate*errrpy
        dxev = np.sqrt(np.dot(dx.T, dx))[0,0]
        if dxev > self.vxyzlmt:
            dx = dx/dxev * self.vxyzlmt
        drpyv = np.sqrt(np.dot(drpy.T, drpy))[0,0]
        if drpyv > self.vrpylmt:
            drpy = drpy/drpyv * self.vrpylmt
        Js = np.dot(self.Ie,self.kin.jacobianAng(self.q))
        #Js = J[0:3,:]
        
        
        Jsp = np.linalg.pinv(Js)
        
        dxyzrpy = np.dot(self.Ie,np.vstack((dx, drpy)))
        
        Jscond = np.linalg.cond(Js)
        if Jscond > 40:
            dxyzrpy = dxyzrpy/Jscond
            
        dq = np.dot(Jsp, dxyzrpy)
        for i in range(0, 7):
            self.q[i] += dq[i,0]/self.rate
            if self.q[i] > self.qmax[i]:
                self.q[i] = self.qmax[i]
            if self.q[i] < self.qmin[i]:
                self.q[i] = self.qmin[i]
                
    #def stop(self):
    #    self.thread_stop = True
###################################################################

class Trajectory(object):
    def __init__(self, limb):
        ns = 'robot/limb/' + limb + '/'
        self._client = actionlib.SimpleActionClient(
            ns + "follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        self._goal = FollowJointTrajectoryGoal()
        self._goal_time_tolerance = rospy.Time(0.01)
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)
        self.clear(limb)

    def add_point(self, positions, time):
        point = JointTrajectoryPoint()
        point.positions = copy(positions)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)

    def stop(self):
        self._client.cancel_goal()

    def wait(self, timeout=15.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def result(self):
        return self._client.get_result()

    def clear(self, limb):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._goal.trajectory.joint_names = [limb + '_' + joint for joint in \
            ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]

######################## Real System ##############################
class CArmController:
    def __init__(self, ArmName, initq, _qmax, _qmin):
        self.name = ArmName
        self.kin = baxter_kinematics(ArmName)
        self.limb = baxter_interface.Limb(ArmName)
        self.gripper = baxter_interface.Gripper(ArmName, CHECK_VERSION);
        self.limb.set_joint_position_speed(0.6)
        self.dqcmd = dict()
        self.qcmd = dict()
        self.q = initq
        # self.q = PyKDL.JntArray(self.kin._num_jnts)
        # if initq[0] < 10:
        #     for i in range(0, 7):
        #         self.q[i] = initq[i]
        # else:
        #     for i in range(0, 7):
        #         self.q[i] = self.kin.joints_to_kdl('positions')[i]

        self.posd = np.mat(self.kin.forward_position_kinematics()).T
        self.rpyd = np.mat([3.14, 0, 3.14]).T
        self.thread = threading.Thread(target=self.Calculate)
        self.thread_stop = False
        self.qmax = _qmax
        self.qmin = _qmin
        self.poen = False
        self.po = np.mat([[0],[0],[0]])
        self.pmino = np.mat([[0],[0],[0]])
        self.pmine = np.mat([[0],[0],[0]])
        self.armp = CParaArmController(ArmName, self.q, _qmax, _qmin)
        self.rate = 100.0
        self.armp.rate = self.rate
        self.valdxr_old = 0
        self.ObsDis = 0.25
        #self.Io = np.mat([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        #self.Io = np.mat([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        self.Ie = np.mat([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        #self.Ie = np.mat([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
        self.Io = np.mat([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
        self.Inow = self.Ie
        self.dxof = 0
        self.dxr2f = 0
        self.dxr4f = 0
        self.alphaf = 0
        self.betaf = 0
        self.beta = 0
        #self.vxyzlmt = 0.5#2.0
        #self.vrpylmt = 1.0#2.0
        self.vxyzlmt = 1.0
        self.vrpylmt = 2.0
        self.pxyz = 25
        self.prpy = 25
        self.kprate = 1.0
        self.armp.vxyzlmt = self.vxyzlmt
        self.armp.vrpylmt = self.vrpylmt
        self.armp.pxyz = self.pxyz
        self.armp.prpy = self.prpy
        self.armp.kprate = self.kprate
	#it decide whether it can avoid to the obstacles
        self.ObsAvoEn =True

        self.joint_states = JointState()
        self.joint_states.position = self.q

    def arrive_begin_pos(self):

        control_rate = rospy.Rate(self.rate)

        p1 = [1.2272, -1.3570, 0.0937, 2.2918, 0.6131, -1.0368, 1.2078]
        limb_interface = baxter_interface.limb.Limb(self.name)
        current_angles = [limb_interface.joint_angle(joint) for joint in limb_interface.joint_names()]

        angle_diff = [p1[i] - current_angles[i] for i in range(len(p1))]
        for i in range(101):
            interp_joint = [angle_diff[j] * 0.01 * i + current_angles[j] for j in range(len(angle_diff))]

            self.qcmd = self.SetJointStates(self.limb, self.limb.joint_names(), interp_joint)
            if self.name == 'right':
                print(interp_joint)
            self.armp.posd = self.posd
            self.armp.rpyd = self.rpyd
            self.armp.Ie = self.Ie
            # self.armp.Calculate1()
            
            self.limb.set_joint_positions(self.qcmd)
            
            control_rate.sleep()


        # traj = Trajectory(self.name)
        # rospy.on_shutdown(traj.stop)
        # # Command Current Joint Positions first
        # limb_interface = baxter_interface.limb.Limb(self.name)
        # current_angles = [limb_interface.joint_angle(joint) for joint in limb_interface.joint_names()]
        # traj.add_point(current_angles, 0.0)
        
        # p1 = [1.2272, -1.3570, 0.0937, 2.2918, 0.6131, -1.0368, 1.2078]
    
        # traj.add_point([x * 0.5 for x in p1], 5)
        # while True:
        #     traj.add_point(p1, 7.5)
        #     # traj.add_point([x * 1.25 for x in p1], 12.0)
        #     traj.start()
        #     traj.wait(15.0)

        #     self.arrived_begin_pos = [limb_interface.joint_angle(joint) for joint in limb_interface.joint_names()]
        #     print(self.arrived_begin_pos)
        #     avg_diff = max([self.arrived_begin_pos[i] - p1[i] for i in range(len(self.arrived_begin_pos))])
        #     print(avg_diff)
        #     if avg_diff > 0.05:
        #         # traj.stop()
        #         traj.clear(self.name)
        #         traj.add_point(self.arrived_begin_pos, 0.0)
        #         traj.add_point(p1, 5)
        #         traj.start()
        #         traj.wait(5.0)
        #     else:
        #         break

        # traj.add_point(p1, 7.5)
        # # traj.add_point([x * 1.25 for x in p1], 12.0)
        # traj.start()
        # traj.wait(10.0)

        self.arrived_begin_pos = p1
        print(self.arrived_begin_pos)
        self.joint_states.position = self.arrived_begin_pos

        # # traj.stop()
        # traj.clear(self.name)
        # print(traj.result())
        # print("Exiting - Joint Begin Complete")

    def start(self):
        self.thread.start()
        
    def GetAllJointPos(self):
        pos = np.mat(np.zeros((12, 3)))
        for i in range(1,12):
            posraw = self.GetJointPos(i)
            pos[i,0] = posraw[0]
            pos[i,1] = posraw[1]
            pos[i,2] = posraw[2]
        return pos
        
    def GetAllJointPosD(self):
        pos = np.mat(np.zeros((12, 3)))
        for i in range(1,12):
            posraw = self.GetJointPosD(i)
            pos[i,0] = posraw[0]
            pos[i,1] = posraw[1]
            pos[i,2] = posraw[2]
        return pos
        
    def GetJointPos(self,idx):
        return self.kin.forward_position_kinematicsp_chain(idx)
        
    def GetJointPosD(self,idx):
        return self.kin.forward_position_kinematicspang_chain(idx,self.q)
        
    def GetCollisionPoints(self):
        posall = self.GetAllJointPosD()
        dmin = 100000
        nmin = 0
        pmin0 = np.mat([[0],[0],[0]])
        pmin1 = np.mat([[0],[0],[0]])
        if not self.poen:
            return (2, pmin0, pmin1)
        for i in range(3, 7):
            (dtemp, ptemp0, ptemp1) = self.Dis(posall[i].T, posall[i+1].T)
            if dtemp < dmin:
                dmin = dtemp
                pmin0 = ptemp0
                pmin1 = ptemp1
                nmin = i
        N = nmin - 1
        if N == -1 or N == 3:
            N = 2
        #if N == 4:
        #    N = 2
        if N == 5:
            N = 4
        return (N, pmin0, pmin1)
    
    def Dis(self, pe0, pe1):
        num = 5
        ke = (pe1-pe0)/num
        dmin = 100000
        pmin0 = np.mat([[0],[0],[0]])
        pmin1 = np.mat([[0],[0],[0]])
        po = self.po
        for i in range(0, num+1):
            pe = pe0 + i*ke
            dtemp = np.dot((po-pe).T, (po-pe))
            if dtemp < dmin:
                dmin = dtemp
                pmin1 = pe
        return (dmin, po, pmin1)
        
    def JointStatesCallback(self, joint_states):
        self.joint_states = joint_states
            
        # print("recv one action")

    def SetJointStates(self, limb, joint_name, joint_position):
        # current_position = limb.joint_angle(joint_name)
        for i in range(0, 7):
            if joint_position[i] > self.qmax[i]:
                joint_position[i] = self.qmax[i]
            if joint_position[i] < self.qmin[i]:
                joint_position[i] = self.qmin[i]

        joint_command = {joint_name[0]: joint_position[0], joint_name[1]: joint_position[1], joint_name[2]: joint_position[2], joint_name[3]: joint_position[3],
                        joint_name[4]: joint_position[4], joint_name[5]: joint_position[5] ,joint_name[6]: joint_position[6]}

        return joint_command
        
    def Calculate(self):
        if self.name == 'right':
            self.arrive_begin_pos()
        
        control_rate = rospy.Rate(self.rate)
        #time = 0
        while not self.thread_stop:
            self.qcmd = self.SetJointStates(self.limb, self.limb.joint_names(), self.joint_states.position)
            # if self.name == 'right':
            #     print(self.joint_states.position)
            self.armp.posd = self.posd
            self.armp.rpyd = self.rpyd
            self.armp.Ie = self.Ie
            # self.armp.Calculate1()
            
            self.limb.set_joint_positions(self.qcmd)
            
            control_rate.sleep()
    
    def stop(self):
        self.thread_stop = True
        #self.armp.stop()


def main():
###################INIT#########################################
    print("Initializing node... ")
    rospy.init_node("baxter_obstacle_avoidance")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting...")
        leftarm.stop()
        rightarm.stop()
        # my_force_receiver.stop()
        if not init_state:
            print("Disabling robot...")
            rs.disable()
    rospy.on_shutdown(clean_shutdown)

    print("Enabling robot... ")
    rs.enable()
#################################################################

    print("Calibrating... ")
    #s0,
    leftarm = CArmController('left', 
                            [0.08, -1, -1.19, 1.94, 0.67, 1.03, 0.5], 
                            [1.7, 1.05, 3.05, 2.61, 3.05, 2.09, 3.05], 
                            [-1.7, -2.14, -3.05, -0.05, -3.05, -1.57, -3.05])

    rightarm = CArmController('right', 
                            [0.08, -1, 1.19, 1.94, -0.67, 1.03, -0.5], 
                            [1.7, 1.05, 3.05, 2.61, 3.05, 2.09, 3.05], 
                            [-1.7, -2.14, -3.05, -0.05, -3.05, -1.57, -3.05])
                            
    rightarm.gripper.calibrate()
    leftarm.gripper.calibrate()
    control_rate = rospy.Rate(10.0)

    print("Start Control... ")
    rightarm.start()
    leftarm.start()
    
    rospy.Subscriber("joint_states", JointState, rightarm.JointStatesCallback)

    rospy.spin()
        
    print("Done.")

if __name__ == '__main__':
    main()

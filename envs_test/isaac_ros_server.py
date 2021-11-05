#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

def joint_states_server(joint_position):
    pub = rospy.Publisher('joint_states', JointState, queue_size=10)
    rospy.init_node('joint_state_publisher')
    rate = rospy.Rate(20) # 10hz
    joint_states = JointState()
    joint_states.header = Header()
    joint_states.header.stamp = rospy.Time.now()
    joint_states.name = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2']
    joint_states.position = joint_position
    joint_states.velocity = []
    joint_states.effort = []
    pub.publish(joint_states)
    rate.sleep()

if __name__ == '__main__':
    try:
        joint_states_server()
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Float32MultiArray, Float32
import threading

class isaac_ros_server():
    def __init__(self) -> None:
        print("Initializing node... ")
        rospy.init_node("isaac_ros_server")
        print("Initializing Done... ")

        self.force = 0.0
        self.force_sensor = Float32()
        rospy.Subscriber("force", Float32, self.ForceCallback)
        self.pub = rospy.Publisher('joint_states', JointState, queue_size=1)

        def thread_job():
            while not self.thread_stop:
                rospy.spin()

        self.thread = threading.Thread(target = thread_job)
        self.thread_stop = False
        self.thread.start()

        def clean_shutdown():
            self.thread_stop = True

        rospy.on_shutdown(clean_shutdown)

    def ForceCallback(self, force_sensor):
        self.force = force_sensor.data

    def joint_states_server(self, joint_position):
        rate = rospy.Rate(100) # 10hz
        joint_states = JointState()
        joint_states.header = Header()
        joint_states.header.stamp = rospy.Time.now()
        joint_states.name = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'lfinger', 'rfinger', "correct_flag"]
        joint_states.position = joint_position
        joint_states.velocity = []
        joint_states.effort = []
        self.pub.publish(joint_states)
        rate.sleep()

if __name__ == '__main__':
    try:
        i_s = isaac_ros_server()
        i_s.joint_states_server()
    except rospy.ROSInterruptException:
        pass
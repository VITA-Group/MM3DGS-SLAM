#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
import numpy

twist_msg_global = TwistStamped()

def callback_twist(twist_msg):
    global twist_msg_global
    twist_msg_global = twist_msg

def callback_pose(pose_msg):
    global twist_msg_global
    # Publish Odom
    odom = Odometry()
    odom.header = pose_msg.header
    # odom.child_frame_id = "base_link"
    odom.pose.pose = pose_msg.pose
    odom.twist.twist = twist_msg_global.twist
    
    odom_pub.publish(odom)



if __name__ == '__main__':
    rospy.init_node('concat_pose_and_twist', anonymous=True)

    odom_pub = rospy.Publisher("/vrpn_client_node/Jackal_Latest/odom", Odometry, queue_size = 1)

    rospy.Subscriber("/vrpn_client_node/Jackal_Latest/twist", TwistStamped, callback_twist)
    rospy.Subscriber("/vrpn_client_node/Jackal_Latest/pose", PoseStamped, callback_pose)
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

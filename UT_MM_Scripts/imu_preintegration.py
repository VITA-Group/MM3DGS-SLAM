#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
import tf
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_matrix
from tf.transformations import quaternion_matrix
from tf.transformations import quaternion_from_matrix
import numpy as np
import time
import csv

bag_folder = "fast_straight"
imu_file = open("imu_"+bag_folder+".csv", "w")
imu_file_writer = csv.writer(imu_file)
imu_file_writer.writerow(["Timestamp","T_11","T_12","T_13","T_14","T_21","T_22","T_23","T_24","T_31","T_32","T_33","T_34","T_41","T_42","T_43","T_44"])

tf_broadcast = tf.TransformBroadcaster()


twist_msg_global = TwistStamped()

position = np.array([0.0, 0.0, 0.0])
velocity = np.array([0.0, 0.0, 0.0])
orientation = np.array([0.0, 0.0, 0.0])
W_T_N = np.eye(4)


done_init = False
dt = 0.010 #100Hz
# dt = 0.100 #10Hz

prev_time = time.time()

def callback_twist(twist_msg):
    global twist_msg_global
    twist_msg_global = twist_msg

def callback_pose(pose_msg):
    global twist_msg_global, W_T_N, done_init, prev_time

    # print("Time difference:", prev_time-time.time())
    # prev_time = time.time()

    if not done_init:
        initial_pose = pose_msg
        W_R_I = quaternion_matrix([pose_msg.pose.orientation.x,pose_msg.pose.orientation.y,pose_msg.pose.orientation.z,pose_msg.pose.orientation.w])
        W_T_I = W_R_I.copy()
        W_T_I[0:3,3] = np.array([pose_msg.pose.position.x,pose_msg.pose.position.y,pose_msg.pose.position.z])
        W_T_N = W_T_I
        done_init = True

    # Publish Odom
    odom = Odometry()
    odom.header = pose_msg.header
    # odom.child_frame_id = "base_link"
    odom.pose.pose = pose_msg.pose
    odom.twist.twist = twist_msg_global.twist

    t_tf = pose_msg.pose.position.x,pose_msg.pose.position.y,pose_msg.pose.position.z
    q_tf = pose_msg.pose.orientation.x,pose_msg.pose.orientation.y,pose_msg.pose.orientation.z,pose_msg.pose.orientation.w

    tf_broadcast.sendTransform(t_tf,q_tf,rospy.Time.now(),"robot","world")

    tf_broadcast.sendTransform(t_tf,q_tf,rospy.Time.now(),"os_sensor","world")
    
    vicon_odom_pub.publish(odom)

def callback_imu(imu_msg):
    global position, velocity, orientation, W_T_N

    lin_accel = np.array([imu_msg.linear_acceleration.x, 0, 0] ) #imu_msg.linear_acceleration.z
    ang_vel = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])
    change_in_position = velocity*dt + 0.5*lin_accel*dt*dt
    velocity += lin_accel*dt
    change_in_orientation = ang_vel*dt

    print("Position:", position)
    print("Velocity:", velocity)
    print("Change in Position:", change_in_position)
    print("Change in Orientation:", change_in_orientation)

    I_R_N = euler_matrix(*change_in_orientation, axes='sxyz')
    I_T_N = I_R_N.copy()
    I_T_N[0:3,3] = change_in_position

    # print("W_T_N", W_T_N)

    W_T_N = np.matmul(W_T_N,I_T_N)

    # print("I_R_N", I_R_N)
    # print("I_T_N", I_T_N)
    # print("W_T_N", W_T_N)

    position += change_in_position
    orientation += change_in_orientation

    # print("Position:", position)
    # print("Orientation:", orientation)
    print("\n")

    W_T_N_quat = W_T_N.copy()
    W_T_N_quat[:3,3] = np.zeros(3)
    quaternions = quaternion_from_matrix(W_T_N_quat)

    odom = Odometry()
    odom.header = imu_msg.header
    odom.header.frame_id = "world"
    # odom.child_frame_id = "static_frame"
    odom.pose.pose.position.x = W_T_N[0,3]
    odom.pose.pose.position.y = W_T_N[1,3]
    odom.pose.pose.position.z = W_T_N[2,3]
    odom.pose.pose.orientation.x = quaternions[0]
    odom.pose.pose.orientation.y = quaternions[1]
    odom.pose.pose.orientation.z = quaternions[2]
    odom.pose.pose.orientation.w = quaternions[3]

    t_tf = W_T_N[0,3], W_T_N[1,3], W_T_N[2,3]
    q_tf = quaternions[0], quaternions[1], quaternions[2], quaternions[3]

    tf_broadcast.sendTransform(t_tf,q_tf,rospy.Time.now(),"microstrain_link","world")

    odom_initial_frame_pub.publish(odom)


    I_T_N_quat = I_T_N.copy()
    I_T_N_quat[:3,3] = np.zeros(3)
    quaternions = quaternion_from_matrix(I_T_N_quat)

    odom = Odometry()
    odom.header = imu_msg.header
    odom.header.frame_id = "relative_frame"
    # odom.child_frame_id = "static_frame"
    odom.pose.pose.position.x = I_T_N[0,3]
    odom.pose.pose.position.y = I_T_N[1,3]
    odom.pose.pose.position.z = I_T_N[2,3]
    odom.pose.pose.orientation.x = quaternions[0]
    odom.pose.pose.orientation.y = quaternions[1]
    odom.pose.pose.orientation.z = quaternions[2]
    odom.pose.pose.orientation.w = quaternions[3]

    t_tf = I_T_N[0,3], I_T_N[1,3], I_T_N[2,3]
    q_tf = quaternions[0], quaternions[1], quaternions[2], quaternions[3]

    odom_relative_frame_pub.publish(odom)

    imu_file_writer.writerow([str(imu_msg.header.stamp.secs)+'.'+'{0:09d}'.format(imu_msg.header.stamp.nsecs),*np.linalg.inv(I_T_N).flatten()])

if __name__ == '__main__':
    rospy.init_node('concat_pose_and_twist', anonymous=True)

    vicon_odom_pub = rospy.Publisher("/vrpn_client_node/Jackal_Latest/odom", Odometry, queue_size = 1)
    odom_initial_frame_pub = rospy.Publisher("/preintegration/odom/initial_frame", Odometry, queue_size = 1)
    odom_relative_frame_pub = rospy.Publisher("/preintegration/odom/relative_frame", Odometry, queue_size = 1)

    rospy.Subscriber("/vrpn_client_node/Jackal_Latest/twist", TwistStamped, callback_twist)
    rospy.Subscriber("/vrpn_client_node/Jackal_Latest/pose", PoseStamped, callback_pose)
    rospy.Subscriber("/microstrain/imu/data", Imu, callback_imu)
    # rospy.Subscriber("/microstrain/nav/filtered_imu/data", Imu, callback_imu) # Change to filtered topic
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    imu_file.close()
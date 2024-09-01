#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy
import cv2
import csv
import os

bridge = CvBridge()

root_dir = os.getcwd()

if not os.path.exists('rgb'):
   os.makedirs('rgb')
if not os.path.exists('depth'):
   os.makedirs('depth')

image = numpy.zeros((1536,2048,3))

bag_folder = "fast_straight"
rgb_file = open(root_dir + '/rgb_'+bag_folder+'.csv',"w")
rgb_writer = csv.writer(rgb_file, delimiter = ",")
rgb_writer.writerow(["Timestamp","Frame"])

depth_file = open(root_dir + '/depth_'+bag_folder+'.csv',"w")
depth_writer = csv.writer(depth_file, delimiter = ",")
depth_writer.writerow(["Timestamp","Frame"])

def callback_rgb_Image(ros_image):
    image = bridge.compressed_imgmsg_to_cv2(ros_image, desired_encoding="bgr8")
    cv2.imwrite(root_dir+"/rgb/"+str(ros_image.header.stamp.secs)+'.'+'{0:09d}'.format(ros_image.header.stamp.nsecs)+".png",image)
    rgb_writer.writerow([str(ros_image.header.stamp.secs)+'.'+'{0:09d}'.format(ros_image.header.stamp.nsecs),"/rgb/"+str(ros_image.header.stamp.secs)+'.'+'{0:09d}'.format(ros_image.header.stamp.nsecs)+".png"])

def callback_depth_Image(ros_depth_image):
    depth_image = bridge.imgmsg_to_cv2(ros_depth_image)
    cv2.imwrite(root_dir+"/depth/"+str(ros_depth_image.header.stamp.secs)+'.'+'{0:09d}'.format(ros_depth_image.header.stamp.nsecs)+".png",depth_image)
    depth_writer.writerow([str(ros_depth_image.header.stamp.secs)+'.'+'{0:09d}'.format(ros_depth_image.header.stamp.nsecs),"/depth/"+str(ros_depth_image.header.stamp.secs)+'.'+'{0:09d}'.format(ros_depth_image.header.stamp.nsecs)+".png"])

def listener():


    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/realsense/color/image_raw/compressed", CompressedImage, callback_rgb_Image)
    rospy.Subscriber("/realsense/depth/image_rect_raw", Image, callback_depth_Image)
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
    cv2.destroyAllWindows()
    rgb_file.close()
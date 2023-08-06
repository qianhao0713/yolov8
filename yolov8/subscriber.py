

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory
import time
import sys,os
sys.path.append("/home/gywrc-s1/xfy/xufengyu_BasePerception_0720_sam/src/msgs")

from perception_msgs.msg import BoundingBox, BoundingBoxes
# from yolo_msgs.msg import BoundingBox, BoundingBoxes

from ultralytics import YOLO
import cv2

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            # CompressedImage,
            # '/image2/compressed',
            Image,
            '/hik03/forwardright/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        self.publisher_ = self.create_publisher(BoundingBoxes, '/yolov8/bounding_boxes', 1)

        package_folder = get_package_share_directory('yolov8')
        model_path = os.path.join(package_folder, "best.pt")
        self.get_logger().info(f'loading model from path : {model_path}')
        self.model = YOLO(model_path)
        self.get_logger().info(f'load model completed !')

    def listener_callback(self, msg):
        # current_frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        current_frame = self.bridge.imgmsg_to_cv2(msg);
        self.get_logger().info(f'get image : {current_frame.shape}')
        start_time = time.time()

        results = self.model(current_frame)
        cost_time = time.time() - start_time
        self.get_logger().info(f'cost_time : {cost_time}')

        bounding_boxes_msg = BoundingBoxes()
        bounding_boxes_msg.header.stamp.sec = msg.header.stamp.sec
        bounding_boxes_msg.header.stamp.nanosec = msg.header.stamp.nanosec
        for result in results:
            boxes = result.boxes
            # print(f"{boxes = }")
            xywh_s = boxes.xywh.detach().cpu()
            object_classes = boxes.cls.detach().cpu()
            # print(f"{xywh_s = }")
            # print(f"{object_classes = }")
            for xyhw, obj_class in zip(xywh_s, object_classes):
                # print(f"{xyhw = }")
                # print(f"{obj_class = }")
                b_box = BoundingBox()
                b_box.x = xyhw[0].item()
                b_box.y = xyhw[1].item()
                b_box.w = xyhw[2].item()
                b_box.h = xyhw[3].item()
                b_box.object_class = int(obj_class.item())
                bounding_boxes_msg.data.append(b_box)
            # print(bounding_boxes_msg.data)
        self.publisher_.publish(
                bounding_boxes_msg)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



import rclpy
from rclpy.node import Node

import time
import os

from yolo_msgs.msg import BoundingBox, BoundingBoxes

class BoundingBoxSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            BoundingBoxes,
            '/yolov8/bounding_boxes',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        for b_box in msg.data:
            print(f"{b_box = }")


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = BoundingBoxSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

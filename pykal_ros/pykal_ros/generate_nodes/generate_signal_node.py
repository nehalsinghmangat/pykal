import rclpy
from rclpy.node import Node
import dill
import os

from pykal_ros.utils.message_translators import MESSAGE_TRANSLATORS, MESSAGE_TYPES


class GenerateSignalNode(Node):
    def __init__(self):
        super().__init__("generate_signal_node")

        # Declare ROS parameters
        self.declare_parameter("signal_path")
        self.declare_parameter("topic_name")
        self.declare_parameter("message_type")
        self.declare_parameter("timer_period", 0.1)

        # Read parameters
        signal_path = (
            self.get_parameter("signal_path").get_parameter_value().string_value
        )
        topic_name = self.get_parameter("topic_name").get_parameter_value().string_value
        message_type = (
            self.get_parameter("message_type").get_parameter_value().string_value
        )
        timer_period = (
            self.get_parameter("timer_period").get_parameter_value().double_value
        )

        # Validate message type
        if message_type not in MESSAGE_TRANSLATORS:
            raise ValueError(
                f"Unsupported message_type: {message_type}. Supported: {list(MESSAGE_TRANSLATORS.keys())}"
            )

        if not os.path.isfile(signal_path):
            raise FileNotFoundError(f"Signal file not found: {signal_path}")

        # Load signal from dill
        with open(signal_path, "rb") as f:
            signal_obj = dill.load(f)

        # Extract user-defined signal function
        self.signal_function = signal_obj.user_defined_signal
        self.msg_type = message_type
        self.msg_converter = MESSAGE_TRANSLATORS[message_type]
        self.msg_class = MESSAGE_TYPES[message_type]

        self.publisher = self.create_publisher(self.msg_class, topic_name, 10)

        self.t_start = self.get_clock().now().seconds_nanoseconds()[0]
        self.timer = self.create_timer(timer_period, self.publish_signal)

        self.get_logger().info(
            f"Publishing {message_type} messages on '{topic_name}' from signal '{signal_path}'"
        )

    def publish_signal(self):
        t_now = self.get_clock().now().nanoseconds / 1e9
        tk = t_now - self.t_start

        try:
            u = self.signal_function(tk)
            msg = self.msg_converter(u)
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(
                f"Failed to generate or publish message at t={tk:.2f}: {e}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = GenerateSignalNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

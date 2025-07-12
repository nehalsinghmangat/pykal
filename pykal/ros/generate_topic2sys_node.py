from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from pykal.system import System
from system_robots import available_systems
import numpy as np


class pykalSystemNode(Node):
    def __init__(self, system: System):
        super().__init__("pykal_system")

        self.sys = system

        self.sim_meas = self.create_publisher(Float64MultiArray, "/sys/meas", 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        # Latest cmd_vel stored for dynamic access via system.u(t)
        self.latest_u = np.zeros((2, 1), dtype=np.float64)

        # Assign dynamic control function to system.u
        def cmd_vel_as_u() -> NDArray:
            return self.latest_u

        self.sys.u = cmd_vel_as_u

    def cmd_vel_callback(self, msg: Twist):
        self.latest_u = np.array([[msg.linear.x], [msg.angular.z]], dtype=np.float64)

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular

        # Optionally convert quaternion to yaw (theta), but for now use raw orientation
        # Simplified assumption: planar motion, only theta from orientation.z and orientation.w
        theta = 2 * np.arctan2(ori.z, ori.w)

        x = np.array([pos.x, pos.y, theta, lin.x, ang.z], dtype=np.float64)
        msg_out = Float64MultiArray(data=x.flatten().tolist())
        self.sim_meas.publish(msg_out)


def main():
    rclpy.init()
    system = available_systems["turtlebot"]

    node = pykalSystemNode(system=system)
    try:
        rclpy.spin(node)
    finally:

        node.destroy_node()
        rclpy.shutdown()

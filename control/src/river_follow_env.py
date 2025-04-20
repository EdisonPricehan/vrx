import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import subprocess
from typing import Optional


class WamvGazeboEnv(gym.Env):
    """Gymnasium Environment for WAMV in Gazebo with ROS2"""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
            self,
            episode_max_step: int = 1000,
            obs_timeout_sec: float = 2.,
            render_mode: Optional[str] = None,
    ):
        super(WamvGazeboEnv, self).__init__()

        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init()

        # Create ROS2 node
        self.node = Node('wamv_gym_wrapper')
        self.bridge = CvBridge()
        self.render_mode = render_mode
        self.episode_max_step: int = episode_max_step
        self.obs_timeout_sec: float = obs_timeout_sec
        self.episode_step: int = 0

        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # Normalized thrust values [-1, 1]
            high=np.array([1.0, 1.0]),
            dtype=np.float32)

        # Observation space is the camera image
        # Assuming 640x480 RGB image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(480, 640, 3),  # Height, Width, Channels
            dtype=np.uint8)

        # Setup ROS2 communications
        self._setup_ros_communications()

        # Initialize state variables
        self.current_image = None
        self.received_image = False

        # For rendering
        if self.render_mode == 'human':
            cv2.namedWindow('WAMV Camera View', cv2.WINDOW_NORMAL)

    def _setup_ros_communications(self):
        """Set up all ROS2 publishers and subscribers"""
        # Subscriber for camera image
        self.image_sub = self.node.create_subscription(
            Image,
            '/wamv/sensors/cameras/front_left_camera_sensor/image_raw',
            self._image_callback,
            10)

        # Publishers for thrust commands
        self.left_thrust_pub = self.node.create_publisher(
            Float64,
            '/wamv/thrusters/left/thrust',
            10)

        self.right_thrust_pub = self.node.create_publisher(
            Float64,
            '/wamv/thrusters/right/thrust',
            10)

        # Optional: Service client for reset if available

    def _image_callback(self, msg):
        """Callback for processing camera images"""
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_image = cv_image
            self.received_image = True

            # Update render if in human mode
            if self.render_mode == 'human':
                self.render()
        except Exception as e:
            self.node.get_logger().error(f'Error processing image: {str(e)}')

    def pub_thrust(self, left: float = 0., right: float = 0.) -> None:
        if left < -1. or left > 1.:
            self.node.get_logger().warn(f'Left thrust cmd needs to be in range [-1, 1], given {left}.')
        if right < -1. or right > 1.:
            self.node.get_logger().warn(f'Right thrust cmd needs to be in range [-1, 1], given {right}.')

        # Publish the thrust commands
        left_thrust = Float64()
        left_thrust.data = float(left)
        self.left_thrust_pub.publish(left_thrust)

        right_thrust = Float64()
        right_thrust.data = float(right)
        self.right_thrust_pub.publish(right_thrust)

    def wait_for_obs(self) -> bool:
        """
        Wait for specified duration until image observation is available.
        """
        self.received_image = False
        start_time = self.node.get_clock().now()
        while not self.received_image:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if (self.node.get_clock().now() - start_time).nanoseconds > self.obs_timeout_sec * 1e9:
                self.node.get_logger().error('Timeout waiting for image during reset.')
                return False

        return True

    def step(self, action: np.array):
        """
        Execute one time step in the environment

        Args:
            action: [left_thrust, right_thrust] normalized between -1 and 1

        Returns:
            observation (image), reward, terminated, truncated, info
        """
        self.episode_step += 1

        # Publish the thrust commands
        self.pub_thrust(left=action[0], right=action[1])

        terminated: bool = False
        truncated: bool = False

        # Wait for new image observation
        if not self.wait_for_obs():
            terminated = True

        # Max episodic steps reached
        if self.episode_step > self.episode_max_step:
            truncated = True

        # Calculate reward (implement your own reward function)
        reward = self._calculate_reward()

        # Additional info (optional)
        info = {
            'left_thrust': action[0],
            'right_thrust': action[1]
        }

        return self.current_image, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state

        Args:
            seed: Optional seed for random number generator
            options: Optional additional options

        Returns:
            observation (image) and info
        """
        # We don't need the seed for this environment
        super().reset(seed=seed)

        # Publish zero thrust commands first
        self.pub_thrust()

        self.episode_step = 0

        # Call reset service
        x = -10
        y = -1000
        yaw = 1.14
        _, _, z, w = Rotation.from_euler('xyz', [0, 0, yaw]).as_quat()
        command = [
            "gz", "service",
            "-s", "/world/wabash/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000",
            "--req", f'name: "wamv", position: {{x: {x}, y: {y}, z: 0}}, orientation: {{x: 0, y: 0, z: {z},w: {w}}}'
        ]

        try:
            # Run the command
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            # Print the result of the service call
            print("Reset successful")
            # print("Output:", result.stdout)
            # print("Error (if any):", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            print("Output:", e.output)
            print("Error:", e.stderr)

        # Wait for new image observation
        if not self.wait_for_obs():
            raise RuntimeError('Failed to receive image observation during reset.')

        info = {}  # Can add reset-specific info here if needed
        return self.current_image, info

    def render(self):
        """Render the environment"""
        if self.render_mode == 'rgb_array':
            return self.current_image
        elif self.render_mode == 'human':
            if self.current_image is not None:
                cv2.imshow('WAMV Camera View', self.current_image)
                cv2.waitKey(1)
            return
        # No return value needed for human mode

    def close(self):
        """Clean up resources"""
        # Publish zero thrust before closing
        zero_thrust = Float64()
        zero_thrust.data = 0.0
        self.left_thrust_pub.publish(zero_thrust)
        self.right_thrust_pub.publish(zero_thrust)

        # Destroy node and shutdown ROS
        self.node.destroy_node()
        rclpy.shutdown()

        # Close any OpenCV windows
        if self.render_mode == 'human':
            cv2.destroyAllWindows()

    def _calculate_reward(self):
        """Implement your custom reward function here"""
        # Example: simple reward for moving forward
        # You'll want to replace this with your actual reward logic
        return 1.0  # Placeholder


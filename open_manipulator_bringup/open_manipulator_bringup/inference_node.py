import logging
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Thread

import cv2
import draccus
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flow.modeling_diffusion import DiffusionPolicy
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


@dataclass
class PolicyConfig:
    policy_path: str | Path
    device: str = "cuda"  # Device to run the policy on, e.g. 'cuda' or 'cpu'.


class PolicyWrapper:
    def __init__(self, policy_config: PolicyConfig):
        self.policy = DiffusionPolicy.from_pretrained(policy_config.policy_path)

        self.device = policy_config.device

    def predict(
        self,
        images: dict[str, np.ndarray],
        state: list[float],
        task_instruction: str = None,
    ) -> list:
        observation = self._preprocess(images, state, task_instruction)
        # with torch.inference_mode():
        #     action = self.policy.select_action(observation)
        #     action = action.squeeze(0).to('cpu').numpy()

        # return action

    def _preprocess(
        self, images: dict[str, np.ndarray], state: list, task_instruction: str = None
    ) -> dict:
        observation = self._convert_images2tensors(images)
        observation["observation.state"] = self._convert_np2tensors(state)
        for key in observation.keys():
            observation[key] = observation[key].to(self.device)

        if task_instruction is not None:
            observation["task"] = [task_instruction]

        return observation

    def _convert_images2tensors(
        self, images: dict[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        processed_images = {}
        for key, value in images.items():
            image = torch.from_numpy(value)
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            image = image.to(self.device, non_blocking=True)
            image = image.unsqueeze(0)
            processed_images["observation.images." + key] = image

        return processed_images

    def _convert_np2tensors(self, data):
        if isinstance(data, list):
            data = np.array(data)
        tensor_data = torch.from_numpy(data)
        tensor_data = tensor_data.to(torch.float32)
        tensor_data = tensor_data.to(self.device, non_blocking=True)
        tensor_data = tensor_data.unsqueeze(0)

        return tensor_data


class Communicator:
    def __init__(self, node: Node):
        self.node = node

        self.camera_topics = [
            "/camera/camera/camera_wrist",
            "/camera/camera/camera_scene1",
            "/camera/camera/camera_scene2",
        ]
        self.joint_state_topic = "/joint_states"
        self.joint_order = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "rh_r1_joint",
        ]

        self.subscribers = {"camera": {}, "joint_state": None}
        self.publisher = None

        self.camera_topic_msgs = {topic: None for topic in self.camera_topics}
        self.joint_state_msg = None

        self.bridge = CvBridge()

        self.init_subscribers()
        self.init_publisher()

    def init_subscribers(self):
        qos_profile = QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        )
        for topic in self.camera_topics:
            self.subscribers["camera"][topic] = self.node.create_subscription(
                CompressedImage,
                topic,
                partial(self._camera_callback, topic=topic),
                qos_profile=qos_profile,
            )
            self.node.get_logger().info(f"Subscribed to {topic} for camera images")

        self.subscribers["joint_state"] = self.node.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_callback,
            qos_profile=qos_profile,
        )
        self.node.get_logger().info(
            f"Subscribed to {self.joint_state_topic} for joint states"
        )

    def init_publisher(self):
        self.publisher = self.node.create_publisher(
            JointTrajectory, "/arm_controller/joint_trajectory", 100
        )

    def _camera_callback(self, msg: CompressedImage, topic: str):
        self.camera_topic_msgs[topic] = msg

    def _joint_state_callback(self, msg: JointState):
        self.joint_state_msg = msg

    def publish_action(self, action: list[float]):
        msg = JointTrajectory(
            joint_names=self.joint_order,
            points=[JointTrajectoryPoint(positions=action)],
        )
        self.publisher.publish(msg)
        # self.node.get_logger().info(f'Published action: {action}')

    def _convert_images2cvmat(
        self, msg: CompressedImage, desired_encoding: str = "rgb8"
    ) -> np.ndarray:
        cv_image = self.bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding=desired_encoding
        )  # Ensure the image is in RGB format
        if cv_image.dtype == np.uint16:
            cv_image = cv2.normalize(
                cv_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        return cv_image

    def _convert_joint_state2tensor_array(self, msg: JointState) -> np.ndarray:
        joint_pos_map = dict(zip(msg.name, msg.position))

        ordered_positions = [joint_pos_map[name] for name in self.joint_order]
        return np.array(ordered_positions, dtype=np.float32)

    def get_latest_data(self):
        images = {}
        # for topic, msg in self.camera_topic_msgs.items():
        #     if msg is not None:
        #         images[topic] = self._convert_images2cvmat(msg)

        state = None
        if self.joint_state_msg is not None:
            state = self._convert_joint_state2tensor_array(self.joint_state_msg)
        return images, state


class InferenceNode(Node):
    def __init__(self, policy: PolicyWrapper):
        super().__init__("inference_node")
        self.policy = policy
        self.communicator = Communicator(self)

    def _inference_callback(self):
        pass


def replay():
    rclpy.init()
    policy = PolicyWrapper(
        PolicyConfig(
            policy_path="/root/ros2_ws/src/lerobot/outputs/unet_ddpm_300k/checkpoints/last/pretrained_model",
            device="cuda",
        )
    )

    node = InferenceNode(policy)
    thread = Thread(target=rclpy.spin, args=(node,))
    thread.start()
    # Wait for the node to initialize
    time.sleep(1)
    dataset = LeRobotDataset("hieu1344/omy_baseline", episodes=[50])
    actions = dataset.hf_dataset.select_columns("action")["action"]
    # actions = dataset.hf_dataset.select_columns("observation.state")['observation.state']
    for action in actions:
        start_episode_t = time.perf_counter()
        node.communicator.publish_action(action.cpu().numpy().tolist())
        dt_s = time.perf_counter() - start_episode_t
        print(node.communicator.get_latest_data()[1], action.cpu().numpy().tolist())
        busy_wait(max(1.0 / dataset.fps - dt_s, 0.0))

    node.destroy_node()
    rclpy.shutdown()


def run_policy():
    rclpy.init()
    policy = PolicyWrapper(
        PolicyConfig(
            policy_path="/root/ros2_ws/src/lerobot/outputs/unet_ddpm_300k/checkpoints/last/pretrained_model",
            device="cuda",
        )
    )
    policy.policy.diffusion.num_inference_steps = 10
    node = InferenceNode(policy)
    thread = Thread(target=rclpy.spin, args=(node,))
    thread.start()

    dataset = LeRobotDataset("hieu1344/omy_baseline", episodes=[1])
    len_dataset = len(
        dataset.hf_dataset.select_columns("observation.state")["observation.state"]
    )

    for i in range(len_dataset):
        start_episode_t = time.perf_counter()
        batch = dataset[i]
        batch = {
            k: v.to("cuda").unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch
        images, state = node.communicator.get_latest_data()
        with torch.inference_mode():
            action = (
                policy.policy.select_action(batch).cpu().numpy().squeeze(0).tolist()
            )
        # _ = policy.predict(images, state)
        node.communicator.publish_action(action)
        dt_s = time.perf_counter() - start_episode_t
        print(f"Step {i + 1}/{len_dataset}, Action: {action}, Time taken: {dt_s:.4f}s")
        # busy_wait(max(1.0 / dataset.fps - dt_s, 0.0))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    # replay()
    run_policy()

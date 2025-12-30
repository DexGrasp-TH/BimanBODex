#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Example computing collisions using curobo"""

# Third Party
import torch
import os
import sys
import trimesh as tm

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

# 添加上级目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.visualizer import Visualizer

if __name__ == "__main__":
    robot_urdf_path = "src/curobo/content/assets/robot/shadow_hand/dual_dummy_arm_shadow.urdf"
    mesh_dir_path = "src/curobo/content/assets/robot/shadow_hand"
    device = "cuda:0"
    visualize = Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path, device=device)

    robot_file = "dual_dummy_arm_shadow.yml"
    world_file = "collision_test.yml"
    tensor_args = TensorDeviceType()
    config = RobotWorldConfig.load_from_config(robot_file, world_file, collision_activation_distance=0.0)
    curobo_fn = RobotWorld(config)

    # q_sph = torch.zeros((10, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    # q_sph[..., 1] = 0.1
    # q_sph[..., 3] = 0.1
    # d = curobo_fn.get_collision_distance(q_sph)
    # print(d)

    q_s = curobo_fn.sample(5, mask_valid=False)
    q_s[0, :] = 0
    q_s[0, -1] = 0.5

    d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(q_s)
    print("Collision Distance:")
    print("World:", d_world)
    print("Self:", d_self)

    # Pass the joint angles to the visualizer
    curobo_joint_names = curobo_fn.kinematics.kinematics_config.joint_names
    pk_joint_names = visualize.joint_names
    joint_cu2pk = [curobo_joint_names.index(name) for name in pk_joint_names]
    robot_pose = torch.zeros((q_s.shape[0], 7 + q_s.shape[1]), dtype=q_s.dtype, device=q_s.device)
    robot_pose[:, 3] = 1.0  # quat w
    robot_pose[:, 7:] = q_s[:, joint_cu2pk]
    visualize.set_robot_parameters(robot_pose)

    # Visualize
    robot_mesh = visualize.get_robot_trimesh_data(i=0)
    scene = tm.Scene(geometry=[robot_mesh])
    scene.show()

    a = 1

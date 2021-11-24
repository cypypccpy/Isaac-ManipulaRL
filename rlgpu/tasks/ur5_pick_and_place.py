# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from einops.einops import rearrange
import numpy as np
import os

from isaacgym import gymtorch
from isaacgym import gymapi
from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask
import torch

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image as Im
import math
from einops.layers.torch import Rearrange, Reduce

class UR5PickAndPlace(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        self.num_obs = 3
        self.num_acts = 3
        self.scale = 1.5

        self.cfg["env"]["numObservations"] = self.num_obs
        self.cfg["env"]["numActions"] = self.num_acts

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.num_dof_end = 12
       
        # Camera Sensor

        super().__init__(cfg=self.cfg)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.ur5_default_dof_pos = to_torch([-1.57, -1.57, 1.57, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.ur5_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur5_dofs]
        self.ur5_dof_pos = self.ur5_dof_state[..., 0]
        self.ur5_dof_vel = self.ur5_dof_state[..., 1]
        # self.cabinet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_ur5_dofs:]
        # self.cabinet_dof_pos = self.cabinet_dof_state[..., 0]
        # self.cabinet_dof_vel = self.cabinet_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.ur5_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 150
        self.sim_params.physx.num_velocity_iterations = 25
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"
        ur5_asset_file = "ur_robotics/ur5_gripper/ur5_gripper_origin.urdf"

        # load ur5 asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.use_mesh_materials = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 500000
        ur5_asset = self.gym.load_asset(self.sim, asset_root, ur5_asset_file, asset_options)

        table_dims = gymapi.Vec3(0.6, 1.0, 0.2)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0, 0, 0.1)

        asset_options = gymapi.AssetOptions()
        asset_options.density = 400
        asset_options.fix_base_link = True
        asset_options.thickness = 0.0001
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # create box asset
        box_size = 0.045
        asset_options = gymapi.AssetOptions()
        asset_options.density = 400
        box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, asset_options)

        ur5_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4], dtype=torch.float, device=self.device)
        ur5_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)
        ur5_dof_lower_limit = to_torch([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0.0, -3.14, -3.14, -3.14], dtype=torch.float, device=self.device)
        ur5_dof_upper_limit = to_torch([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 0.8, 3.14, 3.14], dtype=torch.float, device=self.device)

        self.num_ur5_bodies = self.gym.get_asset_rigid_body_count(ur5_asset)
        self.num_ur5_dofs = self.gym.get_asset_dof_count(ur5_asset)

        print("num ur5 bodies: ", self.num_ur5_bodies)
        print("num ur5 dofs: ", self.num_ur5_dofs)

        # set ur5 dof properties
        ur5_dof_props = self.gym.get_asset_dof_properties(ur5_asset)
        self.ur5_dof_lower_limits = []
        self.ur5_dof_upper_limits = []
        for i in range(self.num_ur5_dofs):
            ur5_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                ur5_dof_props['stiffness'][i] = ur5_dof_stiffness[i]
                ur5_dof_props['damping'][i] = ur5_dof_damping[i]
            else:
                ur5_dof_props['stiffness'][i] = 7000.0
                ur5_dof_props['damping'][i] = 50.0

            self.ur5_dof_lower_limits.append(ur5_dof_lower_limit[i])
            self.ur5_dof_upper_limits.append(ur5_dof_upper_limit[i])
            # self.ur5_dof_lower_limits.append(ur5_dof_props['lower'][i])
            # self.ur5_dof_upper_limits.append(ur5_dof_props['upper'][i])

        self.ur5_dof_lower_limits = to_torch(self.ur5_dof_lower_limits, device=self.device)
        self.ur5_dof_upper_limits = to_torch(self.ur5_dof_upper_limits, device=self.device)
        self.ur5_dof_speed_scales = torch.ones_like(self.ur5_dof_lower_limits)

        # create prop assets

        ur5_start_pose = gymapi.Transform()
        ur5_start_pose.p = gymapi.Vec3(0.6, 0.2, 0.1)
        ur5_start_pose.r = gymapi.Quat.from_euler_zyx(-1 * math.pi, 0.5 * math.pi, 0.5 * math.pi)

        box_pose = gymapi.Transform()

        # compute aggregate size
        num_ur5_bodies = self.gym.get_asset_rigid_body_count(ur5_asset)
        num_ur5_shapes = self.gym.get_asset_rigid_shape_count(ur5_asset)
        max_agg_bodies = num_ur5_bodies
        max_agg_shapes = num_ur5_shapes

        self.ur5s = []
        self.bases = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []
        self.camera_handles = []
        self.shafts = []
        self.shaft_poses = []
        self.lfinger_poses = []
        self.rfinger_poses = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            ur5_actor = self.gym.create_actor(env_ptr, ur5_asset, ur5_start_pose, "ur5", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, ur5_actor, ur5_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)

            # add box
            box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
            box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
            box_pose.p.z = table_dims.z + 0.5 * box_size
            box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            box_handle = self.gym.create_actor(env_ptr, box_asset, box_pose, "box", i, 0, 0)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.ur5s.append(ur5_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur5_actor, "wrist_3_link")
        self.wrist_2_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur5_actor, "wrist_2_link")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur5_actor, "robotiq_85_left_finger_tip_link")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur5_actor, "robotiq_85_right_finger_tip_link")
        self.lfinger_inner_knuckle_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur5_actor, "robotiq_85_left_inner_knuckle_link")
        self.rfinger_inner_knuckle_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur5_actor, "robotiq_85_right_inner_knuckle_link")

        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur5")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)
        self.hand_index = self.gym.get_asset_rigid_body_dict(ur5_asset)["wrist_3_link"]
        self.j_eef = self.jacobian[:, self.hand_index - 1, :]

        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur5s[0], "wrist_3_link")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur5s[0], "robotiq_85_left_finger_tip_link")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur5s[0], "robotiq_85_right_finger_tip_link")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        ur5_local_grasp_pose = hand_pose_inv * finger_pose
        ur5_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        self.ur5_local_grasp_pos = to_torch([ur5_local_grasp_pose.p.x, ur5_local_grasp_pose.p.y,
                                                ur5_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.ur5_local_grasp_rot = to_torch([ur5_local_grasp_pose.r.x, ur5_local_grasp_pose.r.y,
                                                ur5_local_grasp_pose.r.z, ur5_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))


        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))

        self.ur5_grasp_pos = torch.zeros_like(self.ur5_local_grasp_pos)
        self.ur5_grasp_rot = torch.zeros_like(self.ur5_local_grasp_rot)
        self.ur5_grasp_rot[..., -1] = 1  # xyzw

        self.ur5_lfinger_pos = torch.zeros_like(self.ur5_local_grasp_pos)
        self.ur5_rfinger_pos = torch.zeros_like(self.ur5_local_grasp_pos)
        self.ur5_lfinger_rot = torch.zeros_like(self.ur5_local_grasp_rot)
        self.ur5_rfinger_rot = torch.zeros_like(self.ur5_local_grasp_rot)
        self.shaft_pos = torch.zeros_like(self.ur5_local_grasp_pos)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_ur5_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.ur5_grasp_pos,
            self.shaft_tail[:, 0:3],
            self.shaft_tail_euler_angle,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.scale, self.max_episode_length
        )

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        hand_linear_vel = self.rigid_body_states[:, self.hand_handle][:, 7:10]

        self.ur5_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.ur5_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.ur5_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.ur5_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        self.lfinger_inner_knuckle_pos = self.rigid_body_states[:, self.lfinger_inner_knuckle_handle][:, 0:3]
        self.rfinger_inner_knuckle_pos = self.rigid_body_states[:, self.rfinger_inner_knuckle_handle][:, 0:3]
        self.lfinger_inner_knuckle_rot = self.rigid_body_states[:, self.lfinger_inner_knuckle_handle][:, 3:7]
        self.rfinger_inner_knuckle_rot = self.rigid_body_states[:, self.rfinger_inner_knuckle_handle][:, 3:7]

        # self.rigid_body_states[:, self.shaft_handle][:, :] = - (self.rigid_body_states[:, self.rfinger_inner_knuckle_handle] + self.rigid_body_states[:, self.lfinger_inner_knuckle_handle]) / 2.0 * 0.4 + (self.rigid_body_states[:, self.lfinger_handle] + self.rigid_body_states[:, self.lfinger_handle]) / 2.0 * 1.4

        self.shaft_tail = self.rigid_body_states[:, self.hand_handle].clone()
        for i in range(self.num_envs):
            self.shaft_tail[:, 0:3][i] += quat_apply(self.rigid_body_states[:, self.hand_handle][:, 3:7][i], to_torch([1, 0, 0], device=self.device) * 0.03)
            self.shaft_tail[:, 0:3][i] += quat_apply(self.rigid_body_states[:, self.hand_handle][:, 3:7][i], to_torch([0, 1, 0], device=self.device) * 0.26)
            self.shaft_tail[:, 0:3][i] += quat_apply(self.rigid_body_states[:, self.hand_handle][:, 3:7][i], to_torch([0, 0, 1], device=self.device) * 0.16)

        # compute euler angle
        self.shaft_tail_euler_angle = torch.zeros(self.num_envs, 3)
        for i in range(self.num_envs):
            shaft_tail_quat = gymapi.Quat(self.shaft_tail[i, 3], self.shaft_tail[i, 4], self.shaft_tail[i, 5], self.shaft_tail[i, 6])
            euler_angle = gymapi.Quat.to_euler_zyx(shaft_tail_quat)
            self.shaft_tail_euler_angle[i, 0] = euler_angle[0]
            self.shaft_tail_euler_angle[i, 1] = euler_angle[1]
            self.shaft_tail_euler_angle[i, 2] = euler_angle[2]

        dof_pos_scaled = (2.0 * (self.ur5_dof_pos - self.ur5_dof_lower_limits)
                          / (self.ur5_dof_upper_limits - self.ur5_dof_lower_limits) - 1.0)

        # num: 12 + 12 + 3 + 1 + 1
        # hand_pos = hand_pos - to_torch([0.2075, -0.1989, 0.4304], dtype=torch.float, device=self.device)
        # self.obs_buf = torch.cat((self.shaft_tail[:, 0:3], to_target.unsqueeze(-1)), dim=-1)
        self.obs_buf = self.shaft_tail[:, 0:3]

        return self.obs_buf

    def reset(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset ur5
        pos = tensor_clamp(
            self.ur5_default_dof_pos.unsqueeze(0),
            self.ur5_dof_lower_limits, self.ur5_dof_upper_limits)
        self.ur5_dof_pos[env_ids, :] = pos
        self.ur5_dof_vel[env_ids, :] = torch.zeros_like(self.ur5_dof_vel[env_ids])
        self.ur5_dof_targets[env_ids, :self.num_ur5_dofs] = pos

        # reset base

        multi_env_ids_int32 = self.global_indices[env_ids, :1].flatten()
        
        # pos_err = torch.rand((self.num_envs, 3), device=self.device) * 4 - 2
        # # pos_err[:, 2] = (self.actions[:, 2] - 1) / 0.5 * self.dt * self.action_scale

        # orn_err = to_torch([0, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # # solve damped least squares
        # j_eef_T = torch.transpose(self.j_eef, 1, 2)
        # d = 0.05  # damping term
        # lmbda = torch.eye(6).to('cpu') * (d ** 2)
        # u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6, 1)

        # # update position targets
        # self.ur5_dof_targets[:, :self.num_ur5_dofs] = self.ur5_dof_targets[:, :self.num_ur5_dofs] + u.squeeze(-1)


        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ur5_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # self.actions = to_torch([0, 0, -1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        # pos_cur = self.rigid_body_states[:, self.hand_handle][:, :3]
        orn_cur = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        # from action's euler angle to quat
        orn_des_quat = torch.zeros_like(orn_cur)
        for i in range(self.num_envs):
            orn_tem = gymapi.Quat.from_euler_zyx(0, 0, 0)
            orn_des_quat[i, :] = to_torch([orn_tem.w, orn_tem.x, orn_tem.y, orn_tem.z], dtype=torch.float, device=self.device)

        # pos_des = self.actions[:, :3] * self.dt * self.action_scale + self.rigid_body_states[:, self.hand_handle][:, :3]

        # orn_des = orn_des_quat[:, :] * self.dt * self.action_scale + self.rigid_body_states[:, self.hand_handle][:, 3:7]
        # pos_des = self.rigid_body_states[:, self.hand_handle][:, :3]
        orn_des = orn_des_quat[:, :]

        orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
        self.orn_err = orientation_error(orn_des, orn_cur)

        # pos_err = (pos_des - pos_cur)
        # pos_err = to_torch([-0.01, -0.01, -0.1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        pos_err = self.actions[:, :3] * self.dt * self.action_scale
        pos_err[:, 2] = (self.actions[:, 2] - 0.5) / 1.0 * self.dt * self.action_scale

        orn_err = to_torch([0, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        d = 0.05  # damping term
        lmbda = torch.eye(6).to('cuda:0') * (d ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 12, 1)

        # update position targets
        self.ur5_dof_targets[:, :self.num_ur5_dofs] = self.ur5_dof_targets[:, :self.num_ur5_dofs] + u.squeeze(-1)

        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.ur5_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            #self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                # px = (self.rigid_body_states[:, self.hand_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.hand_handle][:, 3:7][i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.rigid_body_states[:, self.hand_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.hand_handle][:, 3:7][i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.rigid_body_states[:, self.hand_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.hand_handle][:, 3:7][i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                # p0 = self.rigid_body_states[:, self.hand_handle][:, 0:3][i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # px = (self.rigid_body_states[:, self.wrist_2_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.wrist_2_handle][:, 3:7][i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.rigid_body_states[:, self.wrist_2_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.wrist_2_handle][:, 3:7][i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.rigid_body_states[:, self.wrist_2_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.wrist_2_handle][:, 3:7][i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                # p0 = self.rigid_body_states[:, self.wrist_2_handle][:, 0:3][i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.base_entry[:, 0:3][i] + quat_apply(self.base_entry[:, 3:7][i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.base_entry[:, 0:3][i] + quat_apply(self.base_entry[:, 3:7][i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.base_entry[:, 0:3][i] + quat_apply(self.base_entry[:, 3:7][i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.base_entry[:, 0:3][i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.shaft_tail[:, 0:3][i] + quat_apply(self.shaft_tail[:, 3:7][i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.shaft_tail[:, 0:3][i] + quat_apply(self.shaft_tail[:, 3:7][i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.shaft_tail[:, 0:3][i] + quat_apply(self.shaft_tail[:, 3:7][i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.shaft_tail[:, 0:3][i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])
#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ur5_reward(
    reset_buf, progress_buf, actions,
    ur5_grasp_pos, 
    shaft_tail_pos,
    shaft_tail_euler_angle,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, scale, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # distance from hand to the drawer
    # d = torch.norm(shaft_tail_pos - base_entry_pos, p=2, dim=-1)
    # dist_reward = 1.0 / (1.0 + d ** 2)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    rewards =  - action_penalty_scale * action_penalty

    # rewards = torch.where(torch.abs(shaft_tail_pos[:, 0]) > 0.2,
    #                       torch.ones_like(rewards) * -2.5, rewards)

    # reset_buf = torch.where(torch.abs(shaft_tail_pos[:, 0]) > 0.2,
    #                         torch.ones_like(reset_buf), reset_buf)

    # rewards = torch.where(torch.abs(shaft_tail_pos[:, 1]) > 0.2,
    #                       torch.ones_like(rewards) * -2.5, rewards)

    # reset_buf = torch.where(torch.abs(shaft_tail_pos[:, 1]) > 0.2,
    #                         torch.ones_like(reset_buf), reset_buf)

    # rewards = torch.where(torch.abs(shaft_tail_pos[:, 2]) > 0.4,
    #                       torch.ones_like(rewards) * -2.5, rewards)

    # reset_buf = torch.where(torch.abs(shaft_tail_pos[:, 2]) > 0.4,
    #                         torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, ur5_local_grasp_rot, ur5_local_grasp_pos,
                             drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_ur5_rot, global_ur5_pos = tf_combine(
        hand_rot, hand_pos, ur5_local_grasp_rot, ur5_local_grasp_pos)
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

    return global_ur5_rot, global_ur5_pos, global_drawer_rot, global_drawer_pos

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
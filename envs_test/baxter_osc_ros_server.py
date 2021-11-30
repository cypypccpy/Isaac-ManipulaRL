#!/usr/bin/env python3
"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


baxter Operational Space Control
----------------
Operational Space Control of baxter robot to demonstrate Jacobian and Mass Matrix Tensor APIs
"""

from os import POSIX_FADV_WILLNEED, device_encoding
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
from torch._C import _jit_pass_onnx_block

from isaac_ros_server import joint_states_server


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def set_hand_pose(x, y, z, q1, q2, q3, q4):
    pos_ = init_pos.clone()
    orn_ = init_orn.clone()
    pos_[:, 0] = x
    pos_[:, 1] = y
    pos_[:, 2] = z
    orn_[:, 0] = q1
    orn_[:, 1] = q2
    orn_[:, 2] = q3
    orn_[:, 3] = q4
    return pos_, orn_

# Parse arguments
args = gymutil.parse_arguments(description="baxter Tensor OSC Example",
                               custom_parameters=[
                                   {"name": "--pos_control", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Trace circular path in XZ plane"},
                                   {"name": "--orn_control", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Send random orientation commands"}])

# Initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    raise Exception("Failed to create sim")

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Load baxter asset
asset_root = "/home/lohse/isaac_ws/src/isaac-gym/scripts/Isaac-drlgrasp/assets"
baxter_asset_file = "baxter/baxter_isaac.urdf"
cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

# load baxter asset
asset_options = gymapi.AssetOptions()
asset_options.flip_visual_attachments = True
asset_options.fix_base_link = True
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = True
asset_options.thickness = 0.001
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.use_mesh_materials = True
# asset_options.vhacd_enabled = True
# asset_options.vhacd_params.resolution = 300000
# asset_options.vhacd_params.max_convex_hulls = 10
# asset_options.vhacd_params.max_num_vertices_per_ch = 64
baxter_asset = gym.load_asset(sim, asset_root, baxter_asset_file, asset_options)

print("Loading asset '%s' from '%s'" % (baxter_asset_file, asset_root))
baxter_asset = gym.load_asset(
    sim, asset_root, baxter_asset_file, asset_options)

# load cabinet asset
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.armature = 0.005
cabinet_asset = gym.load_asset(sim, asset_root, cabinet_asset_file, asset_options)

# get joint limits and ranges for baxter
baxter_dof_props = gym.get_asset_dof_properties(baxter_asset)
baxter_lower_limits = baxter_dof_props['lower']
baxter_upper_limits = baxter_dof_props['upper']
baxter_ranges = baxter_upper_limits - baxter_lower_limits
baxter_mids = 0.5 * (baxter_upper_limits + baxter_lower_limits)
baxter_num_dofs = len(baxter_dof_props)

# set default DOF states
default_dof_state = np.zeros(baxter_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = baxter_mids
# default_dof_state["pos"][10:19] = baxter_mids[10:19]

# set DOF control properties (except grippers)
baxter_dof_props["driveMode"][:17].fill(gymapi.DOF_MODE_POS)
# baxter_dof_props["stiffness"][:17].fill(0.0)
# baxter_dof_props["damping"][:17].fill(0.0)

# # set DOF control properties for grippers
baxter_dof_props["driveMode"][17:].fill(gymapi.DOF_MODE_POS)
baxter_dof_props["stiffness"][17:].fill(800.0)
baxter_dof_props["damping"][17:].fill(40.0)

# set cabinet dof properties
num_cabinet_dofs = gym.get_asset_dof_count(cabinet_asset)
cabinet_dof_props = gym.get_asset_dof_properties(cabinet_asset)
for i in range(num_cabinet_dofs):
    cabinet_dof_props['damping'][i] = 10.0

# Set up the env grid
num_envs = 1
num_per_row = int(math.sqrt(num_envs))
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# default baxter pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(1.4, 0, 1.0)
pose.r = gymapi.Quat(0, 0, 1, 0)
default_dof_pos = to_torch([0, 0, -1.57, 0, 2.5, 0, 0, 0, 0, 0, 0, -1.57, 0, 2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cpu')
# dof_state_tensor = gym.acquire_dof_state_tensor(sim)
# dof_state = gymtorch.wrap_tensor(dof_state_tensor)
# baxter_dof_state = dof_state.view(num_envs, -1, 2)[:, :19]
# baxter_dof_pos = baxter_dof_state[..., 0]
# baxter_dof_vel = baxter_dof_state[..., 1]
# print(baxter_dof_pos)

cabinet_start_pose = gymapi.Transform()
cabinet_start_pose.p = gymapi.Vec3(*get_axis_params(1.1, 2))

print("Creating %d environments" % num_envs)

envs = []
hand_idxs = []
init_pos_list = []
init_orn_list = []

for i in range(num_envs):
    # Create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # Add baxter
    baxter_handle = gym.create_actor(env, baxter_asset, pose, "baxter", i, 1)

    # Set initial DOF states
    gym.set_actor_dof_states(env, baxter_handle, default_dof_state, gymapi.STATE_ALL)

    # Set DOF control properties
    gym.set_actor_dof_properties(env, baxter_handle, baxter_dof_props)

    # Get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, baxter_handle, "right_wrist")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_orn_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # Get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, baxter_handle, "right_wrist", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    cabinet_actor = gym.create_actor(env, cabinet_asset, cabinet_start_pose, "cabinet", i, 2, 0)
    gym.set_actor_dof_properties(env, cabinet_actor, cabinet_dof_props)
    

# Point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 3)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API to access and control the physics simulation
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3)
init_orn = torch.Tensor(init_orn_list).view(num_envs, 4)

# desired hand positions and orientations
pos_des = init_pos.clone()
orn_des = init_orn.clone()

# Prepare jacobian tensor
# For baxter, tensor shape is (num_envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "baxter")
jacobian = gymtorch.wrap_tensor(_jacobian)

# Jacobian entries for end effector
hand_index = gym.get_asset_rigid_body_dict(baxter_asset)["right_wrist"]
j_eef = jacobian[:, hand_index - 1, :]
#j_eef = jacobian[:, hand_index - 1, :][:, :, 10:19]
# Prepare mass matrix tensor
# For baxter, tensor shape is (num_envs, 9, 9)
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "baxter")
mm = gymtorch.wrap_tensor(_massmatrix)
#mm = mm[:, 10:19, 10:19]

kp = 5
kv = 2 * math.sqrt(kp)

# Rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# DOF state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
# dof_vel = dof_states[:, 1][10:19].view(num_envs, 9, 1)
# dof_pos = dof_states[:, 0][10:19].view(num_envs, 9, 1)
dof_vel = dof_states[:, 1][:19].view(num_envs, 19, 1)
dof_pos = dof_states[:, 0][:19].view(num_envs, 19, 1)
dof_cabinet_pos = dof_states[:, 0][19:].view(num_envs, -1, 1)

itr = 0

gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(default_dof_pos))

def open_close_gripper(env, is_open):
    # dof_tensor = gym.get_dof_target_position(env, 17)
    # print(dof_tensor)
    if is_open:
        gym.set_dof_target_position(env, 17, 0.02)
        gym.set_dof_target_position(env, 18, -0.02)
        dof_tensor = gym.get_dof_target_position(env, 17)
        if abs(dof_tensor - 0.02) < 0.000001:
            return True
        else:
            return False
    else:
        gym.set_dof_target_position(env, 17, 0.)
        gym.set_dof_target_position(env, 18, 0.)
        dof_tensor = gym.get_dof_target_position(env, 17)
        if abs(dof_tensor - 0.) < 0.000001:
            return True
        else:
            return False

pos_des_list = []
orn_des_list = []
pos_des_list.append(init_pos.clone())
orn_des_list.append(init_pos.clone())
pose_index = 1

for i in range(20):
    pos, orn = set_hand_pose(0.8 - 0.01 * i, 0.0, 1.44, -0.5, -0.5, 0.5, 0.5)
    pos_des_list.append(pos)
    orn_des_list.append(orn)
for i in range(40):
    pos, orn = set_hand_pose(0.61 + 0.01 * i, 0.0, 1.44, -0.5, -0.5, 0.5, 0.5)
    pos_des_list.append(pos)
    orn_des_list.append(orn)

gripper_executed = True
is_gripper_open = 0

flag = to_torch([is_gripper_open], dtype=torch.float, device='cpu')
dof_pos_flag = torch.cat((dof_pos, flag.unsqueeze(-1).unsqueeze(0), dof_cabinet_pos), dim=1)
dof_pos_np = dof_pos_flag.squeeze(0)
actor_indexed = to_torch([10, 11, 12, 13, 14, 15, 16], device='cpu')

cabinet_u = to_torch([0, 0, 0, 0], device='cpu')
while not gym.query_viewer_has_closed(viewer):

    # Randomize desired hand orientations
    if itr % 250 == 0 and args.orn_control:
        orn_des = torch.rand_like(orn_des)
        orn_des /= torch.norm(orn_des)

    itr += 1

    # Update jacobian and mass matrix
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_actor_root_state_tensor(sim)

    flag = to_torch([is_gripper_open], dtype=torch.float, device='cpu')
    dof_pos_flag = torch.cat((dof_pos, flag.unsqueeze(-1).unsqueeze(0), dof_cabinet_pos), dim=1)
    dof_pos_np = torch.cat((dof_pos_np, dof_pos_flag.squeeze(0)), dim=1)

    joint_position = dof_pos[0, 10:17, 0].detach().numpy().tolist()

    joint_states_server(joint_position)
    # Get current hand poses
    pos_cur = rb_states[hand_idxs, :3]
    orn_cur = rb_states[hand_idxs, 3:7]

    # Set desired hand positions
    if args.pos_control:
        pos_des = pos_des_list[pose_index]
        orn_des = orn_des_list[pose_index]

    orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
    orn_err = orientation_error(orn_des, orn_cur)

    pos_err = (pos_des - pos_cur)

    if not args.pos_control:
        pos_err *= 0

    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    d = 0.1  # damping term
    lmbda = torch.eye(6).to('cpu') * (d ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 19, 1)

    # update position targets
    pos_target = dof_pos + u

    if is_gripper_open == 0:
        gripper_dof = to_torch([0, 0], device='cpu')
    if is_gripper_open == 1:
        gripper_dof = to_torch([0.02, -0.02], device='cpu')

    if 60 >= pose_index >= 20:
        dof_cabinet_pos[:, 3, :] += 0.0000

    gripper_cabinet_dof = torch.cat((gripper_dof.unsqueeze(0).unsqueeze(-1), dof_cabinet_pos), dim = 1)
    pos_target = torch.cat((pos_target[:, :17, :], gripper_cabinet_dof), dim = 1)

    # Set tensor action
    if gripper_executed == True and itr % 1 == 0:
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_target))

    if torch.abs(dpose).sum() < 0.01:
        if pose_index == 1:
            #gripper_executed = open_close_gripper(envs[0], 1)
            is_gripper_open = 1

        if pose_index == 20:
            #gripper_executed = open_close_gripper(envs[0], 0)
            is_gripper_open = 0
            
        if pose_index < len(pos_des_list) - 1:
            pose_index += 1

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

print("Done")
print(dof_pos_np.shape)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Kuka bin perfromance test
-------------------------------
Test simulation perfromance and stability of the robotic arm dealing with a set of complex objects in a bin.
"""

from __future__ import print_function, division, absolute_import

import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from rlgpu.utils.torch_jit_utils import *

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image as Im
from copy import copy

axes_geom = gymutil.AxesGeometry(0.1)

sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

colors = [gymapi.Vec3(1.0, 0.0, 0.0),
          gymapi.Vec3(1.0, 127.0/255.0, 0.0),
          gymapi.Vec3(1.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 0.0, 1.0),
          gymapi.Vec3(39.0/255.0, 0.0, 51.0/255.0),
          gymapi.Vec3(139.0/255.0, 0.0, 1.0)]

tray_color = gymapi.Vec3(0.24, 0.35, 0.8)
banana_color = gymapi.Vec3(0.85, 0.88, 0.2)
brick_color = gymapi.Vec3(0.9, 0.5, 0.1)


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Kuka Bin Test",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
        {"name": "--num_objects", "type": int, "default": 1, "help": "Number of objects in the bin"},
        {"name": "--object_type", "type": int, "default": 0, "help": "Type of bjects to place in the bin: 0 - box, 1 - meat can, 2 - banana, 3 - mug, 4 - brick, 5 - random"}])

num_envs = args.num_envs
num_objects = args.num_objects
box_size = 0.05

# configure sim
sim_type = args.physics_engine
sim_params = gymapi.SimParams()
if sim_type == gymapi.SIM_FLEX:
    sim_params.substeps = 4
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 20
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
elif sim_type == gymapi.SIM_PHYSX:
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 25
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.rest_offset = 0.001
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity.x = 0
    sim_params.gravity.y = 0
    sim_params.gravity.z = -9.81
    #sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer_props = gymapi.CameraProperties()
viewer_props.horizontal_fov = 75.0
viewer_props.width = 1920
viewer_props.height = 1080
viewer = gym.create_viewer(sim, viewer_props)
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load assets
asset_root = "../assets"

table_dims = gymapi.Vec3(0.6, 0.8, 1.0)
base_dims = gymapi.Vec3(0.2, 0.2, 0.2)

baxter_pose = gymapi.Transform()
baxter_pose.p = gymapi.Vec3(0, 1.0, 0.0)
baxter_pose.r = gymapi.Quat.from_euler_zyx(-0.5 * math.pi, 0, 0)

asset_options = gymapi.AssetOptions()
asset_options.armature = 0.001
asset_options.fix_base_link = True
asset_options.thickness = 0.002

asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.5 * table_dims.y + 0.001, 0.0)
base_pose = gymapi.Transform()
base_pose.p = gymapi.Vec3(0.0, 0.5 * base_dims.y, 0.0)

bin_pose = gymapi.Transform()
bin_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

object_pose = gymapi.Transform()

table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# load assets of objects in a bin
asset_options.fix_base_link = False

lower = gymapi.Vec3(-1.5, -1.5, 0.0)
upper = gymapi.Vec3(1.5, 1.5, 1.5)

asset_root = "../assets"
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

# load cabinet asset
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = False
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
asset_options.armature = 0.005
cabinet_asset = gym.load_asset(sim, asset_root, cabinet_asset_file, asset_options)

baxter_default_dof_pos = to_torch([0, 0, -1.57, 0, 2.5, 0, 0, 0, 0, 0, 0, -1.57, 0, 2.5, 0, 0, 0, 0, 0], device='cuda:0')
baxter_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 400, 400, 400, 400, 400, 400, 400], dtype=torch.float, device='cuda:0')
baxter_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 80, 80, 80, 80, 80, 80, 80], dtype=torch.float, device='cuda:0')
baxter_dof_lower_limit = to_torch([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14], dtype=torch.float, device='cuda:0')
baxter_dof_upper_limit = to_torch([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14], dtype=torch.float, device='cuda:0')

num_baxter_bodies = gym.get_asset_rigid_body_count(baxter_asset)
num_baxter_dofs = gym.get_asset_dof_count(baxter_asset)
num_cabinet_bodies = gym.get_asset_rigid_body_count(cabinet_asset)
num_cabinet_dofs = gym.get_asset_dof_count(cabinet_asset)

print("num baxter bodies: ", num_baxter_bodies)
print("num baxter dofs: ", num_baxter_dofs)
print("num cabinet bodies: ", num_cabinet_bodies)
print("num cabinet dofs: ", num_cabinet_dofs)

# set baxter dof properties
baxter_dof_props = gym.get_asset_dof_properties(baxter_asset)
baxter_dof_lower_limits = []
baxter_dof_upper_limits = []
for i in range(num_baxter_dofs):
    baxter_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
    if sim_type == gymapi.SIM_PHYSX:
        baxter_dof_props['stiffness'][i] = baxter_dof_props['stiffness'][i]
        baxter_dof_props['damping'][i] = baxter_dof_props['damping'][i]
    else:
        baxter_dof_props['stiffness'][i] = baxter_dof_props['stiffness'][i]
        baxter_dof_props['damping'][i] = baxter_dof_props['damping'][i]

    baxter_dof_lower_limits.append(baxter_dof_props['lower'][i])
    baxter_dof_upper_limits.append(baxter_dof_props['upper'][i])
    # baxter_dof_lower_limits.append(baxter_dof_props['lower'][i])
    # baxter_dof_upper_limits.append(baxter_dof_props['upper'][i])

baxter_mids = 0.5 * (baxter_dof_props['lower'] + baxter_dof_props['lower'])
baxter_dof_lower_limits = to_torch(baxter_dof_lower_limits, device='cuda:0')
baxter_dof_upper_limits = to_torch(baxter_dof_upper_limits, device='cuda:0')
baxter_dof_speed_scales = torch.ones_like(baxter_dof_lower_limits)
baxter_dof_speed_scales[[17, 18]] = 0.1
# baxter_dof_props['effort'][17] = 200
# baxter_dof_props['effort'][18] = 200

# set cabinet dof properties
cabinet_dof_props = gym.get_asset_dof_properties(cabinet_asset)
for i in range(num_cabinet_dofs):
    cabinet_dof_props['damping'][i] = 10.0

# create prop assets
box_opts = gymapi.AssetOptions()
box_opts.density = 400
prop_asset = gym.create_box(sim, 0.08, 0.08, 0.08, box_opts)

baxter_start_pose = gymapi.Transform()
baxter_start_pose.p = gymapi.Vec3(1.4, 0.0, 1.0)
baxter_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

cabinet_start_pose = gymapi.Transform()
cabinet_start_pose.p = gymapi.Vec3(*get_axis_params(1.1, 2))

print("Creating %d environments" % num_envs)
baxters = []
cabinets = []
default_prop_states = []
prop_start = []
envs = []
attractor_handles = []


for i in range(num_envs):
    env_ptr = gym.create_env(sim, lower, upper, int(np.sqrt(num_envs)))

    baxter_actor = gym.create_actor(env_ptr, baxter_asset, baxter_start_pose, "baxter", i, 1, 0)
    gym.set_actor_dof_properties(env_ptr, baxter_actor, baxter_dof_props)

    cabinet_pose = cabinet_start_pose
    dz = 0.5 * np.random.rand()
    dy = np.random.rand() - 0.5
    cabinet_actor = gym.create_actor(env_ptr, cabinet_asset, cabinet_pose, "cabinet", i, 2, 0)
    gym.set_actor_dof_properties(env_ptr, cabinet_actor, cabinet_dof_props)
    
    attractor_handles.append([])
    baxter_body_dict = gym.get_actor_rigid_body_dict(env_ptr, baxter_actor)
    baxter_props = gym.get_actor_rigid_body_states(env_ptr, baxter_actor, gymapi.STATE_POS)

    attractor_properties = gymapi.AttractorProperties()
    body_handle = gym.find_actor_rigid_body_handle(env_ptr, baxter_actor, "right_wrist")
    attractor_properties.axes = gymapi.AXIS_ALL
    attractor_properties.target = baxter_props['pose'][:][baxter_body_dict["right_wrist"]]
    attractor_properties.rigid_handle = body_handle
    attractor_handle = gym.create_rigid_body_attractor(env_ptr, attractor_properties)
    
    attractor_handles[i].append(attractor_handle)
    
    # attractor_properties = gymapi.AttractorProperties()
    # body_handle = gym.find_actor_rigid_body_handle(env_ptr, baxter_actor, "r_gripper_r_finger")
    # attractor_properties.axes = gymapi.AXIS_ALL
    # attractor_properties.target = baxter_props['pose'][:][baxter_body_dict["r_gripper_r_finger"]]
    # attractor_properties.rigid_handle = body_handle
    # attractor_handle = gym.create_rigid_body_attractor(env_ptr, attractor_properties)
    
   # attractor_handles[i].append(attractor_handle)

    envs.append(env_ptr)
    baxters.append(baxter_actor)
    cabinets.append(cabinet_actor)
    
hand_handle = gym.find_actor_rigid_body_handle(env_ptr, baxter_actor, "right_wrist")
drawer_handle = gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
lfinger_handle = gym.find_actor_rigid_body_handle(env_ptr, baxter_actor, "r_gripper_l_finger")
rfinger_handle = gym.find_actor_rigid_body_handle(env_ptr, baxter_actor, "r_gripper_r_finger")

hand = gym.find_actor_rigid_body_handle(envs[0], baxters[0], "right_wrist")
lfinger = gym.find_actor_rigid_body_handle(envs[0], baxters[0], "r_gripper_l_finger")
rfinger = gym.find_actor_rigid_body_handle(envs[0], baxters[0], "r_gripper_r_finger")

hand_pose = gym.get_rigid_transform(envs[0], hand)
lfinger_pose = gym.get_rigid_transform(envs[0], lfinger)
rfinger_pose = gym.get_rigid_transform(envs[0], rfinger)

finger_pose = gymapi.Transform()
finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
finger_pose.r = lfinger_pose.r

hand_pose_inv = hand_pose.inverse()
grasp_pose_axis = 1
baxter_local_grasp_pose = hand_pose_inv * finger_pose
# baxter_local_grasp_pose = hand_pose
baxter_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
baxter_local_grasp_pos = to_torch([baxter_local_grasp_pose.p.x, baxter_local_grasp_pose.p.y,
                                        baxter_local_grasp_pose.p.z], device='cuda:0').repeat((num_envs, 1))
baxter_local_grasp_rot = to_torch([baxter_local_grasp_pose.r.x, baxter_local_grasp_pose.r.y,
                                        baxter_local_grasp_pose.r.z, baxter_local_grasp_pose.r.w], device='cuda:0').repeat((num_envs, 1))

drawer_local_grasp_pose = gymapi.Transform()
drawer_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
drawer_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
drawer_local_grasp_pos = to_torch([drawer_local_grasp_pose.p.x, drawer_local_grasp_pose.p.y,
                                        drawer_local_grasp_pose.p.z], device='cuda:0').repeat((num_envs, 1))
drawer_local_grasp_rot = to_torch([drawer_local_grasp_pose.r.x, drawer_local_grasp_pose.r.y,
                                        drawer_local_grasp_pose.r.z, drawer_local_grasp_pose.r.w], device='cuda:0').repeat((num_envs, 1))

gripper_forward_axis = to_torch([0, 0, 1], device='cuda:0').repeat((num_envs, 1))
drawer_inward_axis = to_torch([-1, 0, 0], device='cuda:0').repeat((num_envs, 1))
gripper_up_axis = to_torch([0, 1, 0], device='cuda:0').repeat((num_envs, 1))
drawer_up_axis = to_torch([0, 0, 1], device='cuda:0').repeat((num_envs, 1))

baxter_grasp_pos = torch.zeros_like(baxter_local_grasp_pos)
baxter_grasp_rot = torch.zeros_like(baxter_local_grasp_rot)
baxter_grasp_rot[..., -1] = 1  # xyzw
drawer_grasp_pos = torch.zeros_like(drawer_local_grasp_pos)
drawer_grasp_rot = torch.zeros_like(drawer_local_grasp_rot)
drawer_grasp_rot[..., -1] = 1
baxter_lfinger_pos = torch.zeros_like(baxter_local_grasp_pos)
baxter_rfinger_pos = torch.zeros_like(baxter_local_grasp_pos)
baxter_lfinger_rot = torch.zeros_like(baxter_local_grasp_rot)
baxter_rfinger_rot = torch.zeros_like(baxter_local_grasp_rot)

baxter_begin_dof = 10
target_poses = []

target_pose1 = gymapi.Transform()
target_pose1.p = gymapi.Vec3(0.8, 0.1, 1.4)
target_pose1.r = gymapi.Quat.from_euler_zyx(-0.5 * math.pi, 0, 0.5 * math.pi)
target_poses.append(target_pose1)
target_pose2 = gymapi.Transform()
target_pose2.p = gymapi.Vec3(0.7, 0.7, 1.66)
target_pose2.r = gymapi.Quat.from_euler_zyx(-0.5 * math.pi, 0, 0.5 * math.pi)
target_poses.append(target_pose2)

attractor_properties = gym.get_attractor_properties(envs[0], attractor_handles[0][0])
base_pose = attractor_properties.target
def update_baxter(t):
    gym.clear_lines(viewer)
    for i in range(num_envs):
        for j in range(len(attractor_handles[i])):
            attractor_pose = copy(base_pose)
            sec = 5
            attractor_pose.p = attractor_pose.p + (target_poses[j].p - attractor_pose.p) * t / sec
            attractor_pose.r = target_poses[j].r
            if (t < sec):
                gym.set_attractor_target(envs[i], attractor_handles[i][j], attractor_pose)
            gymutil.draw_lines(axes_geom, gym, viewer, envs[i], target_poses[j])
            gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], target_poses[j])

next_baxter_update_time = 0.1
frame = 0

# Camera Sensor
camera_props = gymapi.CameraProperties()
camera_props.width = 1280
camera_props.height = 1280
camera_props.enable_tensors = True
camera_handle = gym.create_camera_sensor(envs[0], camera_props)

transform = gymapi.Transform()
transform.p = gymapi.Vec3(1,1,1)
transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
gym.set_camera_transform(camera_handle, envs[0], transform)
debug_fig = plt.figure("debug")

while not gym.query_viewer_has_closed(viewer):
    # check if we should update
    t = gym.get_sim_time(sim)
    if t >= next_baxter_update_time:
        update_baxter(t)
        next_baxter_update_time += 0.01

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

#    for env in envs:
#        gym.draw_env_rigid_contacts(viewer, env, colors[0], 0.5, True)

    # step rendering
    gym.step_graphics(sim)

    # digest image
    # gym.render_all_camera_sensors(sim)
    # gym.start_access_image_tensors(sim)

    # camera_tensor = gym.get_camera_image_gpu_tensor(sim, envs[0], camera_handle, gymapi.IMAGE_COLOR)
    # torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
    # cam_img = torch_camera_tensor.cpu().numpy()
    # cam_img = Im.fromarray(cam_img)
    # plt.imshow(cam_img)
    # plt.pause(1e-9)
    # debug_fig.clf()

    # gym.end_access_image_tensors(sim)

    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

    frame = frame + 1

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

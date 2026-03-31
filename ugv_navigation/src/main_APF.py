#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import env
import APF_Vel_ROS


gazebo_ugv = env.GazeboUGV(max_step=300)
# -------------------------Params------------------------------------
mass = 1.48
total_episode = 100
max_step_per_episode = 300
success_num = 0
print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)


# --------------------------Path Finding with APF--------------------
for i_episode in range(total_episode):
    state1, state2, dist_normalized = gazebo_ugv.reset()
    sum_vx_ugv = []
    sum_vz_ugv = []
    att_values = []
    rep_values = []
    for t in range(max_step_per_episode + 1):
        goal = np.array(gazebo_ugv.goal)
        curr_pos = np.array(gazebo_ugv.self_state[0:2])
        obs_pos = np.array(gazebo_ugv.cylinder_pos)
        action_space_vx = gazebo_ugv.action_space_vx
        action_space_vz = gazebo_ugv.action_space_vz
        att, rep, vx_world, vz_world = APF_Vel_ROS.vel_control(
            target_location=goal, current_position=curr_pos, obs_pos=obs_pos, mass=mass, obs_radius=2.0
        )
        yaw = gazebo_ugv.self_state[3]
        # -------Convert the velocity in world frame to the body frame
        vx_ugv, vz_ugv = APF_Vel_ROS.convert_to_ugv_frame(vx_world, vz_world, yaw)
        sum_vx_ugv.append(vx_ugv)
        sum_vz_ugv.append(vz_ugv)
        vx_ugv_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(vx_ugv, action_space_vx, strategy="min")
        vz_ugv_mapped = APF_Vel_ROS.fuzzy_map_v_triangular(vz_ugv, action_space_vz, strategy="max")
        gazebo_ugv.execute_linear_velocity(vx_ugv_mapped, vz_ugv_mapped)
        terminal, reward, success = gazebo_ugv.get_reward_and_terminate(time_step=t)
        if terminal:
            if gazebo_ugv.success:
                success_num += 1
                print("Step = ", t)
                avg_vx_ugv = sum(sum_vx_ugv) / len(sum_vx_ugv)
                avg_vz_ugv = sum(sum_vz_ugv) / len(sum_vz_ugv)
                print(f"Episode {i_episode}: Average vx_ugv = {avg_vx_ugv}, Average vz_ugv = {avg_vz_ugv}")
                print("Max Vx = ", max(sum_vx_ugv))
                print("Max Vz = ", max(sum_vz_ugv))
            break
print("Success Rate = {:.2f}%".format(success_num / total_episode * 100))

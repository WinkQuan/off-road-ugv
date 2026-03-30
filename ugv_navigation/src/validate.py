#!/home/yuhang/anaconda3/bin/python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import pandas as pd
import time
import env
import ddqn
import onnxruntime as ort


max_step_per_episode = 400
max_episode = 100
success_count = 0
step_count = 0


gazebo_ugv = env.GazeboUGV(max_step=max_step_per_episode)
agent = ddqn.DQN(
    gazebo_ugv,
    gazebo_ugv.action_space_vx,
    gazebo_ugv.action_space_vz,
    batch_size=64,
    memory_size=10000,
    target_update=4,
    gamma=0.99,
    learning_rate=1e-4,
    eps=0.0,
    eps_min=0.0,
    eps_period=5000,
    network="Duel",
)
## ONNX Model for Vehicle Decision-Making ##
model = "./Model/best_model.onnx"
csv_path = "./Your_CSV_File"
tra_path = "./Your_Tra_File"
sess = ort.InferenceSession(model)
obs_img = sess.get_inputs()[0].name
obs_pos_onnx = sess.get_inputs()[1].name


print("Action Space_vx = ", gazebo_ugv.action_space_vx)
print("Action Space_vz = ", gazebo_ugv.action_space_vz)
linear_x_values = []
angular_z_values = []
ugv_pos_list = []
t = 1

for i in range(max_episode):
    state1, state2, dist_normalized = gazebo_ugv.reset()
    print("dist_normalized = ", dist_normalized)
    rospy.sleep(0.1)
    for t in range(max_step_per_episode + 1):
        goal = np.array(gazebo_ugv.goal)
        curr_pos = np.array(gazebo_ugv.self_state[0:2])
        ugv_pos_list.append(curr_pos)
        # obs_pos and action_spaces not needed for validation
        yaw = gazebo_ugv.self_state[2]
        output_velocity = sess.run(
            None,
            {
                obs_img: np.array(np.expand_dims(state1, axis=0), dtype=np.float32),
                obs_pos_onnx: np.array(state2, dtype=np.float32).reshape(1, -1),
            },
        )
        output_vx_index = np.argmax(output_velocity[0])  # linear_vx
        output_vz_index = np.argmax(output_velocity[1])  # angular_vz
        gazebo_ugv.execute_linear_velocity(output_vx_index, output_vz_index)
        # Record velocities for analysis
        linear_x_values.append(gazebo_ugv.action_space_vx[output_vx_index])
        angular_z_values.append(gazebo_ugv.action_space_vz[output_vz_index])

        next_state1, next_state2, terminal, reward, success = gazebo_ugv.step(time_step=t + 1)
        if terminal:
            if success:
                np.savetxt(tra_path, ugv_pos_list)
                df_velocity_pre = pd.DataFrame({"linear_x": linear_x_values, "angular_z": angular_z_values})
                df_velocity_pre.to_csv(csv_path, index=False)  # Save velocity data
                success_count += 1
                step_count += t + 1
                print("Time Step = ", t + 1)
                linear_x_values.clear()
                angular_z_values.clear()
                ugv_pos_list.clear()
            break
        state1 = next_state1
        state2 = next_state2

success_rate = success_count / max_episode
if success_count > 0:
    average_step = step_count / success_count
else:
    average_step = 0
    print("Warning: No successful episodes, cannot calculate average steps.")
print("Success Rate: {:.2f}%".format(success_rate * 100))
print("Average Time Step: {:.0f}".format(average_step))

import mujoco
import numpy as np
import mink

model = mujoco.MjModel.from_xml_path("../robot/scene_sit_to_stand.xml")
data = mujoco.MjData(model)

# Start with stand pose
stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
mujoco.mj_resetDataKeyframe(model, data, stand_id)
mujoco.mj_forward(model, data)

q_start = data.qpos.copy()
configuration = mink.Configuration(model)
configuration.update(q_start)

# Targets
al = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
ar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
foot_z = 0.5 * (data.xpos[al, 2] + data.xpos[ar, 2])

left_foot_rot = data.xmat[al].reshape(3, 3)
right_foot_rot = data.xmat[ar].reshape(3, 3)
left_foot_pos = data.xpos[al].copy()
right_foot_pos = data.xpos[ar].copy()

# Move feet forward slightly so they don't clip the box
left_foot_pos[0] += 0.15
right_foot_pos[0] += 0.15

def se3(rot, pos):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return mink.SE3.from_matrix(T)

lf_task = mink.FrameTask(frame_name="left_ankle_roll_link", frame_type="body", position_cost=10, orientation_cost=10)
lf_task.set_target(se3(left_foot_rot, left_foot_pos))

rf_task = mink.FrameTask(frame_name="right_ankle_roll_link", frame_type="body", position_cost=10, orientation_cost=10)
rf_task.set_target(se3(right_foot_rot, right_foot_pos))

pelvis_task = mink.FrameTask(frame_name="pelvis", frame_type="body", position_cost=10, orientation_cost=10)
# Seat box is at x=-0.15, z=0.3 top. Put pelvis at x=0, z=0.35
pelvis_rot = np.eye(3)
pelvis_task.set_target(se3(pelvis_rot, [0.0, 0.0, 0.35]))

tasks = [lf_task, rf_task, pelvis_task]
limits = [mink.ConfigurationLimit(model)]

for _ in range(500):
    vel = mink.solve_ik(configuration, tasks, 0.01, solver="daqp", limits=limits)
    configuration.integrate_inplace(vel, 0.01)

data.qpos[:] = configuration.q
mujoco.mj_forward(model, data)

print("IK Sit Pose:")
# format for XML
q = data.qpos
print(f"      {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}")
print(f"      {q[3]:.4f} {q[4]:.4f} {q[5]:.4f} {q[6]:.4f}")
s = "      "
for i in range(7, 13): s += f"{q[i]:.4f} "
print(s)
s = "      "
for i in range(13, 19): s += f"{q[i]:.4f} "
print(s)
s = "      "
for i in range(19, 22): s += f"{q[i]:.4f} "
print(s)
s = "      "
for i in range(22, 29): s += f"{q[i]:.4f} "
print(s)
s = "      "
for i in range(29, 36): s += f"{q[i]:.4f} "
print(s)

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("../robot/scene_sit_to_stand.xml")
data = mujoco.MjData(model)

# Start with stand pose to get zeroed angles
stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
mujoco.mj_resetDataKeyframe(model, data, stand_id)
mujoco.mj_forward(model, data)

q = data.qpos.copy()

# Base pos/quat
q[0:3] = [0, 0, 0.5] # temporary Z
q[3:7] = [1, 0, 0, 0]

# Left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
# Joints are at index 7 to 12
q[7] = -1.5708
q[10] = 1.5708
q[11] = 0.0

# Right leg (13 to 18)
q[13] = -1.5708
q[16] = 1.5708
q[17] = 0.0

data.qpos[:] = q
mujoco.mj_forward(model, data)

# Measure foot Z to calculate correct pelvis Z
al = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
foot_z = data.xpos[al, 2]
foot_x = data.xpos[al, 0]
pelvis_z = data.qpos[2]

# If foot_z is negative, pelvis needs to be moved up by that amount
offset_z = -foot_z
correct_pelvis_z = pelvis_z + offset_z

print(f"Correct Pelvis Z: {correct_pelvis_z:.4f}")
print(f"Foot X relative to pelvis: {foot_x:.4f}")

# Update Z and forward kinematics
q[2] = correct_pelvis_z
data.qpos[:] = q
mujoco.mj_forward(model, data)

pelvis_x = data.xpos[1, 0] # pelvis body
print(f"Pelvis body X: {pelvis_x:.4f}")

print("\nIK Sit Pose for XML:")
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

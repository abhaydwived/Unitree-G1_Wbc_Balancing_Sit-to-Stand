import mujoco
import mujoco.viewer
import numpy as np
import time


XML_PATH = "robot/g1_balance.xml"


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("=" * 60)
print("G1 MODEL LOADED")
print("=" * 60)

print(f"Number of bodies     : {model.nbody}")
print(f"Number of joints     : {model.njnt}")
print(f"Number of DOF        : {model.nv}")
print(f"Number of actuators  : {model.nu}")
print(f"Number of sensors    : {model.nsensor}")


# ---------------------------------------------------------
# Print joints
# ---------------------------------------------------------

print("\nJOINTS")
print("-" * 60)

for i in range(model.njnt):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    print(
        f"{i:2d} | "
        f"{name:35s} | "
        f"type={model.jnt_type[i]} | "
        f"qposadr={model.jnt_qposadr[i]} | "
        f"dofadr={model.jnt_dofadr[i]}"
    )


# ---------------------------------------------------------
# Print actuators
# ---------------------------------------------------------

print("\nACTUATORS")
print("-" * 60)

for i in range(model.nu):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    joint_id = model.actuator_trnid[i, 0]

    joint_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id
    )

    print(
        f"{i:2d} | "
        f"{name:35s} | "
        f"joint={joint_name}"
    )


# ---------------------------------------------------------
# Load standing keyframe
# ---------------------------------------------------------

stand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "stand"
)

if stand_id == -1:
    raise RuntimeError("Could not find 'stand' keyframe")

mujoco.mj_resetDataKeyframe(
    model,
    data,
    stand_id
)

# Forward dynamics
mujoco.mj_forward(model, data)


# ---------------------------------------------------------
# Useful body/site IDs
# ---------------------------------------------------------

pelvis_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pelvis"
)

torso_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "torso_link"
)

left_foot_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "left_foot"
)

right_foot_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_foot"
)


# ---------------------------------------------------------
# Print IDs
# ---------------------------------------------------------

print("\nIMPORTANT IDs")
print("-" * 60)

print("Pelvis body ID    :", pelvis_id)
print("Torso body ID     :", torso_id)
print("Left foot site ID :", left_foot_id)
print("Right foot site ID:", right_foot_id)


# ---------------------------------------------------------
# COM
# ---------------------------------------------------------

com_position = data.subtree_com[0].copy()

print("\nINITIAL STATE")
print("-" * 60)

print("COM position:")
print(com_position)

print("\nPelvis position:")
print(data.xpos[pelvis_id])

print("\nTorso position:")
print(data.xpos[torso_id])

print("\nLeft foot position:")
print(data.site_xpos[left_foot_id])

print("\nRight foot position:")
print(data.site_xpos[right_foot_id])


# ---------------------------------------------------------
# Joint configuration
# ---------------------------------------------------------

print("\nJOINT POSITIONS")
print("-" * 60)

for i in range(model.njnt):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    qpos_start = model.jnt_qposadr[i]

    # Skip free joint for simple scalar printing
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
        continue

    print(
        f"{name:35s} "
        f"q = {data.qpos[qpos_start]: .5f} rad "
        f"({np.degrees(data.qpos[qpos_start]): .2f} deg)"
    )


# ---------------------------------------------------------
# Open viewer
# ---------------------------------------------------------

print("\nOpening MuJoCo viewer...")
print("Close the viewer to exit.")


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # For now we do NOT apply any control.
        # We are only inspecting the model.

        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(0.01)
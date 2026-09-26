import mujoco
import mujoco.viewer
import numpy as np
import time


XML_PATH = "./robot/scene.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# =========================================================
# Load standing configuration
# =========================================================

stand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "stand"
)

mujoco.mj_resetDataKeyframe(
    model,
    data,
    stand_id
)

mujoco.mj_forward(model, data)


# =========================================================
# Desired joint configuration
# =========================================================

q_des = data.qpos[7:].copy()


# =========================================================
# Gains
# =========================================================

Kp = np.array([
    # legs
    80, 80, 80, 100, 40, 40,
    80, 80, 80, 100, 40, 40,

    # waist
    50, 50, 50,

    # left arm
    20, 20, 20, 20, 10, 10, 10,

    # right arm
    20, 20, 20, 20, 10, 10, 10
], dtype=float)


Kd = np.array([
    # legs
    4, 4, 4, 4, 1.5, 1.5,
    4, 4, 4, 4, 1.5, 1.5,

    # waist
    2, 2, 2,

    # left arm
    0.8, 0.8, 0.8, 0.8, 0.3, 0.3, 0.3,

    # right arm
    0.8, 0.8, 0.8, 0.8, 0.3, 0.3, 0.3
], dtype=float)


# =========================================================
# Simulation
# =========================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # -------------------------------------------------
        # Current joint state
        # -------------------------------------------------

        q = data.qpos[7:43]

        # IMPORTANT:
        # qvel has 35 elements.
        # First 6 = floating base velocity
        # Next 29 = actuated joint velocities.

        qdot = data.qvel[6:35]


        # -------------------------------------------------
        # Posture feedback
        # -------------------------------------------------

        position_error = q_des - q

        velocity_error = -qdot

        tau_pd = (
            Kp * position_error
            + Kd * velocity_error
        )


        # -------------------------------------------------
        # Gravity compensation
        # -------------------------------------------------

        # MuJoCo computes generalized bias forces.
        # For a static configuration, this contains
        # gravity and other velocity-dependent terms.

        tau_gravity = data.qfrc_bias[6:35]


        # -------------------------------------------------
        # Total joint torque
        # -------------------------------------------------

        tau = tau_pd + tau_gravity


        # -------------------------------------------------
        # Apply torque
        # -------------------------------------------------

        data.ctrl[:] = tau

        print(
            "tau_pd max:",
            np.max(np.abs(tau_pd)),
            "gravity max:",
            np.max(np.abs(tau_gravity)),
            "total max:",
            np.max(np.abs(tau))
        )

        print("contacts:", data.ncon)


        # -------------------------------------------------
        # Step simulation
        # -------------------------------------------------

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.001)
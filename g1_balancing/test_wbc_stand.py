import time

import mujoco
import mujoco.viewer
import numpy as np

from wbc import G1WBC


MODEL_PATH = "./robot/scene.xml"


# Load

model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

data = mujoco.MjData(model)


# Reset to stand

stand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "stand"
)

if stand_id < 0:

    raise RuntimeError(
        "Keyframe 'stand' not found."
    )


mujoco.mj_resetDataKeyframe(
    model,
    data,
    stand_id
)

mujoco.mj_forward(
    model,
    data
)


# WBC

wbc = G1WBC(
    model,
    data
)

# Desired posture


q_des = data.qpos[
    7:36
].copy()


# Desired COM

com_des = (
    wbc.get_com_position()
    .copy()
)


print(
    "\nDesired joint configuration:"
)

print(q_des)

print(
    "\nDesired COM:"
)

print(com_des)

print(
    "\nSimulation timestep:",
    model.opt.timestep
)


# Simulation

step = 0

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        # Solve WBC

        (
            tau,
            qdd,
            forces,
            solved
        ) = wbc.solve(
            q_des,
            com_des
        )

        # Apply torque

        if solved:

            data.ctrl[:] = tau

        else:

            data.ctrl[:] = 0.0

        # Step

        mujoco.mj_step(
            model,
            data
        )

        # Debug

        if step % 100 == 0:

            pelvis = data.xpos[
                1
            ]

            left_foot = (
                data.site_xpos[
                    wbc.left_foot_id
                ]
            )

            right_foot = (
                data.site_xpos[
                    wbc.right_foot_id
                ]
            )

            com = (
                wbc.get_com_position()
            )

            print(
                f"\nstep = {step}"
            )

            print(
                "pelvis:",
                pelvis
            )

            print(
                "COM:",
                com
            )

            print(
                "COM error:",
                com - com_des
            )

            print(
                "left foot:",
                left_foot
            )

            print(
                "right foot:",
                right_foot
            )

            print(
                "tau max:",
                np.max(
                    np.abs(tau)
                )
            )

            print(
                "tau min:",
                np.min(tau)
            )

            print(
                "F:",
                forces
            )

            if forces.size >= 6:
                print(
                    "Fz left/right:",
                    forces[2],
                    forces[5]
                )
            else:
                print(
                    "Fz left/right: unavailable "
                    "(no active contacts)"
                )

            print(
                "qdd max:",
                np.max(
                    np.abs(qdd)
                )
            )

            print(
                "solver:",
                solved
            )

        viewer.sync()

        step += 1

        time.sleep(
            model.opt.timestep
        )
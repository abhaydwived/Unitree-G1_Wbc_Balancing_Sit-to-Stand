import time
import numpy as np
import mujoco
import mujoco.viewer
import mink
import os

# =============================================================================
# Trajectry generator — produces reference (q_des, com_des) at each step
# using the same mink IK as test_ik_sit_to_stand.py
# =============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path    = os.path.abspath(os.path.join(current_dir, "..", "robot", "scene_sit_to_stand.xml"))

IK_SOLVER = "daqp"
DT        = 0.002           # must match WBC timestep

SEAT_KF   = "sts"
STAND_KF  = "stand"

T_HOLD    = 1.0             # lean phase (s)
T_RISE    = 1.5             # rise phase (s)
T_STAND   = 0.5             # brief hold at IK standing height (s)
T_RECENTER= 1.5             # move pelvis back to X=0, Z=0.79 (s)
TOTAL_DUR = T_HOLD + T_RISE + T_STAND + T_RECENTER

Z_STAND   = 0.76            # target pelvis Z (m) — fixed throughout
COM_KP    = 0.5             # IK CoM bias gain


def smooth(t, dur):
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t / dur, 0.0, 1.0)))


def get_body_se3(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    T   = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    T[:3,  3] = data.xpos[bid]
    return mink.SE3.from_matrix(T)


def make_se3(rot3, pos3):
    T = np.eye(4)
    T[:3, :3] = rot3
    T[:3,  3] = pos3
    return mink.SE3.from_matrix(T)


def load_qpos(model, name):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if kid < 0:
        raise ValueError(f"Keyframe '{name}' not found")
    tmp = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, tmp, kid)
    return tmp.qpos.copy()


def generate_trajectory(model):
    """
    Run mink IK offline for TOTAL_DUR seconds and return
        traj_q   : list of qpos arrays  (one per DT step)
        traj_com : list of CoM [3]      (one per DT step)
    """
    data = mujoco.MjData(model)

    al = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    ar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    # ── sit reference ────────────────────────────────────────────────────────
    q_sit = load_qpos(model, SEAT_KF)
    data.qpos[:] = q_sit;  data.qvel[:] = 0.0
    mujoco.mj_kinematics(model, data)

    sit_pelvis_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")].copy()
    bid_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    sit_pelvis_rot = data.xmat[bid_p].reshape(3, 3).copy()

    sit_lf_pos = data.xpos[al].copy()
    sit_rf_pos = data.xpos[ar].copy()
    sit_lf_rot = data.xmat[al].reshape(3, 3).copy()
    sit_rf_rot = data.xmat[ar].reshape(3, 3).copy()

    foot_mid_xy = 0.5 * (sit_lf_pos[:2] + sit_rf_pos[:2])

    # ── stand posture target ─────────────────────────────────────────────────
    q_stand = load_qpos(model, STAND_KF)

    # ── mink setup ───────────────────────────────────────────────────────────
    configuration = mink.Configuration(model)
    configuration.update(q_sit)
    mujoco.mj_kinematics(model, configuration.data)

    config_stand = mink.Configuration(model)
    config_stand.update(q_stand)
    posture_task = mink.PostureTask(model, cost=0.01)
    posture_task.set_target_from_configuration(config_stand)

    lf_task = mink.FrameTask("left_ankle_roll_link",  "body", position_cost=10.0, orientation_cost=5.0, lm_damping=1e-3)
    rf_task = mink.FrameTask("right_ankle_roll_link", "body", position_cost=10.0, orientation_cost=5.0, lm_damping=1e-3)
    lf_task.set_target(make_se3(sit_lf_rot, sit_lf_pos))
    rf_task.set_target(make_se3(sit_rf_rot, sit_rf_pos))

    pelvis_task = mink.FrameTask("pelvis", "body", position_cost=2.0, orientation_cost=1.0, lm_damping=1e-3)
    pelvis_task.set_target(make_se3(sit_pelvis_rot, sit_pelvis_pos))

    tasks  = [lf_task, rf_task, pelvis_task, posture_task]
    limits = [mink.ConfigurationLimit(model, gain=0.99)]

    _T1 = T_HOLD
    _T2 = T_HOLD + T_RISE
    _T3 = T_HOLD + T_RISE + T_STAND       # start of RECENTER
    _T4 = T_HOLD + T_RISE + T_STAND + T_RECENTER  # end of RECENTER

    # Stand keyframe pelvis X and Y (target for RECENTER)
    data.qpos[:] = q_stand
    mujoco.mj_kinematics(model, data)
    stand_pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    stand_pelvis_x = data.xpos[stand_pid][0]   # ~0.0
    stand_pelvis_y = data.xpos[stand_pid][1]   # ~0.0

    def get_tgt_x(t):
        if t < _T1:
            return sit_pelvis_pos[0] + smooth(t, _T1) * (foot_mid_xy[0] - sit_pelvis_pos[0])
        elif t < _T3:
            return foot_mid_xy[0]
        else:
            # RECENTER: move X back towards stand keyframe X
            alpha = smooth(t - _T3, T_RECENTER)
            return foot_mid_xy[0] + alpha * (stand_pelvis_x - foot_mid_xy[0])

    def get_tgt_y(t):
        if t < _T3:
            return sit_pelvis_pos[1]
        alpha = smooth(t - _T3, T_RECENTER)
        return sit_pelvis_pos[1] + alpha * (stand_pelvis_y - sit_pelvis_pos[1])

    def get_tgt_z(t):
        if t < _T1:
            return sit_pelvis_pos[2]
        elif t < _T2:
            return sit_pelvis_pos[2] + smooth(t - _T1, T_RISE) * (Z_STAND - sit_pelvis_pos[2])
        return Z_STAND   # hold at 0.76 for STAND and RECENTER phases

    def get_tgt_rot(t):
        if t < _T1:
            return sit_pelvis_rot.copy()
        alpha = smooth(t - _T1, T_RISE)
        R = (1.0 - alpha) * sit_pelvis_rot + alpha * np.eye(3)
        U, _, Vt = np.linalg.svd(R)
        return U @ Vt

    traj_q   = []
    traj_com = []
    com_x_bias = 0.0
    n_steps    = int(TOTAL_DUR / DT) + 1

    for step in range(n_steps):
        t = step * DT
        t_c = min(t, _T4)   # clamp to end of RECENTER

        mujoco.mj_kinematics(model, configuration.data)
        com = configuration.data.subtree_com[0].copy()

        if _T1 < t < _T2:
            com_x_bias -= COM_KP * (com[0] - foot_mid_xy[0]) * DT
            com_x_bias  = float(np.clip(com_x_bias, -0.12, 0.12))
        elif t >= _T3:
            # During RECENTER, stop com_x_bias so pelvis returns cleanly
            com_x_bias *= 0.97
        else:
            com_x_bias *= 0.97

        tgt_x   = get_tgt_x(t_c) + (com_x_bias if t < _T3 else 0.0)
        tgt_y   = get_tgt_y(t_c)
        tgt_z   = get_tgt_z(t_c)
        tgt_rot = get_tgt_rot(t_c)

        pelvis_task.set_target(make_se3(tgt_rot, np.array([tgt_x, tgt_y, tgt_z])))

        try:
            vel = mink.solve_ik(configuration, tasks, DT, IK_SOLVER, limits=limits)
            configuration.integrate_inplace(vel, DT)
        except Exception:
            pass

        mujoco.mj_kinematics(model, configuration.data)
        traj_q.append(configuration.q.copy())
        traj_com.append(configuration.data.subtree_com[0].copy())

    print(f"[Traj] Generated {len(traj_q)} steps ({TOTAL_DUR:.1f} s at {1/DT:.0f} Hz)")
    return traj_q, traj_com

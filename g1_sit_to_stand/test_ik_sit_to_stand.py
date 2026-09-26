import time
import numpy as np
import mujoco
import mujoco.viewer
import mink
import os

# =============================================================================
# Paths / solver / timing
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.abspath(os.path.join(current_dir, "..", "robot", "scene_sit_to_stand.xml"))
SOLVER = "daqp"
DT     = 0.002  # 500 Hz

SEAT_KEYFRAME_NAME  = "sts"
STAND_KEYFRAME_NAME = "stand"

# =============================================================================
# Motion parameters
# =============================================================================
Z_STAND = 0.76   # m - target standing pelvis height for G1

# Phase durations (s)
T_HOLD = 1.0     # Phase 0: hold sitting pose, lean pelvis forward over feet
T_RISE = 1.5     # Phase 1: pelvis rises from sit height to stand height

# CoM forward-lean proportional gain (X-axis only, since G1 faces +X)
COM_KP = 0.5


# =============================================================================
# Helpers
# =============================================================================
def smooth(t: float, dur: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t / dur, 0.0, 1.0)))

def get_body_se3(model, data, name: str) -> mink.SE3:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"Body '{name}' not found.")
    T = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    T[:3,  3] = data.xpos[bid]
    return mink.SE3.from_matrix(T)

def make_se3(rot3x3, pos3) -> mink.SE3:
    T = np.eye(4)
    T[:3, :3] = rot3x3
    T[:3,  3] = pos3
    return mink.SE3.from_matrix(T)

def load_qpos(model, keyframe_name: str) -> np.ndarray:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if key_id < 0:
        raise ValueError(f"Keyframe '{keyframe_name}' not found.")
    tmp = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, tmp, key_id)
    return tmp.qpos.copy()

def compute_com(model, data) -> np.ndarray:
    return data.subtree_com[0].copy()


# =============================================================================
# Main
# =============================================================================
def main():
    print("[Init] Loading G1 model...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # G1 ankle body indices (foot proxies)
    al = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    ar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    # ── 1. Read STS (sit) keyframe - this is our exact start ─────────────────
    print(f"[Init] Loading '{SEAT_KEYFRAME_NAME}' keyframe...")
    q_sit = load_qpos(model, SEAT_KEYFRAME_NAME)
    data.qpos[:] = q_sit
    data.qvel[:]  = 0.0
    mujoco.mj_kinematics(model, data)

    # Pelvis pose at sit
    sit_pelvis_se3 = get_body_se3(model, data, "pelvis")
    sit_pelvis_pos = sit_pelvis_se3.translation().copy()
    # Extract rotation as plain numpy 3x3 matrix
    bid_p = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    sit_pelvis_rot = data.xmat[bid_p].reshape(3, 3).copy()

    # Foot positions/orientations at sit - pin these as targets
    sit_lf_pos = data.xpos[al].copy()
    sit_rf_pos = data.xpos[ar].copy()
    sit_lf_rot = data.xmat[al].reshape(3, 3).copy()
    sit_rf_rot = data.xmat[ar].reshape(3, 3).copy()

    # Midpoint of feet (support polygon centre)
    foot_mid_xy = 0.5 * (sit_lf_pos[:2] + sit_rf_pos[:2])
    foot_z      = 0.5 * (sit_lf_pos[2]  + sit_rf_pos[2])

    print(f"[Init] Sit pelvis  : X={sit_pelvis_pos[0]:.4f}  Y={sit_pelvis_pos[1]:.4f}  Z={sit_pelvis_pos[2]:.4f}")
    print(f"[Init] Foot mid XY : X={foot_mid_xy[0]:.4f}  Y={foot_mid_xy[1]:.4f}  foot_Z={foot_z:.4f}")

    # ── 2. Stand keyframe for posture target ──────────────────────────────────
    q_stand = load_qpos(model, STAND_KEYFRAME_NAME)

    # ── 3. mink setup ─────────────────────────────────────────────────────────
    configuration = mink.Configuration(model)
    configuration.update(q_sit)      # start exactly at sts keyframe
    mujoco.mj_kinematics(model, configuration.data)

    config_stand = mink.Configuration(model)
    config_stand.update(q_stand)
    posture_task = mink.PostureTask(model, cost=0.01)
    posture_task.set_target_from_configuration(config_stand)

    # Foot tasks: pin feet at their EXACT sit-keyframe world positions
    lf_task = mink.FrameTask(
        frame_name="left_ankle_roll_link", frame_type="body",
        position_cost=10.0, orientation_cost=5.0, lm_damping=1e-3
    )
    lf_task.set_target(make_se3(sit_lf_rot, sit_lf_pos))

    rf_task = mink.FrameTask(
        frame_name="right_ankle_roll_link", frame_type="body",
        position_cost=10.0, orientation_cost=5.0, lm_damping=1e-3
    )
    rf_task.set_target(make_se3(sit_rf_rot, sit_rf_pos))

    # Pelvis task: updated every step along the trajectory
    pelvis_task = mink.FrameTask(
        frame_name="pelvis", frame_type="body",
        position_cost=2.0, orientation_cost=1.0, lm_damping=1e-3
    )
    pelvis_task.set_target(sit_pelvis_se3)  # init at exact sit pose

    tasks  = [lf_task, rf_task, pelvis_task, posture_task]
    limits = [mink.ConfigurationLimit(model, gain=0.99)]

    # ── 4. Frozen start - exact keyframe ──────────────────────────────────────
    q_start = q_sit.copy()

    _T1 = T_HOLD
    _T2 = T_HOLD + T_RISE
    TOTAL_DUR = _T2

    # ── 5. Trajectory ─────────────────────────────────────────────────────────
    # G1 faces +X direction.
    # HOLD  : pelvis X shifts from sit_pelvis_pos[0] toward foot_mid_xy[0] (lean forward)
    # RISE  : pelvis Z goes from sit_pelvis_pos[2] to Z_STAND, X stays over feet
    # Pelvis Y is locked at sit_pelvis_pos[1] at all times

    def get_tgt_z(t: float) -> float:
        if t < _T1:
            return sit_pelvis_pos[2]
        elif t < _T2:
            return sit_pelvis_pos[2] + smooth(t - _T1, T_RISE) * (Z_STAND - sit_pelvis_pos[2])
        return Z_STAND

    def get_tgt_x(t: float) -> float:
        if t < _T1:
            # lean pelvis forward toward foot midpoint X
            return sit_pelvis_pos[0] + smooth(t, _T1) * (foot_mid_xy[0] - sit_pelvis_pos[0])
        return foot_mid_xy[0]

    def get_pelvis_rot(t: float) -> np.ndarray:
        """Interpolate pelvis from sit rotation to upright (identity) during RISE."""
        if t < _T1:
            return sit_pelvis_rot.copy()
        alpha = smooth(t - _T1, T_RISE)
        R = (1.0 - alpha) * sit_pelvis_rot + alpha * np.eye(3)
        U, _, Vt = np.linalg.svd(R)
        return U @ Vt

    # ── 6. Viewer loop ────────────────────────────────────────────────────────
    print(f"\n[Sim] Phase 0 HOLD   : 0.0 -> {_T1:.1f} s  (lean forward)")
    print(f"[Sim] Phase 1 RISE   : {_T1:.1f} -> {_T2:.1f} s  (stand up)")
    print(f"[Sim] Phase 2 STAND  : {_T2:.1f} s ->  (hold)")
    print("[Sim] Press R to reset.\n")

    last_print = -1.0
    last_time  = -1.0
    com_x_bias = 0.0

    def key_callback(keycode):
        nonlocal last_print, com_x_bias
        if keycode in (ord('r'), ord('R')):
            data.time  = 0.0
            last_print = -1.0
            com_x_bias = 0.0
            data.qpos[:] = q_start
            data.qvel[:]  = 0.0
            configuration.update(q_start)
            mujoco.mj_kinematics(model, data)
            print("[Reset] Back to sit keyframe.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.azimuth   = 140.0
        viewer.cam.elevation = -15.0
        viewer.cam.distance  = 3.0

        while viewer.is_running():
            t0 = time.perf_counter()

            # detect viewer reset
            if data.time < last_time:
                com_x_bias = 0.0
                data.qpos[:] = q_start
                data.qvel[:]  = 0.0
                configuration.update(q_start)
                mujoco.mj_kinematics(model, data)
                last_print = -1.0
            last_time = data.time

            # CoM feedback - correct X drift during RISE
            mujoco.mj_kinematics(model, configuration.data)
            com = compute_com(model, configuration.data)
            if _T1 < data.time < TOTAL_DUR:
                com_x_bias -= COM_KP * (com[0] - foot_mid_xy[0]) * DT
                com_x_bias  = float(np.clip(com_x_bias, -0.12, 0.12))
            else:
                com_x_bias *= 0.97

            # Pelvis target for this step
            t_c     = min(data.time, TOTAL_DUR)
            tgt_x   = get_tgt_x(t_c) + com_x_bias
            tgt_y   = sit_pelvis_pos[1]   # locked laterally
            tgt_z   = get_tgt_z(t_c)
            tgt_rot = get_pelvis_rot(t_c)

            pelvis_task.set_target(make_se3(tgt_rot, np.array([tgt_x, tgt_y, tgt_z])))

            # IK solve
            try:
                vel = mink.solve_ik(configuration, tasks, DT, SOLVER, limits=limits)
                configuration.integrate_inplace(vel, DT)
            except Exception:
                pass

            # Apply to simulation (pure kinematic override)
            data.qpos[:] = configuration.q[:]
            data.qvel[:]  = 0.0
            data.ctrl[:]  = 0.0
            mujoco.mj_step(model, data)
            data.qpos[:] = configuration.q[:]
            data.qvel[:]  = 0.0

            viewer.sync()

            # Console log
            if data.time - last_print >= 0.5:
                last_print = data.time
                mujoco.mj_kinematics(model, data)
                p_now   = get_body_se3(model, data, "pelvis").translation()
                com_now = compute_com(model, data)
                if data.time < _T1:
                    phase = "HOLD (lean)"
                elif data.time < _T2:
                    phase = "RISE      "
                else:
                    phase = "STANDING  "
                print(f"  t={data.time:5.2f}s [{phase}]  "
                      f"pelvis=({p_now[0]:.3f},{p_now[1]:.3f},{p_now[2]:.3f})  "
                      f"CoM_X={com_now[0]:.3f}  bias={com_x_bias:.3f}")

            elapsed = time.perf_counter() - t0
            if DT - elapsed > 0:
                time.sleep(DT - elapsed)

    print("[Done] Viewer closed.")


if __name__ == "__main__":
    main()

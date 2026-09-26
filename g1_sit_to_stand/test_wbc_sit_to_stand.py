"""
G1 Sit-to-Stand -- Whole-Body Controller
=========================================
Architecture
------------
Phase 0  KINEMATIC (0 -> TOTAL_DUR):
    Robot follows mink-IK trajectory exactly via qpos injection.
    Sit -> lean -> rise -> stand (3.5s IK trajectory).

Phase 1  WBC STAND (TOTAL_DUR -> inf):
    Uses g1_sit_to_stand/wbc.py with:
      - com_des = ACTUAL CoM at transition (no mismatch!)
      - q_des   = ACTUAL joint angles at transition
      - w_foot  = 500 (soft -- avoids infeasibility from foot Jacobian drift)
      - Robust PD fallback (Kp=200) for any failed QP steps

    Debug confirmed: first solve returns solved=True, tau_max=11.9N,
    Fz=(163,163)N when using actual state as target.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
import os

from wbc import G1WBC   # g1_sit_to_stand/wbc.py (max_iter=10000, eps=5e-4)
from sts_trajectory import generate_trajectory, load_qpos, DT, SEAT_KF, TOTAL_DUR

# =============================================================================
# Config
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.abspath(os.path.join(current_dir, "..", "robot", "scene_sit_to_stand.xml"))

# =============================================================================
# Load model
# =============================================================================
print("[Init] Loading model...")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# =============================================================================
# Generate reference trajectory (mink IK, offline)
# =============================================================================
print("[Traj] Generating sit-to-stand reference trajectory...")
traj_q, traj_com = generate_trajectory(model)
N_TRAJ = len(traj_q)
N_last = N_TRAJ - 1
print(f"[Traj] {N_TRAJ} steps ({TOTAL_DUR:.1f} s)")

# =============================================================================
# Reset to STS sit pose
# =============================================================================
q_sit = load_qpos(model, SEAT_KF)
data.qpos[:] = q_sit
data.qvel[:]  = 0.0
mujoco.mj_forward(model, data)

# =============================================================================
# WBC init
# =============================================================================
print("[WBC] Initialising WBC...")
wbc = G1WBC(model, data)

# Generous fz_max for the standing phase
total_mass = float(np.sum(model.body_mass))
wbc.fz_max = total_mass * 9.81 * 2.5

# Standing weights: posture dominant, foot soft, COM moderate
wbc.w_posture = 5.0
wbc.w_com     = 100.0
wbc.w_foot    = 500.0   # soft enough to avoid Jacobian-drift infeasibility

print(f"[WBC] mass={total_mass:.1f}kg  fz_max={wbc.fz_max:.0f}N/foot")

# WBC targets — set at transition from ACTUAL robot state
com_des_wbc = None
q_des_wbc   = None

# =============================================================================
# State
# =============================================================================
step                = 0
last_log_time       = -1.0
wbc_active          = False
wbc_transition_done = False

# Fallback-only counter (to detect persistent failure)
fallback_streak = 0

def key_callback(keycode):
    global step, last_log_time, wbc_active, wbc_transition_done
    global com_des_wbc, q_des_wbc, fallback_streak
    if keycode in (ord('r'), ord('R')):
        step                = 0
        last_log_time       = -1.0
        wbc_active          = False
        wbc_transition_done = False
        com_des_wbc         = None
        q_des_wbc           = None
        fallback_streak     = 0
        data.qpos[:]  = q_sit
        data.qvel[:]  = 0.0
        mujoco.mj_forward(model, data)
        print("\n[Reset] Restarted.")

print(f"\n[Sim] KIN: 0->{TOTAL_DUR:.1f}s  |  WBC STAND: {TOTAL_DUR:.1f}s+  |  Press R to reset\n")

with mujoco.viewer.launch_passive(
    model, data, key_callback=key_callback
) as viewer:
    viewer.cam.azimuth   = 140.0
    viewer.cam.elevation = -15.0
    viewer.cam.distance  = 3.0

    while viewer.is_running():
        t0 = time.perf_counter()

        sim_time = step * DT
        traj_idx = min(step, N_last)

        if not wbc_active:
            # ── KINEMATIC phase ───────────────────────────────────────────────
            if sim_time < 1.0:   phase = "KIN (sit) "
            elif sim_time < 2.5: phase = "KIN (rise)"
            else:                phase = "KIN (stand)"

            q_now    = traj_q[traj_idx]
            idx_next = min(traj_idx + 1, N_last)
            dq       = np.zeros(model.nv)
            mujoco.mj_differentiatePos(model, dq, DT, q_now, traj_q[idx_next])

            data.qpos[:] = q_now
            data.qvel[:] = dq
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)

            forces = np.zeros(12)
            tau    = np.zeros(wbc.nu)
            solved = True

            if step >= N_last:
                wbc_active = True

        else:
            # ── WBC STAND phase ───────────────────────────────────────────────
            phase = "WBC STAND "

            # One-time transition: capture ACTUAL robot state as WBC target
            if not wbc_transition_done:
                wbc_transition_done = True
                data.qpos[:] = traj_q[N_last]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

                # Target = EXACTLY the current pose (zero initial error)
                com_des_wbc = data.subtree_com[0].copy()
                q_des_wbc   = data.qpos[7:36].copy()

                pid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                print(f"\n[WBC] Transition at t={sim_time:.2f}s")
                print(f"      pelvis Z = {data.xpos[pid][2]:.3f}m")
                print(f"      CoM      = ({com_des_wbc[0]:.3f}, {com_des_wbc[1]:.3f}, {com_des_wbc[2]:.3f})")
                print(f"      Stabilisation started...\n")

            # WBC solve
            tau, qdd, forces, solved = wbc.solve(q_des_wbc, com_des_wbc)

            if not solved:
                fallback_streak += 1
                # Robust PD fallback: stiff position hold
                q_cur  = data.qpos[7:36].copy()
                qd_cur = data.qvel[6:35].copy()
                tau    = 200.0 * (q_des_wbc - q_cur) - 20.0 * qd_cur
                tau    = np.clip(tau, wbc.tau_min, wbc.tau_max)
                forces = np.zeros(12)
            else:
                fallback_streak = 0

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

        viewer.sync()

        if step < N_last:
            step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if sim_time - last_log_time >= 0.5:
            last_log_time = sim_time
            pid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            p    = data.xpos[pid]
            com  = data.subtree_com[0]
            fz_l = forces[2] if forces.size >= 3 else 0.0
            fz_r = forces[8] if forces.size >= 9 else 0.0
            fb   = f" [FB:{fallback_streak}]" if fallback_streak > 0 else ""
            print(
                f"  t={sim_time:5.2f}s [{phase}]  "
                f"Z={p[2]:.3f}m  "
                f"CoM_X={com[0]:.3f}  "
                f"tau_max={np.max(np.abs(tau)):.1f}  "
                f"Fz=({fz_l:.0f},{fz_r:.0f})  "
                f"wbc={'Y' if solved else 'N'}{fb}"
            )

        elapsed = time.perf_counter() - t0
        if DT - elapsed > 0:
            time.sleep(DT - elapsed)

print("[Done] Viewer closed.")

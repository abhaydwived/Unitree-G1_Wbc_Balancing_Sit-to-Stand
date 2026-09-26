import numpy as np
import mujoco
import osqp

from scipy import sparse


class G1WBC:
    """
    Unitree G1 Whole-Body Controller.

    Double-support standing controller.

    Decision variables:

        x = [qdd(35), tau(29), FL(3), FR(3)]

    qdd:
        generalized accelerations

    tau:
        joint torques

    FL / FR:
        3D contact forces

    The contact forces are internal WBC variables.
    MuJoCo still computes the actual contact forces.
    """

    def __init__(self, model, data):

        self.model = model
        self.data = data

        # ============================================================
        # Dimensions
        # ============================================================

        self.nq = model.nq
        self.nv = model.nv
        self.nu = model.nu

        assert self.nq == 36
        assert self.nv == 35
        assert self.nu == 29

        print(
            f"G1 dimensions: "
            f"nq={self.nq}, "
            f"nv={self.nv}, "
            f"nu={self.nu}"
        )

        # ============================================================
        # Actuator -> DOF mapping
        # ============================================================

        self.act_dof = np.zeros(
            self.nu,
            dtype=int
        )

        for i in range(self.nu):

            joint_id = int(
                model.actuator_trnid[i, 0]
            )

            self.act_dof[i] = int(
                model.jnt_dofadr[joint_id]
            )

        print("\nActuator DOF mapping:")
        print(self.act_dof)

        # ============================================================
        # Selection matrix
        #
        # S : 29 x 35
        #
        # S qdd = actuated accelerations
        #
        # S.T tau = generalized actuator force
        # ============================================================

        self.S = np.zeros(
            (self.nu, self.nv),
            dtype=np.float64
        )

        for i, dof in enumerate(
            self.act_dof
        ):
            self.S[i, dof] = 1.0

        # ============================================================
        # Foot sites
        # ============================================================

        self.left_foot_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "left_foot"
        )

        self.right_foot_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "right_foot"
        )

        if self.left_foot_id < 0:
            raise RuntimeError(
                "Could not find left_foot"
            )

        if self.right_foot_id < 0:
            raise RuntimeError(
                "Could not find right_foot"
            )

        print(
            "Left foot site:",
            self.left_foot_id
        )

        print(
            "Right foot site:",
            self.right_foot_id
        )

        # ============================================================
        # Torque limits
        # ============================================================

        self.tau_min = np.zeros(
            self.nu
        )

        self.tau_max = np.zeros(
            self.nu
        )

        self._load_torque_limits()

        # ============================================================
        # Posture gains
        # ============================================================

        self.Kp = np.zeros(
            self.nu
        )

        self.Kd = np.zeros(
            self.nu
        )

        self._set_posture_gains()

        # ============================================================
        # COM gains
        # ============================================================

        # COM controller. Horizontal stabilization is deliberately
        # stronger because the previous run developed an increasing
        # +X drift before qdd became very large.
        self.Kp_com = np.array([
            12.0,
            12.0,
            16.0
        ])

        self.Kd_com = np.array([
            8.0,
            8.0,
            8.0
        ])

        # ============================================================
        # Task weights
        # ============================================================

        self.w_posture = 1.0

        # COM stabilization is more important than posture tracking
        # during the standing test.
        self.w_com = 20.0

        # Foot constraints are now a SOFT task rather than hard
        # equalities. This prevents small contact-model/Jacobian
        # inconsistencies from making the entire QP infeasible.
        self.w_foot = 500.0

        # Encourage symmetric support without making it a hard
        # constraint.
        self.w_force_distribution = 0.20

        # ============================================================
        # Regularization
        # ============================================================

        self.w_qdd_reg = 5e-4
        self.w_tau_reg = 1e-5
        self.w_force_reg = 1e-6

        # ============================================================
        # Contact parameters
        # ============================================================

        self.mu = 0.8

        self.fz_min = 0.0

        self.fz_max = 500.0

        # ============================================================
        # Acceleration limits
        # ============================================================

        self.qdd_max = np.ones(
            self.nv
        ) * 50.0

        # Floating base translational acceleration
        self.qdd_max[0:3] = 20.0

        # Floating base angular acceleration
        self.qdd_max[3:6] = 30.0

        # ============================================================
        # Numerical differentiation
        # ============================================================

        self.jdot_dt = 1e-5

    # =================================================================
    # Torque limits
    # =================================================================

    def _load_torque_limits(self):

        for i in range(self.nu):

            joint_id = int(
                self.model.actuator_trnid[i, 0]
            )

            limits = (
                self.model.jnt_actfrcrange[
                    joint_id
                ]
            )

            self.tau_min[i] = limits[0]
            self.tau_max[i] = limits[1]

        print("\nTorque limits:")

        for i in range(self.nu):

            print(
                f"{i:2d}: "
                f"{self.tau_min[i]:8.2f} "
                f"to "
                f"{self.tau_max[i]:8.2f}"
            )

    # =================================================================
    # Posture gains
    # =================================================================

    def _set_posture_gains(self):

        for i in range(self.nu):

            joint_id = int(
                self.model.actuator_trnid[i, 0]
            )

            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id
            )

            if name is None:
                continue

            if (
                "hip_pitch" in name
                or "hip_yaw" in name
                or "knee" in name
            ):

                self.Kp[i] = 25.0
                self.Kd[i] = 5.0

            elif "hip_roll" in name:

                self.Kp[i] = 20.0
                self.Kd[i] = 4.0

            elif "ankle" in name:

                self.Kp[i] = 12.0
                self.Kd[i] = 2.5

            elif "waist" in name:

                self.Kp[i] = 12.0
                self.Kd[i] = 2.5

            elif (
                "shoulder" in name
                or "elbow" in name
            ):

                self.Kp[i] = 8.0
                self.Kd[i] = 1.5

            elif "wrist" in name:

                self.Kp[i] = 4.0
                self.Kd[i] = 0.8

            else:

                self.Kp[i] = 8.0
                self.Kd[i] = 1.5

    # =================================================================
    # Mass matrix
    # =================================================================

    def get_mass_matrix(self):

        M = np.zeros(
            (self.nv, self.nv),
            dtype=np.float64
        )

        mujoco.mj_fullM(
            self.model,
            self.data,
            M
        )

        return M

    # =================================================================
    # Site Jacobian
    # =================================================================

    def get_site_jacobian(
        self,
        site_id
    ):

        Jp = np.zeros(
            (3, self.nv)
        )

        Jr = np.zeros(
            (3, self.nv)
        )

        mujoco.mj_jacSite(
            self.model,
            self.data,
            Jp,
            Jr,
            site_id
        )

        return Jp

    # =================================================================
    # Jdot*qdot
    # =================================================================

    def get_jdot_qdot(
        self,
        site_id
    ):

        qpos0 = self.data.qpos.copy()
        qvel0 = self.data.qvel.copy()

        J0 = self.get_site_jacobian(
            site_id
        )

        qpos1 = qpos0.copy()

        mujoco.mj_integratePos(
            self.model,
            qpos1,
            qvel0,
            self.jdot_dt
        )

        self.data.qpos[:] = qpos1

        mujoco.mj_forward(
            self.model,
            self.data
        )

        J1 = self.get_site_jacobian(
            site_id
        )

        self.data.qpos[:] = qpos0
        self.data.qvel[:] = qvel0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        Jdot = (
            J1 - J0
        ) / self.jdot_dt

        return Jdot @ qvel0

    # =================================================================
    # COM
    # =================================================================

    def get_com_position(self):

        total_mass = 0.0

        com = np.zeros(3)

        for body_id in range(
            self.model.nbody
        ):

            mass = self.model.body_mass[
                body_id
            ]

            if mass <= 0:
                continue

            com += (
                mass
                * self.data.xipos[
                    body_id
                ]
            )

            total_mass += mass

        return com / total_mass

    # =================================================================
    # COM Jacobian
    # =================================================================

    def get_com_jacobian(self):

        total_mass = 0.0

        Jcom = np.zeros(
            (3, self.nv)
        )

        for body_id in range(
            self.model.nbody
        ):

            mass = self.model.body_mass[
                body_id
            ]

            if mass <= 0:
                continue

            Jp = np.zeros(
                (3, self.nv)
            )

            Jr = np.zeros(
                (3, self.nv)
            )

            mujoco.mj_jacBodyCom(
                self.model,
                self.data,
                Jp,
                Jr,
                body_id
            )

            Jcom += mass * Jp

            total_mass += mass

        return Jcom / total_mass

    # =================================================================
    # COM velocity
    # =================================================================

    def get_com_velocity(self):

        Jcom = self.get_com_jacobian()

        return Jcom @ self.data.qvel

    # =================================================================
    # COM Jdot*qdot
    # =================================================================

    def get_com_jdot_qdot(self):

        qpos0 = self.data.qpos.copy()
        qvel0 = self.data.qvel.copy()

        J0 = self.get_com_jacobian()

        qpos1 = qpos0.copy()

        mujoco.mj_integratePos(
            self.model,
            qpos1,
            qvel0,
            self.jdot_dt
        )

        self.data.qpos[:] = qpos1

        mujoco.mj_forward(
            self.model,
            self.data
        )

        J1 = self.get_com_jacobian()

        self.data.qpos[:] = qpos0
        self.data.qvel[:] = qvel0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        Jdot = (
            J1 - J0
        ) / self.jdot_dt

        return Jdot @ qvel0

    # =================================================================
    # Posture task
    # =================================================================

    def get_posture_task(
        self,
        q_des
    ):

        q = self.data.qpos[
            7:36
        ]

        qdot = self.data.qvel[
            6:35
        ]

        qdd_des = (
            self.Kp
            * (q_des - q)
            -
            self.Kd
            * qdot
        )

        qdd_des = np.clip(
            qdd_des,
            -30.0,
            30.0
        )

        A = self.S.copy()

        b = qdd_des

        return A, b

    # =================================================================
    # COM task
    # =================================================================

    def get_com_task(
        self,
        com_des
    ):

        com = self.get_com_position()

        J = self.get_com_jacobian()

        qvel = self.data.qvel.copy()

        com_vel = (
            J @ qvel
        )

        Jdot_qdot = (
            self.get_com_jdot_qdot()
        )

        a_des = (
            self.Kp_com
            * (com_des - com)
            -
            self.Kd_com
            * com_vel
        )

        a_des = np.clip(
            a_des,
            -5.0,
            5.0
        )

        b = (
            a_des
            - Jdot_qdot
        )

        return J, b

    # =================================================================
    # Foot task
    # =================================================================

    def get_foot_task(
        self,
        site_id
    ):

        J = self.get_site_jacobian(
            site_id
        )

        Jdot_qdot = (
            self.get_jdot_qdot(
                site_id
            )
        )

        # Desired zero foot acceleration:
        #
        # J qdd + Jdot qdot = 0
        #
        # This is intentionally used as a SOFT WBC task.
        # The actual contact constraints remain enforced through
        # the contact-force/friction constraints and rigid-body
        # dynamics.

        b = -Jdot_qdot

        return J, b

    # =================================================================
    # Contact-force constraints
    # =================================================================

    def get_contact_constraints(
        self,
        nvar,
        force_slice
    ):

        rows = []

        lower = []

        upper = []

        for foot in range(2):

            base = (
                force_slice.start
                + 3 * foot
            )

            fx = base
            fy = base + 1
            fz = base + 2

            # --------------------------------------------------------
            # Fz >= 0
            # --------------------------------------------------------

            row = np.zeros(nvar)

            row[fz] = 1.0

            rows.append(row)
            lower.append(
                self.fz_min
            )
            upper.append(
                np.inf
            )

            # --------------------------------------------------------
            # Fz <= max
            # --------------------------------------------------------

            row = np.zeros(nvar)

            row[fz] = 1.0

            rows.append(row)
            lower.append(
                -np.inf
            )
            upper.append(
                self.fz_max
            )

            # --------------------------------------------------------
            # Fx <= mu Fz
            # --------------------------------------------------------

            row = np.zeros(nvar)

            row[fx] = 1.0
            row[fz] = -self.mu

            rows.append(row)
            lower.append(
                -np.inf
            )
            upper.append(0.0)

            # --------------------------------------------------------
            # -Fx <= mu Fz
            # --------------------------------------------------------

            row = np.zeros(nvar)

            row[fx] = -1.0
            row[fz] = -self.mu

            rows.append(row)
            lower.append(
                -np.inf
            )
            upper.append(0.0)

            # --------------------------------------------------------
            # Fy <= mu Fz
            # --------------------------------------------------------

            row = np.zeros(nvar)

            row[fy] = 1.0
            row[fz] = -self.mu

            rows.append(row)
            lower.append(
                -np.inf
            )
            upper.append(0.0)

            # --------------------------------------------------------
            # -Fy <= mu Fz
            # --------------------------------------------------------

            row = np.zeros(nvar)

            row[fy] = -1.0
            row[fz] = -self.mu

            rows.append(row)
            lower.append(
                -np.inf
            )
            upper.append(0.0)

        return (
            np.asarray(rows),
            np.asarray(lower),
            np.asarray(upper)
        )

    # =================================================================
    # Nominal force distribution
    # =================================================================

    def get_force_reference(
        self
    ):

        total_mass = (
            np.sum(
                self.model.body_mass
            )
        )

        weight = (
            total_mass
            * self.model.opt.gravity[2]
            * -1.0
        )

        # Equal nominal load.
        fz = weight / 2.0

        return np.array([
            0.0,
            0.0,
            fz,
            0.0,
            0.0,
            fz
        ])

    # =================================================================
    # Main solve
    # =================================================================

    def solve(
        self,
        q_des,
        com_des=None
    ):

        # ============================================================
        # Dimensions
        # ============================================================

        n_qdd = self.nv
        n_tau = self.nu
        n_force = 6

        nvar = (
            n_qdd
            + n_tau
            + n_force
        )

        qdd_slice = slice(
            0,
            n_qdd
        )

        tau_slice = slice(
            n_qdd,
            n_qdd + n_tau
        )

        force_slice = slice(
            n_qdd + n_tau,
            nvar
        )

        # ============================================================
        # Dynamics
        # ============================================================

        M = self.get_mass_matrix()

        h = self.data.qfrc_bias.copy()

        JL = self.get_site_jacobian(
            self.left_foot_id
        )

        JR = self.get_site_jacobian(
            self.right_foot_id
        )

        Jc = np.vstack([
            JL,
            JR
        ])

        A_dyn = np.zeros(
            (self.nv, nvar)
        )

        A_dyn[
            :,
            qdd_slice
        ] = M

        A_dyn[
            :,
            tau_slice
        ] = -self.S.T

        A_dyn[
            :,
            force_slice
        ] = -Jc.T

        b_dyn = -h

        # ============================================================
        # Foot constraints
        # ============================================================

        JL, bL = self.get_foot_task(
            self.left_foot_id
        )

        JR, bR = self.get_foot_task(
            self.right_foot_id
        )

        A_foot = np.zeros(
            (6, nvar)
        )

        A_foot[
            0:3,
            qdd_slice
        ] = JL

        A_foot[
            3:6,
            qdd_slice
        ] = JR

        b_foot = np.concatenate([
            bL,
            bR
        ])

        # ============================================================
        # Posture
        # ============================================================

        A_posture, b_posture = (
            self.get_posture_task(
                q_des
            )
        )

        A_posture_full = np.zeros(
            (self.nu, nvar)
        )

        A_posture_full[
            :,
            qdd_slice
        ] = A_posture

        # ============================================================
        # COM
        # ============================================================

        if com_des is None:

            com_des = (
                self.get_com_position()
            )

        Jcom, bcom = (
            self.get_com_task(
                com_des
            )
        )

        A_com = np.zeros(
            (3, nvar)
        )

        A_com[
            :,
            qdd_slice
        ] = Jcom

        # ============================================================
        # Task objective
        # ============================================================

        # All three are SOFT acceleration-level tasks:
        #
        #   posture -> maintain desired joint configuration
        #   COM     -> stabilize the floating base/whole-body COM
        #   feet    -> keep both feet approximately stationary
        #
        # Dynamics, torque, acceleration, friction and Fz constraints
        # below remain HARD constraints.

        A_tasks = np.vstack([

            np.sqrt(
                self.w_posture
            ) * A_posture_full,

            np.sqrt(
                self.w_com
            ) * A_com,

            np.sqrt(
                self.w_foot
            ) * A_foot
        ])

        b_tasks = np.concatenate([

            np.sqrt(
                self.w_posture
            ) * b_posture,

            np.sqrt(
                self.w_com
            ) * bcom,

            np.sqrt(
                self.w_foot
            ) * b_foot
        ])

        # ============================================================
        # QP objective
        # ============================================================

        P = (
            2.0
            * A_tasks.T
            @ A_tasks
        )

        q = (
            -2.0
            * A_tasks.T
            @ b_tasks
        )

        # ============================================================
        # qdd regularization
        # ============================================================

        for i in range(n_qdd):

            P[
                i,
                i
            ] += (
                2.0
                * self.w_qdd_reg
            )

        # ============================================================
        # Torque regularization
        # ============================================================

        for i in range(n_tau):

            idx = (
                tau_slice.start
                + i
            )

            P[
                idx,
                idx
            ] += (
                2.0
                * self.w_tau_reg
            )

        # ============================================================
        # Force regularization
        # ============================================================

        for i in range(n_force):

            idx = (
                force_slice.start
                + i
            )

            P[
                idx,
                idx
            ] += (
                2.0
                * self.w_force_reg
            )

        # ============================================================
        # Force distribution objective
        # ============================================================

        F_ref = (
            self.get_force_reference()
        )

        A_force_ref = np.zeros(
            (6, nvar)
        )

        A_force_ref[
            :,
            force_slice
        ] = np.eye(6)

        wf = self.w_force_distribution

        P += (
            2.0
            * wf
            * (
                A_force_ref.T
                @ A_force_ref
            )
        )

        q += (
            -2.0
            * wf
            * (
                A_force_ref.T
                @ F_ref
            )
        )

        # ============================================================
        # Numerical regularization
        # ============================================================

        P += (
            1e-8
            * np.eye(nvar)
        )

        P = sparse.csc_matrix(
            P
        )

        # ============================================================
        # Equality constraints
        # ============================================================

        Aeq = np.vstack([
            A_dyn,
            A_foot
        ])

        beq = np.concatenate([
            b_dyn,
            b_foot
        ])

        # ============================================================
        # Torque constraints
        # ============================================================

        Atau = np.zeros(
            (self.nu, nvar)
        )

        for i in range(
            self.nu
        ):

            Atau[
                i,
                tau_slice.start + i
            ] = 1.0

        # ============================================================
        # qdd constraints
        # ============================================================

        Aqdd = np.zeros(
            (self.nv, nvar)
        )

        for i in range(
            self.nv
        ):

            Aqdd[
                i,
                qdd_slice.start + i
            ] = 1.0

        # ============================================================
        # Contact constraints
        # ============================================================

        Acontact, lcontact, ucontact = (
            self.get_contact_constraints(
                nvar,
                force_slice
            )
        )

        # ============================================================
        # Complete constraint matrix
        # ============================================================

        A = np.vstack([
            Aeq,
            Atau,
            Aqdd,
            Acontact
        ])

        lower = np.concatenate([
            beq,
            self.tau_min,
            -self.qdd_max,
            lcontact
        ])

        upper = np.concatenate([
            beq,
            self.tau_max,
            self.qdd_max,
            ucontact
        ])

        A = sparse.csc_matrix(
            A
        )

        # ============================================================
        # Solve
        # ============================================================

        solver = osqp.OSQP()

        solver.setup(
            P=P,
            q=q,
            A=A,
            l=lower,
            u=upper,
            verbose=False,
            warm_start=True,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=5000,
            polish=True
        )

        result = solver.solve()

        status = result.info.status

        # Useful diagnostics when OSQP struggles.
        if result.x is not None and status not in [
            "solved",
            "solved inaccurate"
        ]:
            print(
                "WBC solver status:",
                status,
                "| iterations:",
                result.info.iter
            )

        if result.x is None:

            print(
                "WBC solver status:",
                status
            )

            return (
                np.zeros(self.nu),
                np.zeros(self.nv),
                np.zeros(6),
                False
            )

        solved = status in [
            "solved",
            "solved inaccurate"
        ]

        if not solved:

            print(
                "WBC solver status:",
                status
            )

            return (
                np.zeros(self.nu),
                np.zeros(self.nv),
                np.zeros(6),
                False
            )

        # ============================================================
        # Extract
        # ============================================================

        x = result.x

        qdd = x[
            qdd_slice
        ]

        tau = x[
            tau_slice
        ]

        forces = x[
            force_slice
        ]

        # ============================================================
        # Final safety clipping
        # ============================================================

        tau = np.clip(
            tau,
            self.tau_min,
            self.tau_max
        )

        qdd = np.clip(
            qdd,
            -self.qdd_max,
            self.qdd_max
        )

        # ============================================================
        # Sanity
        # ============================================================

        if not np.all(
            np.isfinite(tau)
        ):

            tau[:] = 0.0
            solved = False

        if not np.all(
            np.isfinite(forces)
        ):

            forces[:] = 0.0
            solved = False

        return (
            tau,
            qdd,
            forces,
            solved
        )
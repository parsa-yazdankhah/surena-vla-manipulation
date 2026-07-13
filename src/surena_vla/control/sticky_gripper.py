"""Kinematic sticky-gripper approximation for the fixed SURENA palm."""

from __future__ import annotations

import mujoco
import numpy as np

class StickyGripper:
    """
    Fake / sticky gripper for Surena's fixed hand.

    It kinematically attaches a freejoint object to right_eef_site while the
    gripper command is closed. This is a bridge-layer grasp approximation for the current Surena model,
    which has no mounted actuated gripper.
    """

    def __init__(self, env, ctrl,
                 object_name_filter: str | None = None,
                 attach_distance: float = 0.09,
                 close_threshold: float = 0.5,
                 release_threshold: float = 0.5,
                 verbose: bool = True):
        self.env = env
        self.ctrl = ctrl
        self.model = ctrl.model
        self.data = ctrl.data

        self.object_name_filter = object_name_filter
        self.attach_distance = float(attach_distance)
        self.close_threshold = float(close_threshold)
        self.release_threshold = float(release_threshold)
        self.verbose = verbose

        self.attached = False
        self.attached_body_id = None
        self.attached_body_name = None
        self.attached_qadr = None

        self.R_eef_obj = None
        self.p_eef_obj = None

    def config(self) -> dict:
        return {
            "object_name_filter": self.object_name_filter,
            "attach_distance": self.attach_distance,
            "close_threshold": self.close_threshold,
            "release_threshold": self.release_threshold,
            "verbose": self.verbose,
        }

    def rebind(self, env, ctrl):
        cfg = self.config()
        self.__init__(env=env, ctrl=ctrl, **cfg)
        return self

    @staticmethod
    def quat_to_mat(q_wxyz):
        R_flat = np.zeros(9)
        mujoco.mju_quat2Mat(R_flat, np.asarray(q_wxyz, dtype=float))
        return R_flat.reshape(3, 3)

    @staticmethod
    def mat_to_quat(R):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(-1))
        return q

    def get_eef_pose_mat(self):
        p, q = self.ctrl.get_eef_pose()
        R = self.quat_to_mat(q)
        return p, R, q

    def get_body_pose_mat(self, body_id):
        mujoco.mj_forward(self.model, self.data)
        p = self.data.xpos[body_id].copy()
        R = self.data.xmat[body_id].reshape(3, 3).copy()
        q = self.mat_to_quat(R)
        return p, R, q

    def freejoint_bodies(self) -> list[dict]:
        bodies = []
        for bid in range(self.model.nbody):
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if bname is None:
                continue
            if self.object_name_filter is not None and self.object_name_filter not in bname:
                continue

            jadr = self.model.body_jntadr[bid]
            jnum = self.model.body_jntnum[bid]
            if jnum <= 0:
                continue

            for k in range(jnum):
                jid = jadr + k
                if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                    qadr = self.model.jnt_qposadr[jid]
                    bodies.append({
                        "body_id": bid,
                        "body_name": bname,
                        "joint_id": jid,
                        "qadr": qadr,
                    })
        return bodies

    def list_candidates(self) -> list[dict]:
        p_eef, _, _ = self.get_eef_pose_mat()
        rows = []
        for item in self.freejoint_bodies():
            bid = item["body_id"]
            p_obj, _, _ = self.get_body_pose_mat(bid)
            dist = float(np.linalg.norm(p_obj - p_eef))
            rows.append({
                "body_id": bid,
                "body_name": item["body_name"],
                "qadr": item["qadr"],
                "dist_to_eef": dist,
                "pos": p_obj.copy(),
            })
        return sorted(rows, key=lambda x: x["dist_to_eef"])

    def print_candidates(self, max_rows: int = 20):
        rows = self.list_candidates()
        print(f"Found {len(rows)} freejoint candidate bodies.")
        for r in rows[:max_rows]:
            print(
                f"body_id={r['body_id']:3d} | "
                f"qadr={r['qadr']:3d} | "
                f"dist={r['dist_to_eef']:.4f} | "
                f"name={r['body_name']} | "
                f"pos={np.round(r['pos'], 4)}"
            )
        return rows

    def nearest_attachable_body(self):
        rows = self.list_candidates()
        if not rows:
            return None
        nearest = rows[0]
        if nearest["dist_to_eef"] > self.attach_distance:
            return None
        return nearest

    def attach(self, body_info=None) -> bool:
        if self.attached:
            return True
        if body_info is None:
            body_info = self.nearest_attachable_body()
        if body_info is None:
            if self.verbose:
                print("[StickyGripper] No attachable object near EEF.")
            return False

        bid = body_info["body_id"]
        qadr = body_info["qadr"]
        bname = body_info["body_name"]

        p_eef, R_eef, _ = self.get_eef_pose_mat()
        p_obj, R_obj, _ = self.get_body_pose_mat(bid)

        self.R_eef_obj = R_eef.T @ R_obj
        self.p_eef_obj = R_eef.T @ (p_obj - p_eef)

        self.attached = True
        self.attached_body_id = bid
        self.attached_body_name = bname
        self.attached_qadr = qadr

        self.zero_object_velocity()

        if self.verbose:
            print(f"[StickyGripper] ATTACHED: {bname} | dist={body_info['dist_to_eef']:.4f} m")
        return True

    def release(self):
        if not self.attached:
            return
        if self.verbose:
            print(f"[StickyGripper] RELEASED: {self.attached_body_name}")
        self.attached = False
        self.attached_body_id = None
        self.attached_body_name = None
        self.attached_qadr = None
        self.R_eef_obj = None
        self.p_eef_obj = None

    def zero_object_velocity(self):
        if self.attached_body_id is None:
            return
        bid = self.attached_body_id
        jadr = self.model.body_jntadr[bid]
        jid = jadr
        dadr = self.model.jnt_dofadr[jid]
        self.data.qvel[dadr:dadr + 6] = 0.0

    def enforce_attachment(self):
        if not self.attached:
            return

        p_eef, R_eef, _ = self.get_eef_pose_mat()
        p_obj = p_eef + R_eef @ self.p_eef_obj
        R_obj = R_eef @ self.R_eef_obj
        q_obj = self.mat_to_quat(R_obj)

        qadr = self.attached_qadr
        self.data.qpos[qadr:qadr + 3] = p_obj
        self.data.qpos[qadr + 3:qadr + 7] = q_obj
        self.zero_object_velocity()
        mujoco.mj_forward(self.model, self.data)

    def update(self, gripper_action: float) -> dict:
        """
        Update sticky grasp state from the scalar gripper command.

        Strict rule used for Surena VLA episodes:
            gripper_action > close_threshold  -> sticky grasp is allowed
            gripper_action <= close_threshold -> no sticky grasp is allowed

        This avoids accidental attachment when the palm merely passes close to
        an object. The gripper command must explicitly close.
        """
        g = float(gripper_action)
        closed = g > self.close_threshold

        if not closed:
            # Strict project rule: anything <= 0.5 means the hand should not
            # stick to anything, even if it was attached on an earlier step.
            if self.attached:
                self.release()
            return {
                "attached": False,
                "body_name": None,
            }

        if self.attached:
            self.enforce_attachment()
        else:
            self.attach()

        return {
            "attached": self.attached,
            "body_name": self.attached_body_name,
        }

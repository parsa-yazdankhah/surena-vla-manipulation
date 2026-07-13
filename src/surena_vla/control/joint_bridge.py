"""Joint-command bridge from the validated SURENA controller."""

from __future__ import annotations

import queue
import threading

import mujoco
import numpy as np

from .constants import (
    ARRAY_LEN, GAZEBO_INDEX_MAP_BARE, ARM_INDICES, JOINT_LIMITS, HOME_QPOS
)
from .mujoco_utils import clamp_joints, make_gazebo_array, _mj_id

class GazeboStyleController:
    """
    Translates a 29-element joint-angle array (or a 7-element arm array)
    into MuJoCo data.ctrl writes, using the same index mapping as
    gazebo_bridge.cpp.

    Parameters
    ----------
    model, data : mujoco.MjModel / MjData
        May come from a standalone sim or from env.env.sim.model._model / ._data.
    prefix : str
        Name prefix that robosuite prepends to every element.
        Use "" for standalone sims, "robot0_" inside LIBERO/robosuite.
    apply_home : bool
        If True (default), immediately set qpos + ctrl to HOME_QPOS.
        Pass False when the env resets qpos itself (e.g. inside LIBERO).
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 prefix: str = "", apply_home: bool = True):
        self.model  = model
        self.data   = data
        self.prefix = prefix

        # Build prefixed index map
        self._index_map = {
            gidx: (prefix + aname, prefix + jname)
            for gidx, (aname, jname) in GAZEBO_INDEX_MAP_BARE.items()
        }

        # Resolve actuator IDs
        self._act_id: dict[int, int] = {}
        for gidx, (act_name, _) in self._index_map.items():
            self._act_id[gidx] = _mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)

        # Resolve joint qpos addresses
        self._qadr: dict[int, int] = {}
        for gidx, (_, jname) in self._index_map.items():
            jid = _mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self._qadr[gidx] = model.jnt_qposadr[jid]

        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._current_arr = make_gazebo_array(HOME_QPOS)

        if apply_home:
            self._apply_init_pose()

    def _apply_init_pose(self):
        q0 = HOME_QPOS
        for slot, gidx in enumerate(ARM_INDICES):
            self.data.qpos[self._qadr[gidx]] = q0[slot]
            self.data.ctrl[self._act_id[gidx]] = q0[slot]
        mujoco.mj_forward(self.model, self.data)
        print(f"[GazeboCtrl] HOME pose applied (prefix='{self.prefix}').")

    # ── Producer API ──────────────────────────────────────────────────

    def publish(self, array_29: np.ndarray):
        if len(array_29) < ARRAY_LEN:
            return
        arr = np.asarray(array_29, dtype=float).copy()
        try:
            self._queue.put_nowait(arr)
        except queue.Full:
            try: self._queue.get_nowait()
            except queue.Empty: pass
            self._queue.put_nowait(arr)

    def publish_arm_qpos(self, arm_q: np.ndarray):
        self.publish(make_gazebo_array(clamp_joints(arm_q)))

    # ── Consumer API (call once per control tick) ─────────────────────

    def control_callback(self):
        latest = None
        while not self._queue.empty():
            try: latest = self._queue.get_nowait()
            except queue.Empty: break
        if latest is not None:
            self._current_arr = latest

        arr = self._current_arr
        for gidx, aid in self._act_id.items():
            val = float(arr[gidx])
            val = float(np.clip(val,
                                JOINT_LIMITS[ARM_INDICES.index(gidx), 0],
                                JOINT_LIMITS[ARM_INDICES.index(gidx), 1]))
            self.data.ctrl[aid] = val

    # ── Convenience getters ───────────────────────────────────────────

    def get_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[self._qadr[gi]] for gi in ARM_INDICES])

    def get_arm_ctrl(self) -> np.ndarray:
        return np.array([self.data.ctrl[self._act_id[gi]] for gi in ARM_INDICES])

    def set_joint_pose(self, arm_q_7: np.ndarray):
        """Immediately command a 7-joint pose (bypasses queue — use in Jupyter)."""
        q = clamp_joints(arm_q_7)
        for slot, gidx in enumerate(ARM_INDICES):
            self.data.qpos[self._qadr[gidx]] = q[slot]
            self.data.ctrl[self._act_id[gidx]] = q[slot]
        mujoco.mj_forward(self.model, self.data)

    # ── ROS subscriber ─────────────────────────────────────────────────

    def start_ros_subscriber(self,
                            topic: str = "/joint_angles_gazebo",
                            node_name: str = "surena_mujoco_bridge"):
        """
        Spin a background thread that subscribes to a ROS Float64MultiArray
        topic and feeds arrays into the bridge queue.
        The caller drives the sim (env.step / rs_sim.step); this thread
        only feeds joint angles into the queue at whatever rate ROS publishes.

        Usage in Jupyter:
            ctrl.bridge.start_ros_subscriber()
            while True:
                ctrl.bridge.control_callback()   # drains queue → data.ctrl
                env.sim.step()

        Parameters
        ----------
        topic     : ROS topic name (default /joint_angles_gazebo)
        node_name : ROS node name (ignored if a node already exists)
        """
        if getattr(self, "_ros_thread", None) is not None:
            print("[Bridge] ROS subscriber already running.")
            return

        import threading

        def _ros_spin():
            try:
                import rospy
                from std_msgs.msg import Float64MultiArray
            except ImportError:
                raise RuntimeError(
                    "rospy not found. Source your ROS workspace before starting the ROS subscriber."
                )

            try:
                rospy.init_node(node_name, anonymous=True, disable_signals=True)
            except rospy.exceptions.ROSException:
                pass 

            def _cb(msg):
                self.publish(np.array(msg.data))
                print(f"{msg.data}")

            rospy.Subscriber(topic, Float64MultiArray, _cb, queue_size=1)
            rospy.spin()

        self._ros_thread = threading.Thread(
            target=_ros_spin,
            daemon=True,
            name="ROSSubscriber",
        )
        self._ros_thread.start()

    def stop_ros_subscriber(self):
        """Shut down the ROS subscriber thread (best-effort)."""
        try:
            import rospy
            rospy.signal_shutdown("stop_ros_subscriber called")
        except Exception:
            pass
        self._ros_thread = None
        print("[Bridge] ROS subscriber stopped.")

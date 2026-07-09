#!/usr/bin/env python3
"""
Radialer Sliding-Test mit passiver Fasenerkennung im Prozess-KOS sliding_frame.

Modi
----
mode=validate:
    Kontaktfreier Freiraumtest zwischen start_joints und slide_end_joints.
    Die gemessene TCP-Verschiebung wird im sliding_frame ausgewertet.
    Kein Force Mode.

mode=slide:
    Startpose -> Tare -> Bag -> Force Mode in z_R -> Kontakterkennung ->
    Stabilisierung -> Gelenktrajektorie zum kalibrierten Endpunkt ->
    Force Mode stoppen -> Rückzug zur Startpose -> Bag stoppen.

Wichtig
-------
FollowJointTrajectory akzeptiert Gelenkziele, keine kartesischen Zielpunkte.
Deshalb muss slide_end_joints einmal so kalibriert werden, dass die gemessene
TCP-Verschiebung im sliding_frame ungefähr [slide_distance_m, 0, 0] ergibt.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Sequence

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import TransformStamped, WrenchStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformException,
    TransformListener,
)
from trajectory_msgs.msg import JointTrajectoryPoint
from ur_msgs.srv import SetForceMode


JOINT_NAMES = [
    "elbow_joint",
    "shoulder_lift_joint",
    "shoulder_pan_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def duration_msg(seconds: float) -> Duration:
    if seconds <= 0.0:
        raise ValueError("Trajectory duration must be > 0.")
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def quat_angle(q1: Sequence[float], q2: Sequence[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


class RadialSlidingChamferObserver(Node):
    def __init__(self) -> None:
        super().__init__("radial_sliding_chamfer_observer")

        # Mode
        self.declare_parameter("mode", "validate")  # validate | slide
        self.declare_parameter("free_space_validated", False)

        # Frames
        self.declare_parameter("sliding_frame", "sliding_frame")
        self.declare_parameter("tcp_frame", "tool0_controller")
        self.declare_parameter("publish_sliding_frame", True)
        self.declare_parameter("sliding_parent_frame", "base")
        self.declare_parameter("sliding_tf_x_m", -0.22737)
        self.declare_parameter("sliding_tf_y_m", -0.40030)
        self.declare_parameter("sliding_tf_z_m", 0.03517)
        self.declare_parameter("sliding_tf_qx", -0.029727)
        self.declare_parameter("sliding_tf_qy", 0.047563)
        self.declare_parameter("sliding_tf_qz", -0.975264)
        self.declare_parameter("sliding_tf_qw", -0.213808)

        # Start pose definition and calibrated joints
        self.declare_parameter("start_x_m", 0.0)
        self.declare_parameter("start_y_m", 0.0)
        self.declare_parameter("start_z_m", 0.002)
        self.declare_parameter("start_qx", 0.056396)
        self.declare_parameter("start_qy", 0.030219)
        self.declare_parameter("start_qz", 0.280641)
        self.declare_parameter("start_qw", 0.957678)
        self.declare_parameter("start_joints_calibrated", True)
        self.declare_parameter(
            "start_joints",
            [
                1.4617331663714808,
                -1.1869505506804963,
                0.8272621631622314,
                -1.889268537560934,
                -1.536943260823385,
                -0.22527915636171514,
            ],
        )

        # Must be calibrated by the user
        self.declare_parameter("slide_end_joints_calibrated", False)
        self.declare_parameter("slide_end_joints", [0.0] * 6)

        # Motion and validation
        self.declare_parameter("move_duration_s", 4.0)
        self.declare_parameter("slide_distance_m", 0.003)
        self.declare_parameter("radial_speed_mps", 0.001)
        self.declare_parameter("radial_tolerance_m", 0.0005)
        self.declare_parameter("cross_axis_tolerance_m", 0.0005)
        self.declare_parameter("orientation_tolerance_rad", math.radians(3.0))
        self.declare_parameter("position_tolerance_m", 0.001)

        # Force mode and contact detection
        self.declare_parameter("normal_force_n", -5.0)
        self.declare_parameter("contact_threshold_n", 2.0)
        self.declare_parameter("contact_persistence_s", 0.05)
        self.declare_parameter("contact_timeout_s", 3.0)
        self.declare_parameter("stabilisation_after_contact_s", 0.5)
        self.declare_parameter("speed_limit_z_mps", 0.005)
        self.declare_parameter("max_abs_normal_force_n", 12.0)
        self.declare_parameter("wrench_timeout_s", 0.25)
        self.declare_parameter("deviation_xy_m", 0.002)
        self.declare_parameter("deviation_z_m", 0.005)
        self.declare_parameter("deviation_angle_rad", 0.05)
        self.declare_parameter("damping_factor", 0.5)
        self.declare_parameter("gain_scaling", 0.3)
        self.declare_parameter(
            "wrench_topic",
            "/force_torque_sensor_broadcaster/wrench",
        )

        # Passive chamfer detection: force, position and combined
        self.declare_parameter("chamfer_detection_enabled", True)
        self.declare_parameter("chamfer_reference_window_s", 0.30)
        self.declare_parameter("chamfer_ignore_initial_s", 0.50)
        self.declare_parameter("chamfer_min_reference_samples", 10)
        self.declare_parameter("chamfer_force_drop_threshold_n", 1.50)
        self.declare_parameter("chamfer_position_drop_threshold_m", 0.00030)
        self.declare_parameter("chamfer_persistence_s", 0.05)
        self.declare_parameter(
            "chamfer_detection_topic",
            "/radial_sliding_chamfer_observer/chamfer_detection_state",
        )

        # Bag
        self.declare_parameter(
            "bag_root",
            str(Path.home() / "masterarbeit" / "bags"),
        )
        self.declare_parameter("bag_name", "radial_sliding_01")
        self.declare_parameter("bag_start_wait_s", 2.0)
        self.declare_parameter("tare_wait_s", 1.0)

        # Interfaces
        self.scaled_action = (
            "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )
        self.passthrough_action = (
            "/passthrough_trajectory_controller/follow_joint_trajectory"
        )
        self.scaled_client = ActionClient(
            self, FollowJointTrajectory, self.scaled_action
        )
        self.passthrough_client = ActionClient(
            self, FollowJointTrajectory, self.passthrough_action
        )

        self.switch_client = self.create_client(
            SwitchController,
            "/controller_manager/switch_controller",
        )
        self.list_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
        )
        self.zero_client = self.create_client(
            Trigger,
            "/io_and_status_controller/zero_ftsensor",
        )
        self.force_start_client = self.create_client(
            SetForceMode,
            "/force_mode_controller/start_force_mode",
        )
        self.force_stop_client = self.create_client(
            Trigger,
            "/force_mode_controller/stop_force_mode",
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.current_normal_force_n: Optional[float] = None
        self.last_wrench_monotonic: Optional[float] = None
        self.create_subscription(
            WrenchStamped,
            str(self.p("wrench_topic")),
            self._wrench_cb,
            50,
        )

        self.detection_publisher = self.create_publisher(
            Float64MultiArray,
            str(self.p("chamfer_detection_topic")),
            50,
        )
        self._reset_chamfer_detector()

        self.bag_process: Optional[subprocess.Popen] = None
        self.force_mode_active = False

        if bool(self.p("publish_sliding_frame")):
            self.publish_sliding_frame()

    def p(self, name: str):
        return self.get_parameter(name).value

    def publish_sliding_frame(self) -> None:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.p("sliding_parent_frame"))
        msg.child_frame_id = str(self.p("sliding_frame"))
        msg.transform.translation.x = float(self.p("sliding_tf_x_m"))
        msg.transform.translation.y = float(self.p("sliding_tf_y_m"))
        msg.transform.translation.z = float(self.p("sliding_tf_z_m"))
        msg.transform.rotation.x = float(self.p("sliding_tf_qx"))
        msg.transform.rotation.y = float(self.p("sliding_tf_qy"))
        msg.transform.rotation.z = float(self.p("sliding_tf_qz"))
        msg.transform.rotation.w = float(self.p("sliding_tf_qw"))
        self.static_tf_broadcaster.sendTransform(msg)
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)

    @staticmethod
    def rotate_vector(
        v: Sequence[float],
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> tuple[float, float, float]:
        vx, vy, vz = v
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return (
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        )

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        source = msg.header.frame_id or "tool0"
        try:
            tf = self.tf_buffer.lookup_transform(
                str(self.p("sliding_frame")),
                source,
                Time(),
            )
        except TransformException:
            return

        q = tf.transform.rotation
        f = self.rotate_vector(
            (
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
            ),
            q.x,
            q.y,
            q.z,
            q.w,
        )
        self.current_normal_force_n = float(f[2])
        self.last_wrench_monotonic = time.monotonic()

    def _reset_chamfer_detector(self) -> None:
        self.slide_started_monotonic: Optional[float] = None
        self.slide_start_position: Optional[list[float]] = None
        self.force_reference_history: Deque[tuple[float, float]] = deque()
        self.z_reference_history: Deque[tuple[float, float]] = deque()

        self.force_candidate_since: Optional[float] = None
        self.position_candidate_since: Optional[float] = None
        self.combined_candidate_since: Optional[float] = None

        self.force_detection: Optional[dict] = None
        self.position_detection: Optional[dict] = None
        self.combined_detection: Optional[dict] = None

        self.latest_detection_sample: Optional[dict] = None

    def _start_chamfer_detector(self, start_position: Sequence[float]) -> None:
        self._reset_chamfer_detector()
        self.slide_started_monotonic = time.monotonic()
        self.slide_start_position = [float(value) for value in start_position]
        self.get_logger().info(
            "Passive Fasenerkennung aktiviert: Kraftabfall, "
            "Positionsabfall und Kombination werden parallel beobachtet."
        )

    def _try_tcp_position(self) -> Optional[list[float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                str(self.p("sliding_frame")),
                str(self.p("tcp_frame")),
                Time(),
            )
        except TransformException:
            return None

        translation = tf.transform.translation
        return [
            float(translation.x),
            float(translation.y),
            float(translation.z),
        ]

    @staticmethod
    def _window_mean(history: Deque[tuple[float, float]]) -> Optional[float]:
        if not history:
            return None
        return sum(value for _, value in history) / len(history)

    def _trim_reference_history(
        self,
        history: Deque[tuple[float, float]],
        now: float,
    ) -> None:
        cutoff = now - float(self.p("chamfer_reference_window_s"))
        while history and history[0][0] < cutoff:
            history.popleft()

    def _make_detection_event(
        self,
        detector: str,
        now: float,
        tcp_position: Sequence[float],
        force_abs_n: float,
        force_reference_n: float,
        force_drop_n: float,
        z_reference_m: float,
        position_drop_m: float,
    ) -> dict:
        if self.slide_started_monotonic is None:
            raise RuntimeError("Sliding start time unavailable.")
        if self.slide_start_position is None:
            raise RuntimeError("Sliding start position unavailable.")

        return {
            "detector": detector,
            "time_from_slide_start_s": now - self.slide_started_monotonic,
            "x_from_slide_start_m": (
                float(tcp_position[0]) - self.slide_start_position[0]
            ),
            "y_from_slide_start_m": (
                float(tcp_position[1]) - self.slide_start_position[1]
            ),
            "z_from_slide_start_m": (
                float(tcp_position[2]) - self.slide_start_position[2]
            ),
            "force_abs_n": force_abs_n,
            "force_reference_n": force_reference_n,
            "force_drop_n": force_drop_n,
            "z_reference_m": z_reference_m,
            "position_drop_m": position_drop_m,
        }

    def _update_persistent_detector(
        self,
        condition: bool,
        candidate_attribute: str,
        detection_attribute: str,
        detector_name: str,
        now: float,
        tcp_position: Sequence[float],
        force_abs_n: float,
        force_reference_n: float,
        force_drop_n: float,
        z_reference_m: float,
        position_drop_m: float,
    ) -> None:
        if getattr(self, detection_attribute) is not None:
            return

        if not condition:
            setattr(self, candidate_attribute, None)
            return

        candidate_since = getattr(self, candidate_attribute)
        if candidate_since is None:
            setattr(self, candidate_attribute, now)
            return

        if now - candidate_since < float(self.p("chamfer_persistence_s")):
            return

        event = self._make_detection_event(
            detector_name,
            now,
            tcp_position,
            force_abs_n,
            force_reference_n,
            force_drop_n,
            z_reference_m,
            position_drop_m,
        )
        setattr(self, detection_attribute, event)

        self.get_logger().warning(
            f"Fasenkandidat [{detector_name}] erkannt bei "
            f"x_R={event['x_from_slide_start_m'] * 1000.0:.3f} mm, "
            f"F-Abfall={force_drop_n:.3f} N, "
            f"z-Abfall={position_drop_m * 1000.0:.3f} mm."
        )

    def update_chamfer_detector(self) -> None:
        if not bool(self.p("chamfer_detection_enabled")):
            return
        if self.slide_started_monotonic is None:
            return
        if self.current_normal_force_n is None:
            return

        tcp_position = self._try_tcp_position()
        if tcp_position is None:
            return

        now = time.monotonic()
        elapsed = now - self.slide_started_monotonic
        force_abs_n = abs(float(self.current_normal_force_n))
        current_z_m = float(tcp_position[2])

        self._trim_reference_history(self.force_reference_history, now)
        self._trim_reference_history(self.z_reference_history, now)

        min_samples = int(self.p("chamfer_min_reference_samples"))
        force_reference_n = self._window_mean(self.force_reference_history)
        z_reference_m = self._window_mean(self.z_reference_history)

        reference_ready = (
            force_reference_n is not None
            and z_reference_m is not None
            and len(self.force_reference_history) >= min_samples
            and len(self.z_reference_history) >= min_samples
        )
        detector_enabled_now = (
            elapsed >= float(self.p("chamfer_ignore_initial_s"))
            and reference_ready
        )

        force_drop_n = (
            float(force_reference_n) - force_abs_n
            if force_reference_n is not None else 0.0
        )
        position_drop_m = (
            float(z_reference_m) - current_z_m
            if z_reference_m is not None else 0.0
        )

        force_condition = (
            detector_enabled_now
            and force_drop_n
            >= float(self.p("chamfer_force_drop_threshold_n"))
        )
        position_condition = (
            detector_enabled_now
            and position_drop_m
            >= float(self.p("chamfer_position_drop_threshold_m"))
        )
        combined_condition = force_condition and position_condition

        if reference_ready:
            self._update_persistent_detector(
                force_condition,
                "force_candidate_since",
                "force_detection",
                "force_only",
                now,
                tcp_position,
                force_abs_n,
                float(force_reference_n),
                force_drop_n,
                float(z_reference_m),
                position_drop_m,
            )
            self._update_persistent_detector(
                position_condition,
                "position_candidate_since",
                "position_detection",
                "position_only",
                now,
                tcp_position,
                force_abs_n,
                float(force_reference_n),
                force_drop_n,
                float(z_reference_m),
                position_drop_m,
            )
            self._update_persistent_detector(
                combined_condition,
                "combined_candidate_since",
                "combined_detection",
                "combined",
                now,
                tcp_position,
                force_abs_n,
                float(force_reference_n),
                force_drop_n,
                float(z_reference_m),
                position_drop_m,
            )

        # Freeze the corresponding reference while a possible event is active.
        if not force_condition:
            self.force_reference_history.append((now, force_abs_n))
        if not position_condition:
            self.z_reference_history.append((now, current_z_m))

        self.latest_detection_sample = {
            "time_from_slide_start_s": elapsed,
            "force_abs_n": force_abs_n,
            "force_reference_n": (
                float(force_reference_n)
                if force_reference_n is not None else math.nan
            ),
            "force_drop_n": force_drop_n,
            "z_m": current_z_m,
            "z_reference_m": (
                float(z_reference_m)
                if z_reference_m is not None else math.nan
            ),
            "position_drop_m": position_drop_m,
            "force_condition": force_condition,
            "position_condition": position_condition,
            "combined_condition": combined_condition,
        }

        message = Float64MultiArray()
        message.data = [
            elapsed,
            force_abs_n,
            (
                float(force_reference_n)
                if force_reference_n is not None else math.nan
            ),
            force_drop_n,
            current_z_m,
            (
                float(z_reference_m)
                if z_reference_m is not None else math.nan
            ),
            position_drop_m,
            1.0 if force_condition else 0.0,
            1.0 if position_condition else 0.0,
            1.0 if combined_condition else 0.0,
            1.0 if self.force_detection is not None else 0.0,
            1.0 if self.position_detection is not None else 0.0,
            1.0 if self.combined_detection is not None else 0.0,
        ]
        self.detection_publisher.publish(message)

    def write_chamfer_detection_result(self, bag_path: Path) -> Path:
        result_path = (
            bag_path.parent
            / f"{bag_path.name}_chamfer_detection_result.json"
        )
        payload = {
            "bag_name": bag_path.name,
            "mode": "passive_observation",
            "thresholds": {
                "reference_window_s": float(
                    self.p("chamfer_reference_window_s")
                ),
                "ignore_initial_s": float(
                    self.p("chamfer_ignore_initial_s")
                ),
                "force_drop_threshold_n": float(
                    self.p("chamfer_force_drop_threshold_n")
                ),
                "position_drop_threshold_m": float(
                    self.p("chamfer_position_drop_threshold_m")
                ),
                "persistence_s": float(
                    self.p("chamfer_persistence_s")
                ),
            },
            "force_only": self.force_detection,
            "position_only": self.position_detection,
            "combined": self.combined_detection,
            "topic_layout": {
                "0": "time_from_slide_start_s",
                "1": "force_abs_n",
                "2": "force_reference_n",
                "3": "force_drop_n",
                "4": "z_m",
                "5": "z_reference_m",
                "6": "position_drop_m",
                "7": "force_condition",
                "8": "position_condition",
                "9": "combined_condition",
                "10": "force_detected_latched",
                "11": "position_detected_latched",
                "12": "combined_detected_latched",
            },
        }
        with result_path.open("w", encoding="utf-8") as result_file:
            json.dump(payload, result_file, indent=2, ensure_ascii=False)
        return result_path

    def wait_interfaces(self) -> None:
        if not self.scaled_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Scaled trajectory action unavailable.")
        if not self.passthrough_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Passthrough trajectory action unavailable.")
        for client, name in (
            (self.switch_client, "switch_controller"),
            (self.list_client, "list_controllers"),
            (self.zero_client, "zero_ftsensor"),
            (self.force_start_client, "start_force_mode"),
            (self.force_stop_client, "stop_force_mode"),
        ):
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f"Service unavailable: {name}")

    def controller_states(self) -> dict[str, str]:
        future = self.list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            raise RuntimeError("No controller list received.")
        return {c.name: c.state for c in result.controller}

    def switch(
        self,
        activate: Sequence[str],
        deactivate: Sequence[str],
    ) -> None:
        states = self.controller_states()
        activate_needed = [n for n in activate if states.get(n) != "active"]
        deactivate_needed = [n for n in deactivate if states.get(n) == "active"]
        if not activate_needed and not deactivate_needed:
            return

        req = SwitchController.Request()
        req.activate_controllers = activate_needed
        req.deactivate_controllers = deactivate_needed
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = True
        future = self.switch_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.ok:
            raise RuntimeError(
                f"Controller switch failed: activate={activate_needed}, "
                f"deactivate={deactivate_needed}"
            )
        time.sleep(0.5)

    def switch_scaled(self) -> None:
        self.switch(
            ["scaled_joint_trajectory_controller"],
            [
                "passthrough_trajectory_controller",
                "force_mode_controller",
            ],
        )

    def switch_force(self) -> None:
        self.switch(
            [
                "passthrough_trajectory_controller",
                "force_mode_controller",
            ],
            ["scaled_joint_trajectory_controller"],
        )

    def send_goal(
        self,
        client: ActionClient,
        joints: Sequence[float],
        duration_s: float,
        monitor_force: bool = False,
    ) -> None:
        if len(joints) != 6:
            raise ValueError("Exactly six joint positions required.")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in joints]
        point.time_from_start = duration_msg(duration_s)
        goal.trajectory.points = [point]

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Trajectory goal rejected.")

        result_future = handle.get_result_async()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.005)
            if monitor_force:
                self.assert_force_safe()
                self.update_chamfer_detector()
        if result_future.result() is None:
            raise RuntimeError("Trajectory returned no result.")

    def lookup_tcp(self):
        deadline = time.monotonic() + 5.0
        last_error = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            try:
                return self.tf_buffer.lookup_transform(
                    str(self.p("sliding_frame")),
                    str(self.p("tcp_frame")),
                    Time(),
                )
            except TransformException as exc:
                last_error = exc
        raise RuntimeError(f"TCP TF unavailable: {last_error}")

    def pose_vector(self) -> tuple[list[float], list[float]]:
        tf = self.lookup_tcp().transform
        return (
            [tf.translation.x, tf.translation.y, tf.translation.z],
            [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w],
        )

    def validate_start(self) -> None:
        pos, quat = self.pose_vector()
        target_pos = [
            float(self.p("start_x_m")),
            float(self.p("start_y_m")),
            float(self.p("start_z_m")),
        ]
        target_q = [
            float(self.p("start_qx")),
            float(self.p("start_qy")),
            float(self.p("start_qz")),
            float(self.p("start_qw")),
        ]
        pos_error = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(pos, target_pos))
        )
        ori_error = quat_angle(quat, target_q)
        self.get_logger().info(
            f"Startpose: p={pos}, position error={pos_error*1000:.3f} mm, "
            f"orientation error={math.degrees(ori_error):.3f} deg"
        )
        if pos_error > float(self.p("position_tolerance_m")):
            raise RuntimeError("Start position outside tolerance.")
        if ori_error > float(self.p("orientation_tolerance_rad")):
            raise RuntimeError("Start orientation outside tolerance.")

    def start_joints(self) -> list[float]:
        if not bool(self.p("start_joints_calibrated")):
            raise RuntimeError("start_joints are not calibrated.")
        return [float(x) for x in self.p("start_joints")]

    def end_joints(self) -> list[float]:
        if not bool(self.p("slide_end_joints_calibrated")):
            raise RuntimeError(
                "slide_end_joints_calibrated is false. "
                "Calibrate and validate the radial endpoint first."
            )
        return [float(x) for x in self.p("slide_end_joints")]

    def validate_radial_displacement(
        self,
        start_pos: Sequence[float],
        end_pos: Sequence[float],
    ) -> dict:
        delta = [b - a for a, b in zip(start_pos, end_pos)]
        expected = float(self.p("slide_distance_m"))
        radial_error = abs(delta[0] - expected)
        cross_ok = (
            abs(delta[1]) <= float(self.p("cross_axis_tolerance_m"))
            and abs(delta[2]) <= float(self.p("cross_axis_tolerance_m"))
        )
        radial_ok = radial_error <= float(self.p("radial_tolerance_m"))

        result = {
            "dx_R_m": delta[0],
            "dy_R_m": delta[1],
            "dz_R_m": delta[2],
            "radial_error_m": radial_error,
            "valid": bool(radial_ok and cross_ok),
        }
        self.get_logger().info(
            "Measured displacement: "
            f"dx_R={delta[0]*1000:.3f} mm, "
            f"dy_R={delta[1]*1000:.3f} mm, "
            f"dz_R={delta[2]*1000:.3f} mm"
        )
        if not result["valid"]:
            raise RuntimeError(
                "Sliding endpoint is not sufficiently radial in sliding_frame."
            )
        return result

    def zero_sensor(self) -> None:
        future = self.zero_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError("FT sensor tare failed.")
        time.sleep(float(self.p("tare_wait_s")))

    def start_force_mode(self) -> None:
        req = SetForceMode.Request()
        req.task_frame.header.frame_id = str(self.p("sliding_frame"))
        req.task_frame.pose.orientation.w = 1.0
        req.selection_vector_z = True
        req.wrench.force.z = float(self.p("normal_force_n"))
        req.type = SetForceMode.Request.NO_TRANSFORM
        req.speed_limits.linear.z = float(self.p("speed_limit_z_mps"))
        req.deviation_limits = [
            float(self.p("deviation_xy_m")),
            float(self.p("deviation_xy_m")),
            float(self.p("deviation_z_m")),
            float(self.p("deviation_angle_rad")),
            float(self.p("deviation_angle_rad")),
            float(self.p("deviation_angle_rad")),
        ]
        req.damping_factor = float(self.p("damping_factor"))
        req.gain_scaling = float(self.p("gain_scaling"))

        future = self.force_start_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError("Force mode start failed.")
        self.force_mode_active = True

    def stop_force_mode(self) -> None:
        if not self.force_mode_active:
            return
        future = self.force_stop_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        self.force_mode_active = False

    def assert_force_safe(self) -> None:
        if (
            self.last_wrench_monotonic is None
            or time.monotonic() - self.last_wrench_monotonic
            > float(self.p("wrench_timeout_s"))
        ):
            raise RuntimeError("Wrench data stale.")
        if self.current_normal_force_n is None:
            raise RuntimeError("No normal force available.")
        if abs(self.current_normal_force_n) > float(
            self.p("max_abs_normal_force_n")
        ):
            raise RuntimeError(
                f"Force safety limit exceeded: "
                f"Fz_R={self.current_normal_force_n:.3f} N"
            )

    def wait_contact(self) -> None:
        threshold = float(self.p("contact_threshold_n"))
        persistence = float(self.p("contact_persistence_s"))
        deadline = time.monotonic() + float(self.p("contact_timeout_s"))
        above_since: Optional[float] = None

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.005)
            self.assert_force_safe()
            value = abs(float(self.current_normal_force_n))
            if value >= threshold:
                if above_since is None:
                    above_since = time.monotonic()
                elif time.monotonic() - above_since >= persistence:
                    self.get_logger().info(
                        f"Contact detected: Fz_R={self.current_normal_force_n:.3f} N"
                    )
                    return
            else:
                above_since = None
        raise RuntimeError("Contact timeout.")

    def stabilise_contact(self) -> None:
        duration = float(self.p("stabilisation_after_contact_s"))
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.005)
            self.assert_force_safe()

    def start_bag(self) -> Path:
        root = Path(str(self.p("bag_root"))).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        path = root / str(self.p("bag_name"))
        if path.exists():
            raise FileExistsError(f"Bag already exists: {path}")

        command = [
            "ros2", "bag", "record", "-s", "mcap",
            "/tf", "/tf_static", "/joint_states",
            "/tcp_pose_broadcaster/pose",
            "/force_torque_sensor_broadcaster/wrench",
            str(self.p("chamfer_detection_topic")),
            "-o", str(path),
        ]
        self.bag_process = subprocess.Popen(
            command,
            start_new_session=True,
        )
        time.sleep(float(self.p("bag_start_wait_s")))
        if self.bag_process.poll() is not None:
            raise RuntimeError("ros2 bag record terminated.")
        self.publish_sliding_frame()
        return path

    def stop_bag(self) -> None:
        if self.bag_process is None:
            return
        if self.bag_process.poll() is None:
            try:
                os.killpg(self.bag_process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                self.bag_process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                os.killpg(self.bag_process.pid, signal.SIGTERM)
                self.bag_process.wait(timeout=5.0)
        self.bag_process = None

    def run_validation(self) -> None:
        self.switch_scaled()
        self.send_goal(
            self.scaled_client,
            self.start_joints(),
            float(self.p("move_duration_s")),
        )
        self.validate_start()
        start_pos, _ = self.pose_vector()

        input(
            "\nEnsure the path is contact-free and speed is reduced. "
            "Press Enter to execute the radial endpoint test..."
        )

        slide_duration = (
            float(self.p("slide_distance_m"))
            / float(self.p("radial_speed_mps"))
        )
        self.send_goal(
            self.scaled_client,
            self.end_joints(),
            slide_duration,
        )
        end_pos, _ = self.pose_vector()
        result = self.validate_radial_displacement(start_pos, end_pos)
        self.get_logger().info(f"Free-space validation passed: {result}")

        self.send_goal(
            self.scaled_client,
            self.start_joints(),
            float(self.p("move_duration_s")),
        )

    def run_sliding(self) -> None:
        if not bool(self.p("free_space_validated")):
            raise RuntimeError(
                "free_space_validated is false. Run mode=validate first."
            )

        self.switch_scaled()
        self.send_goal(
            self.scaled_client,
            self.start_joints(),
            float(self.p("move_duration_s")),
        )
        self.validate_start()

        input(
            "\nStart pose verified. Ensure no contact and emergency stop "
            "is reachable. Press Enter to tare and start sliding..."
        )

        self.zero_sensor()
        bag_path = self.start_bag()

        try:
            self.switch_force()
            self.start_force_mode()
            self.wait_contact()
            self.stabilise_contact()

            start_pos, _ = self.pose_vector()
            self._start_chamfer_detector(start_pos)
            slide_duration = (
                float(self.p("slide_distance_m"))
                / float(self.p("radial_speed_mps"))
            )
            self.get_logger().info(
                f"Start radial sliding for {slide_duration:.3f} s."
            )
            self.send_goal(
                self.passthrough_client,
                self.end_joints(),
                slide_duration,
                monitor_force=True,
            )
            end_pos, _ = self.pose_vector()
            self.validate_radial_displacement(start_pos, end_pos)

            self.stop_force_mode()
            self.switch_scaled()
            self.send_goal(
                self.scaled_client,
                self.start_joints(),
                float(self.p("move_duration_s")),
            )
            self.validate_start()
            result_path = self.write_chamfer_detection_result(bag_path)
            self.get_logger().info(
                f"Sliding test completed: {bag_path}"
            )
            self.get_logger().info(
                f"Fasenerkennungs-Ergebnis: {result_path}"
            )

        finally:
            try:
                self.stop_force_mode()
            except Exception:
                pass
            try:
                self.switch_scaled()
            except Exception:
                pass
            try:
                if self.slide_started_monotonic is not None:
                    self.write_chamfer_detection_result(bag_path)
            except Exception as exc:
                self.get_logger().error(
                    f"Fasenerkennungs-Ergebnis konnte nicht gespeichert "
                    f"werden: {exc}"
                )
            self.stop_bag()

    def run(self) -> None:
        self.wait_interfaces()
        mode = str(self.p("mode")).strip().lower()
        if mode == "validate":
            self.run_validation()
        elif mode == "slide":
            self.run_sliding()
        else:
            raise ValueError("mode must be 'validate' or 'slide'.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RadialSlidingChamferObserver()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warning("Interrupted by user.")
    finally:
        try:
            node.stop_force_mode()
        except Exception:
            pass
        try:
            node.switch_scaled()
        except Exception:
            pass
        try:
            node.stop_bag()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

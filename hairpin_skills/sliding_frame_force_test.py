#!/usr/bin/env python3
"""
Isolierter Force-Mode-Versuch mit sliding_frame als Prozess-KOS.

Ablauf
------
1. Mit dem scaled_joint_trajectory_controller zu einer kalibrierten
   Start-Gelenkpose fahren.
2. Über TF prüfen, ob der TCP relativ zu sliding_frame tatsächlich an der
   gewünschten Startpose liegt:
       Position:    [0, 0, start_height_m]
       Orientierung: start_q_* relativ zu sliding_frame
3. Kraftsensor ohne Kontakt nullen.
4. ROS-Bag starten.
5. Auf passthrough_trajectory_controller + force_mode_controller wechseln.
6. Force Mode mit Fz < 0 im sliding_frame starten.
7. Kontakt herstellen lassen und hold_time_s halten.
8. Force Mode stoppen.
9. Auf scaled_joint_trajectory_controller zurückschalten.
10. Zur identischen Start-Gelenkpose zurückfahren.
11. Bag stoppen.

Wichtiger methodischer Punkt
----------------------------
Das sliding_frame definiert eine kartesische Sollpose. Der vorhandene
FollowJointTrajectory-Controller akzeptiert jedoch nur Gelenkwerte. Deshalb
muss start_joints einmal so kalibriert werden, dass die resultierende TCP-Pose
der Sollpose im sliding_frame entspricht. Das Skript überprüft diese Bedingung
vor jedem Versuch und verweigert den Kraftversuch bei zu großer Abweichung.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.action import ActionClient
from geometry_msgs.msg import TransformStamped, WrenchStamped
from rclpy.node import Node
from rclpy.time import Time
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
        raise ValueError("duration must be > 0")
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def quaternion_angle_error(
    q_actual: Sequence[float],
    q_target: Sequence[float],
) -> float:
    """Kleinster Orientierungsfehler zweier Einheitsquaternionen in rad."""
    dot = abs(sum(a * b for a, b in zip(q_actual, q_target)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


class SlidingFrameForceTest(Node):
    def __init__(self) -> None:
        super().__init__("sliding_frame_force_test")

        # Frames and experiment parameters
        self.declare_parameter("sliding_frame", "sliding_frame")
        self.declare_parameter("tcp_frame", "tool0_controller")

        # Static transform base -> sliding_frame. These values correspond to
        # the calibrated PolyScope plane used for the sliding experiment.
        self.declare_parameter("publish_sliding_frame", True)
        self.declare_parameter("sliding_parent_frame", "base")
        self.declare_parameter("sliding_tf_x_m", -0.22737)
        self.declare_parameter("sliding_tf_y_m", -0.40030)
        self.declare_parameter("sliding_tf_z_m", 0.03517)
        self.declare_parameter("sliding_tf_qx", -0.029727)
        self.declare_parameter("sliding_tf_qy", 0.047563)
        self.declare_parameter("sliding_tf_qz", -0.975264)
        self.declare_parameter("sliding_tf_qw", -0.213808)

        # Desired TCP pose expressed in sliding_frame.
        self.declare_parameter("start_x_m", 0.0)
        self.declare_parameter("start_y_m", 0.0)
        self.declare_parameter("start_z_m", 0.002)
        self.declare_parameter("start_qx", 0.0)
        self.declare_parameter("start_qy", 0.0)
        self.declare_parameter("start_qz", 0.0)
        self.declare_parameter("start_qw", 1.0)

        # One-time calibrated joint pose corresponding to the pose above.
        self.declare_parameter("start_joints", [0.0] * 6)
        self.declare_parameter("start_joints_calibrated", False)
        self.declare_parameter("move_duration_s", 4.0)

        # Validation tolerances
        self.declare_parameter("position_tolerance_m", 0.0010)
        self.declare_parameter(
            "orientation_tolerance_rad",
            math.radians(3.0),
        )

        # Force-mode parameters
        self.declare_parameter("normal_force_n", -1.0)
        self.declare_parameter("hold_time_s", 2.0)
        self.declare_parameter("speed_limit_z_mps", 0.005)
        self.declare_parameter("deviation_xy_m", 0.002)
        self.declare_parameter("deviation_z_m", 0.005)
        self.declare_parameter("deviation_angle_rad", 0.05)
        self.declare_parameter("damping_factor", 0.5)
        self.declare_parameter("gain_scaling", 0.3)

        # Contact detection and force safety
        self.declare_parameter("wrench_topic", "/force_torque_sensor_broadcaster/wrench")
        self.declare_parameter("contact_threshold_n", 2.0)
        self.declare_parameter("contact_persistence_s", 0.05)
        self.declare_parameter("contact_timeout_s", 3.0)
        self.declare_parameter("max_abs_normal_force_n", 12.0)
        self.declare_parameter("wrench_timeout_s", 0.25)

        # Bag parameters
        self.declare_parameter(
            "bag_root",
            str(Path.home() / "masterarbeit" / "bags"),
        )
        self.declare_parameter("bag_name", "force_mode_sliding_frame")
        self.declare_parameter("bag_start_wait_s", 2.0)
        self.declare_parameter("tare_wait_s", 1.0)

        self.scaled_action = (
            "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )
        self.scaled_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.scaled_action,
        )

        self.switch_client = self.create_client(
            SwitchController,
            "/controller_manager/switch_controller",
        )
        self.list_controllers_client = self.create_client(
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

        if bool(self.p("publish_sliding_frame")):
            self.publish_sliding_frame()

        self.current_normal_force_n: Optional[float] = None
        self.last_wrench_monotonic: Optional[float] = None
        self.create_subscription(
            WrenchStamped,
            str(self.p("wrench_topic")),
            self._wrench_callback,
            50,
        )

        self.bag_process: Optional[subprocess.Popen] = None
        self.force_mode_active = False

    def p(self, name: str):
        return self.get_parameter(name).value

    def publish_sliding_frame(self) -> None:
        """Publish the calibrated static transform base -> sliding_frame."""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = str(self.p("sliding_parent_frame"))
        transform.child_frame_id = str(self.p("sliding_frame"))

        transform.transform.translation.x = float(self.p("sliding_tf_x_m"))
        transform.transform.translation.y = float(self.p("sliding_tf_y_m"))
        transform.transform.translation.z = float(self.p("sliding_tf_z_m"))

        transform.transform.rotation.x = float(self.p("sliding_tf_qx"))
        transform.transform.rotation.y = float(self.p("sliding_tf_qy"))
        transform.transform.rotation.z = float(self.p("sliding_tf_qz"))
        transform.transform.rotation.w = float(self.p("sliding_tf_qw"))

        norm = math.sqrt(
            transform.transform.rotation.x ** 2
            + transform.transform.rotation.y ** 2
            + transform.transform.rotation.z ** 2
            + transform.transform.rotation.w ** 2
        )
        if norm < 1e-9:
            raise ValueError("Quaternion des sliding_frame ist ungültig.")

        transform.transform.rotation.x /= norm
        transform.transform.rotation.y /= norm
        transform.transform.rotation.z /= norm
        transform.transform.rotation.w /= norm

        self.static_tf_broadcaster.sendTransform(transform)
        self.get_logger().info(
            "Statischer TF veröffentlicht: "
            f"{transform.header.frame_id} -> {transform.child_frame_id}"
        )

        # Allow the local TF listener to receive the transient-local message.
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.05)

    @staticmethod
    def _rotate_vector_by_quaternion(
        vector: Sequence[float],
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> tuple[float, float, float]:
        """Rotate a vector using q = target <- source."""
        vx, vy, vz = vector
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return (
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        )

    def _wrench_callback(self, msg: WrenchStamped) -> None:
        """Transform the measured force orientation into sliding_frame."""
        source_frame = msg.header.frame_id or "tool0"
        target_frame = str(self.p("sliding_frame"))

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"Wrench kann nicht nach {target_frame} transformiert werden: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        q = transform.transform.rotation
        force_target = self._rotate_vector_by_quaternion(
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

        self.current_normal_force_n = float(force_target[2])
        self.last_wrench_monotonic = time.monotonic()

    def wait_for_fresh_wrench(self, timeout_s: float = 3.0) -> None:
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

            if (
                self.current_normal_force_n is not None
                and self.last_wrench_monotonic is not None
                and time.monotonic() - self.last_wrench_monotonic
                <= float(self.p("wrench_timeout_s"))
            ):
                return

        raise RuntimeError(
            "Keine aktuelle, nach sliding_frame transformierbare "
            "Kraftmessung verfügbar."
        )

    def wait_for_contact(self) -> float:
        """Wait for persistent normal contact and return detected Fz_R."""
        threshold = float(self.p("contact_threshold_n"))
        persistence = float(self.p("contact_persistence_s"))
        timeout = float(self.p("contact_timeout_s"))
        max_force = float(self.p("max_abs_normal_force_n"))

        if threshold <= 0.0:
            raise ValueError("contact_threshold_n muss > 0 sein.")
        if persistence <= 0.0:
            raise ValueError("contact_persistence_s muss > 0 sein.")
        if timeout <= 0.0:
            raise ValueError("contact_timeout_s muss > 0 sein.")

        self.get_logger().info(
            "Warte auf Kontakt: "
            f"|Fz_R| >= {threshold:.3f} N für "
            f"{persistence * 1000.0:.0f} ms"
        )

        deadline = time.monotonic() + timeout
        above_since: Optional[float] = None

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.005)

            if (
                self.current_normal_force_n is None
                or self.last_wrench_monotonic is None
            ):
                continue

            if (
                time.monotonic() - self.last_wrench_monotonic
                > float(self.p("wrench_timeout_s"))
            ):
                above_since = None
                continue

            force_z = self.current_normal_force_n
            abs_force = abs(force_z)

            if abs_force > max_force:
                raise RuntimeError(
                    "Sicherheitsabbruch: "
                    f"|Fz_R|={abs_force:.3f} N überschreitet "
                    f"{max_force:.3f} N."
                )

            if abs_force >= threshold:
                if above_since is None:
                    above_since = time.monotonic()
                elif time.monotonic() - above_since >= persistence:
                    self.get_logger().info(
                        f"Kontakt erkannt: Fz_R={force_z:.3f} N"
                    )
                    return force_z
            else:
                above_since = None

        raise RuntimeError(
            f"Kein Kontakt innerhalb von {timeout:.3f} s erkannt."
        )

    def hold_contact(self) -> None:
        """Hold after detected contact while monitoring force safety."""
        hold_time = float(self.p("hold_time_s"))
        max_force = float(self.p("max_abs_normal_force_n"))
        deadline = time.monotonic() + hold_time

        self.get_logger().info(
            f"Kontakt erkannt; halte jetzt {hold_time:.3f} s."
        )

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.005)

            if (
                self.last_wrench_monotonic is None
                or time.monotonic() - self.last_wrench_monotonic
                > float(self.p("wrench_timeout_s"))
            ):
                raise RuntimeError(
                    "Kraftmessung während der Haltephase veraltet."
                )

            if self.current_normal_force_n is not None:
                abs_force = abs(self.current_normal_force_n)
                if abs_force > max_force:
                    raise RuntimeError(
                        "Sicherheitsabbruch während Haltephase: "
                        f"|Fz_R|={abs_force:.3f} N überschreitet "
                        f"{max_force:.3f} N."
                    )

    def wait_for_interfaces(self) -> None:
        self.get_logger().info("Warte auf ROS-Schnittstellen ...")

        if not self.scaled_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(
                f"Action nicht verfügbar: {self.scaled_action}"
            )

        clients = (
            (self.switch_client, "switch_controller"),
            (self.list_controllers_client, "list_controllers"),
            (self.zero_client, "zero_ftsensor"),
            (self.force_start_client, "start_force_mode"),
            (self.force_stop_client, "stop_force_mode"),
        )
        for client, name in clients:
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f"Service nicht verfügbar: {name}")

    def controller_states(self) -> dict[str, str]:
        """Return the current lifecycle state of every loaded controller."""
        future = self.list_controllers_client.call_async(
            ListControllers.Request()
        )
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if result is None:
            raise RuntimeError(
                "Keine Antwort von /controller_manager/list_controllers."
            )

        states = {
            controller.name: controller.state
            for controller in result.controller
        }

        self.get_logger().info(
            "Controllerzustände: "
            + ", ".join(
                f"{name}={state}"
                for name, state in sorted(states.items())
            )
        )
        return states

    def switch_controllers(
        self,
        activate: Sequence[str],
        deactivate: Sequence[str],
    ) -> None:
        """
        Switch only controllers whose current state actually requires a change.

        This avoids STRICT failures caused by trying to deactivate an already
        inactive controller or activate an already active controller.
        """
        states = self.controller_states()

        unknown = [
            name
            for name in list(activate) + list(deactivate)
            if name not in states
        ]
        if unknown:
            raise RuntimeError(
                "Folgende Controller sind nicht geladen: "
                + ", ".join(sorted(set(unknown)))
            )

        activate_required = [
            name
            for name in activate
            if states.get(name) != "active"
        ]
        deactivate_required = [
            name
            for name in deactivate
            if states.get(name) == "active"
        ]

        if not activate_required and not deactivate_required:
            self.get_logger().info(
                "Controller befinden sich bereits in der gewünschten "
                "Konfiguration."
            )
            return

        request = SwitchController.Request()
        request.activate_controllers = activate_required
        request.deactivate_controllers = deactivate_required
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True

        self.get_logger().info(
            "Controllerwechsel: "
            f"activate={activate_required}, "
            f"deactivate={deactivate_required}"
        )

        future = self.switch_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if result is None or not result.ok:
            states_after_failure = self.controller_states()
            raise RuntimeError(
                "Controllerwechsel fehlgeschlagen. "
                f"Angefordert: activate={activate_required}, "
                f"deactivate={deactivate_required}. "
                f"Aktuelle Zustände: {states_after_failure}"
            )

        time.sleep(0.5)

        states_after = self.controller_states()

        not_active = [
            name
            for name in activate
            if states_after.get(name) != "active"
        ]
        still_active = [
            name
            for name in deactivate
            if states_after.get(name) == "active"
        ]

        if not_active or still_active:
            raise RuntimeError(
                "Controllerzustände nach dem Wechsel sind unerwartet: "
                f"nicht aktiv={not_active}, "
                f"noch aktiv={still_active}"
            )

    def switch_to_scaled(self) -> None:
        self.switch_controllers(
            activate=["scaled_joint_trajectory_controller"],
            deactivate=[
                "force_mode_controller",
                "passthrough_trajectory_controller",
            ],
        )

    def switch_to_force_configuration(self) -> None:
        self.switch_controllers(
            activate=[
                "passthrough_trajectory_controller",
                "force_mode_controller",
            ],
            deactivate=["scaled_joint_trajectory_controller"],
        )

    def move_to_start_joints(self) -> None:
        if not bool(self.p("start_joints_calibrated")):
            raise RuntimeError(
                "start_joints_calibrated ist false. "
                "Die Start-Gelenkpose muss zuerst für die Sollpose im "
                "sliding_frame kalibriert werden."
            )

        joints = list(self.p("start_joints"))
        if len(joints) != 6:
            raise ValueError("start_joints muss genau 6 Werte enthalten.")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in joints]
        point.time_from_start = duration_msg(
            float(self.p("move_duration_s"))
        )
        goal.trajectory.points = [point]

        self.get_logger().info(
            "Fahre kalibrierte Start-Gelenkpose an ..."
        )

        future = self.scaled_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()

        if handle is None or not handle.accepted:
            raise RuntimeError("Starttrajektorie wurde abgelehnt.")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped_result = result_future.result()

        if wrapped_result is None:
            raise RuntimeError("Keine Rückmeldung der Starttrajektorie.")

    def lookup_start_pose(self):
        target = str(self.p("sliding_frame"))
        source = str(self.p("tcp_frame"))

        deadline = time.monotonic() + 5.0
        last_error = None

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                return self.tf_buffer.lookup_transform(
                    target,
                    source,
                    Time(),
                )
            except TransformException as exc:
                last_error = exc

        raise RuntimeError(
            f"Keine TF-Verbindung {target} <- {source}: {last_error}"
        )

    def validate_start_pose(self) -> None:
        transform = self.lookup_start_pose()
        t = transform.transform.translation
        q = transform.transform.rotation

        target_position = (
            float(self.p("start_x_m")),
            float(self.p("start_y_m")),
            float(self.p("start_z_m")),
        )
        actual_position = (t.x, t.y, t.z)

        position_error = math.sqrt(
            sum(
                (actual - target) ** 2
                for actual, target in zip(
                    actual_position,
                    target_position,
                )
            )
        )

        target_quaternion = (
            float(self.p("start_qx")),
            float(self.p("start_qy")),
            float(self.p("start_qz")),
            float(self.p("start_qw")),
        )
        actual_quaternion = (q.x, q.y, q.z, q.w)

        orientation_error = quaternion_angle_error(
            actual_quaternion,
            target_quaternion,
        )

        self.get_logger().info(
            "TCP relativ zu sliding_frame: "
            f"p=[{t.x:.6f}, {t.y:.6f}, {t.z:.6f}] m, "
            f"q=[{q.x:.6f}, {q.y:.6f}, {q.z:.6f}, {q.w:.6f}]"
        )
        self.get_logger().info(
            f"Startfehler: Position={position_error * 1000.0:.3f} mm, "
            f"Orientierung={math.degrees(orientation_error):.3f} deg"
        )

        if position_error > float(self.p("position_tolerance_m")):
            raise RuntimeError(
                "Startposition liegt außerhalb der Toleranz."
            )

        if orientation_error > float(
            self.p("orientation_tolerance_rad")
        ):
            raise RuntimeError(
                "Startorientierung liegt außerhalb der Toleranz."
            )

    def zero_sensor(self) -> None:
        self.get_logger().info(
            "Nulle Kraftsensor ohne Oberflächenkontakt ..."
        )
        future = self.zero_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if result is None or not result.success:
            message = "" if result is None else result.message
            raise RuntimeError(
                f"Nullstellen fehlgeschlagen: {message}"
            )

        time.sleep(float(self.p("tare_wait_s")))

    def bag_path(self) -> Path:
        root = Path(str(self.p("bag_root"))).expanduser()
        root.mkdir(parents=True, exist_ok=True)

        name = str(self.p("bag_name")).strip()
        if not name:
            raise ValueError("bag_name darf nicht leer sein.")

        path = root / name
        if path.exists():
            raise FileExistsError(f"Bag existiert bereits: {path}")
        return path

    def start_bag(self, path: Path) -> None:
        command = [
            "ros2",
            "bag",
            "record",
            "-s",
            "mcap",
            "/tf",
            "/tf_static",
            "/joint_states",
            "/tcp_pose_broadcaster/pose",
            "/force_torque_sensor_broadcaster/wrench",
            "-o",
            str(path),
        ]

        self.get_logger().info(f"Starte Bag: {path}")
        self.bag_process = subprocess.Popen(
            command,
            start_new_session=True,
        )

        time.sleep(float(self.p("bag_start_wait_s")))

        if self.bag_process.poll() is not None:
            raise RuntimeError(
                "ros2 bag record wurde unerwartet beendet."
            )

    def stop_bag(self) -> None:
        if self.bag_process is None:
            return

        if self.bag_process.poll() is None:
            self.get_logger().info("Stoppe Bag-Aufnahme ...")
            try:
                os.killpg(self.bag_process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

            try:
                self.bag_process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(
                        self.bag_process.pid,
                        signal.SIGTERM,
                    )
                except ProcessLookupError:
                    pass
                self.bag_process.wait(timeout=5.0)

        self.bag_process = None

    def start_force_mode(self) -> None:
        request = SetForceMode.Request()

        # Die Force-Mode-Achsen sind direkt die Achsen des sliding_frame.
        request.task_frame.header.frame_id = str(
            self.p("sliding_frame")
        )
        request.task_frame.pose.orientation.w = 1.0

        request.selection_vector_x = False
        request.selection_vector_y = False
        request.selection_vector_z = True
        request.selection_vector_rx = False
        request.selection_vector_ry = False
        request.selection_vector_rz = False

        request.wrench.force.z = float(
            self.p("normal_force_n")
        )
        request.type = SetForceMode.Request.NO_TRANSFORM

        request.speed_limits.linear.z = float(
            self.p("speed_limit_z_mps")
        )

        request.deviation_limits = [
            float(self.p("deviation_xy_m")),
            float(self.p("deviation_xy_m")),
            float(self.p("deviation_z_m")),
            float(self.p("deviation_angle_rad")),
            float(self.p("deviation_angle_rad")),
            float(self.p("deviation_angle_rad")),
        ]

        request.damping_factor = float(
            self.p("damping_factor")
        )
        request.gain_scaling = float(
            self.p("gain_scaling")
        )

        self.get_logger().info(
            "Starte Force Mode: "
            f"Fz={request.wrench.force.z:.3f} N im "
            f"{request.task_frame.header.frame_id}"
        )

        future = self.force_start_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if result is None or not result.success:
            raise RuntimeError("Force Mode konnte nicht gestartet werden.")

        self.force_mode_active = True

    def stop_force_mode(self) -> None:
        if not self.force_mode_active:
            return

        future = self.force_stop_client.call_async(
            Trigger.Request()
        )
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if result is None or not result.success:
            raise RuntimeError("Force Mode konnte nicht gestoppt werden.")

        self.force_mode_active = False
        self.get_logger().info("Force Mode gestoppt.")

    def run_experiment(self) -> None:
        self.wait_for_interfaces()
        path = self.bag_path()

        # All robot motion remains in Remote mode.
        self.switch_to_scaled()
        self.move_to_start_joints()
        self.validate_start_pose()

        input(
            "\nStartpose ist geprüft. Sicherstellen, dass kein Kontakt "
            "zur Oberfläche besteht und Not-Halt erreichbar ist.\n"
            "Enter startet Tare und Versuch ..."
        )

        self.zero_sensor()
        self.start_bag(path)

        # Publish the static process frame again after rosbag2 has subscribed.
        # This guarantees that base -> sliding_frame is contained in /tf_static
        # even when the first publication happened before the recorder started.
        if bool(self.p("publish_sliding_frame")):
            self.publish_sliding_frame()
            time.sleep(0.25)

        try:
            self.switch_to_force_configuration()
            self.wait_for_fresh_wrench()
            self.start_force_mode()

            self.wait_for_contact()
            self.hold_contact()

            self.stop_force_mode()
            self.switch_to_scaled()

            # Rückzug zur identischen, geprüften Startpose.
            self.move_to_start_joints()
            self.validate_start_pose()

        finally:
            try:
                self.stop_force_mode()
            except Exception as exc:
                self.get_logger().error(
                    f"Fehler beim Stoppen des Force Mode: {exc}"
                )

            try:
                self.switch_to_scaled()
            except Exception as exc:
                self.get_logger().error(
                    f"Controller konnten nicht zurückgesetzt werden: {exc}"
                )

            self.stop_bag()

        self.get_logger().info(
            f"Versuch abgeschlossen. Bag: {path}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlidingFrameForceTest()

    try:
        node.run_experiment()
    except KeyboardInterrupt:
        node.get_logger().warning(
            "Versuch durch Benutzer abgebrochen."
        )
    except Exception as exc:
        node.get_logger().error(str(exc))
        raise
    finally:
        try:
            node.stop_force_mode()
        except Exception:
            pass
        try:
            node.switch_to_scaled()
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

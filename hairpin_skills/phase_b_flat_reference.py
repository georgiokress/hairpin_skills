#!/usr/bin/env python3
"""
Phase B: Kraftgeregelte Referenzfahrt auf einer ebenen Oberfläche.

Der Node kombiniert:
- die in Phase A validierte radiale IK-Mehrpunkttrajektorie,
- die im 3-mm-Versuch bewährte Kraftregelung in z_R,
- eine direkte CSV-Aufzeichnung von TCP und Kräften im sliding_frame.

Ablauf
------
1. Scaled Controller aktivieren und bekannte Startpose anfahren.
2. Startpose validieren.
3. ATI-Bias und ATI-Messwert im Stillstand validieren; Rosbag starten.
4. Passthrough- und Force-Mode-Controller aktivieren.
5. Force Mode ausschließlich in z_R starten.
6. Kontakt erkennen und stabilisieren.
7. Baseline auf der ebenen Oberfläche aufnehmen.
8. Alle validierten IK-Wegpunkte als EIN Trajektorien-Goal ausführen.
9. Force Mode stoppen.
10. Zum nominalen IK-Endpunkt anheben (gleiche x_R-Position, nominale z-Höhe).
11. Entlang der IK-Wegpunkte zur Startpose zurückfahren.
12. CSV und JSON-Kennwerte speichern.

Wichtig
-------
- Dieser Node enthält KEINE Fasenerkennung.
- Er ist ausschließlich für eine durchgehend ebene Referenzoberfläche gedacht.
- Bei einem Fehler erfolgt keine automatische Recovery-Bewegung.
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence

import rclpy
import yaml
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import TransformStamped, WrenchStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformException,
    TransformListener,
)
from trajectory_msgs.msg import JointTrajectoryPoint
from ur_msgs.srv import SetForceMode


EXPECTED_JOINT_NAMES = [
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
    dot = abs(sum(float(a) * float(b) for a, b in zip(q1, q2)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def linear_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Ordinary least squares y = slope*x + intercept."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("At least two paired values are required.")
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 1e-18:
        return 0.0, y_mean
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    ) / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


class PhaseBFlatReference(Node):
    def __init__(self, node_name: str = "phase_b_flat_reference") -> None:
        super().__init__(node_name)

        # Freigaben
        self.declare_parameter("phase_a_validated", False)
        self.declare_parameter("flat_surface_confirmed", False)

        # Frames
        self.declare_parameter("sliding_frame", "sliding_frame")
        self.declare_parameter("tcp_frame", "tool0_controller")
        self.declare_parameter("tf_timeout_s", 10.0)
        self.declare_parameter("publish_sliding_frame", True)
        self.declare_parameter("sliding_parent_frame", "base")
        self.declare_parameter("sliding_tf_x_m", -0.228)
        self.declare_parameter("sliding_tf_y_m", -0.400)
        self.declare_parameter("sliding_tf_z_m", 0.038)
        self.declare_parameter("sliding_tf_qx", -0.002)
        self.declare_parameter("sliding_tf_qy", 0.0)
        self.declare_parameter("sliding_tf_qz", 0.971)
        self.declare_parameter("sliding_tf_qw", 0.240)

        # Kalibrierte Startpose
        self.declare_parameter("start_x_m", 0.0)
        self.declare_parameter("start_y_m", 0.0)
        self.declare_parameter("start_z_m", 0.002)
        self.declare_parameter("start_qx", 0.056396)
        self.declare_parameter("start_qy", 0.030219)
        self.declare_parameter("start_qz", 0.280641)
        self.declare_parameter("start_qw", 0.957678)
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
        self.declare_parameter("move_to_start_duration_s", 4.0)
        self.declare_parameter("start_position_tolerance_m", 0.001)
        self.declare_parameter(
            "start_orientation_tolerance_rad", math.radians(3.0)
        )

        # Validierte IK-Trajektorie
        self.declare_parameter(
            "waypoint_yaml_file",
            str(Path.home() / "masterarbeit" / "radial_ik_waypoints.yaml"),
        )
        self.declare_parameter("radial_speed_mps", 0.001)
        self.declare_parameter("max_waypoint_index", 2)
        self.declare_parameter("endpoint_radial_tolerance_m", 0.0005)
        self.declare_parameter("cross_axis_tolerance_m", 0.0005)
        self.declare_parameter(
            "orientation_tolerance_rad", math.radians(3.0)
        )
        self.declare_parameter("monotonic_noise_tolerance_m", 0.00010)

        # Kraftregelung – aus dem funktionierenden 3-mm-Skill übernommen
        self.declare_parameter("normal_force_n", -5.0)
        self.declare_parameter("contact_threshold_n", 2.0)
        self.declare_parameter("contact_persistence_s", 0.05)
        self.declare_parameter("contact_timeout_s", 3.0)
        self.declare_parameter("stabilisation_after_contact_s", 0.5)
        self.declare_parameter("baseline_duration_s", 1.0)
        self.declare_parameter("speed_limit_z_mps", 0.005)
        self.declare_parameter("max_abs_normal_force_n", 12.0)
        self.declare_parameter("min_abs_contact_force_n", 1.0)
        self.declare_parameter("contact_loss_persistence_s", 0.20)
        self.declare_parameter("wrench_timeout_s", 0.25)
        self.declare_parameter("deviation_xy_m", 0.002)
        self.declare_parameter("deviation_z_m", 0.005)
        self.declare_parameter("deviation_angle_rad", 0.05)
        self.declare_parameter("damping_factor", 0.5)
        self.declare_parameter("gain_scaling", 0.3)
        self.declare_parameter(
            "wrench_topic", "/force_torque_sensor_broadcaster/wrench"
        )
        self.declare_parameter(
            "measurement_wrench_topic", "/ati/wrench_tcp"
        )
        self.declare_parameter("measurement_wrench_required", True)
        self.declare_parameter("measurement_wrench_timeout_s", 0.25)
        self.declare_parameter(
            "measurement_wrench_expected_frame", "ati_ur_aligned"
        )
        self.declare_parameter(
            "ati_bias_ready_topic", "/ati_ft_rtde_bridge/bias_ready"
        )
        self.declare_parameter("ati_bias_ready_timeout_s", 10.0)
        self.declare_parameter(
            "robot_program_running_topic",
            "/io_and_status_controller/robot_program_running",
        )
        self.declare_parameter("require_robot_program_running", True)
        self.declare_parameter("robot_dashboard_ip", "192.168.214.203")
        self.declare_parameter("robot_dashboard_port", 29999)
        self.declare_parameter("robot_dashboard_timeout_s", 0.25)
        self.declare_parameter("ati_bias_check_duration_s", 0.5)
        self.declare_parameter("ati_bias_max_abs_mean_force_n", 0.25)

        # Online-Sicherheitsgrenzen während der ebenen Referenzfahrt
        self.declare_parameter("abort_abs_dy_from_slide_start_m", 0.0015)
        self.declare_parameter("abort_abs_dz_from_baseline_m", 0.0015)
        self.declare_parameter(
            "abort_orientation_rad", math.radians(6.0)
        )
        self.declare_parameter("abort_radial_undershoot_m", 0.0005)
        self.declare_parameter("abort_radial_overshoot_m", 0.0010)

        # Vorläufige Referenzkriterien; werden vollständig im JSON dokumentiert
        self.declare_parameter("flat_residual_peak_to_peak_limit_m", 0.0005)
        self.declare_parameter("mean_force_error_limit_n", 1.0)

        # Abheben und Rückfahrt
        self.declare_parameter("lift_to_nominal_endpoint_duration_s", 2.0)
        self.declare_parameter("return_speed_mps", 0.001)
        self.declare_parameter("pause_after_force_stop_s", 0.25)
        self.declare_parameter("pause_after_lift_s", 0.25)

        # Abtastung und Ausgabe
        self.declare_parameter("sample_period_s", 0.02)
        self.declare_parameter(
            "result_root",
            str(Path.home() / "masterarbeit" / "results" / "phase_b"),
        )
        self.declare_parameter("run_name", "phase_b_flat_pilot_01")
        self.declare_parameter(
            "bag_root", str(Path.home() / "masterarbeit" / "bags")
        )
        self.declare_parameter("record_bag", True)
        self.declare_parameter("bag_start_wait_s", 2.0)
        self.declare_parameter(
            "bag_qos_overrides_path",
            str(
                Path.home()
                / "hairpin_ws"
                / "src"
                / "hairpin_skills"
                / "config"
                / "rosbag_qos_overrides.yaml"
            ),
        )
        self.declare_parameter(
            "event_topic", "/phase_b_flat_reference/events"
        )

        # ROS-Schnittstellen
        self.ati_bias_ready: Optional[bool] = None
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
            SwitchController, "/controller_manager/switch_controller"
        )
        self.list_client = self.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        self.force_start_client = self.create_client(
            SetForceMode, "/force_mode_controller/start_force_mode"
        )
        self.force_stop_client = self.create_client(
            Trigger, "/force_mode_controller/stop_force_mode"
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.event_publisher = self.create_publisher(
            String, str(self.p("event_topic")), 10
        )
        self.create_subscription(
            WrenchStamped,
            str(self.p("wrench_topic")),
            self._wrench_cb,
            50,
        )
        measurement_qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            WrenchStamped,
            str(self.p("measurement_wrench_topic")),
            self._measurement_wrench_cb,
            measurement_qos,
        )
        bias_state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Bool,
            str(self.p("ati_bias_ready_topic")),
            self._ati_bias_ready_cb,
            bias_state_qos,
        )
        self.create_subscription(
            Bool,
            str(self.p("robot_program_running_topic")),
            self._robot_program_running_cb,
            10,
        )

        # Laufzeitdaten
        self.joint_names: list[str] = []
        self.offsets: list[float] = []
        self.waypoints: list[list[float]] = []
        self.active_offsets: list[float] = []
        self.active_waypoints: list[list[float]] = []
        self.current_force_source: Optional[list[float]] = None
        self.current_force_sliding: Optional[list[float]] = None
        self.current_wrench_source_frame: str = ""
        self.last_wrench_monotonic: Optional[float] = None
        self.current_measurement_force_source: Optional[list[float]] = None
        self.current_measurement_force_sliding: Optional[list[float]] = None
        self.current_measurement_wrench_source_frame: str = ""
        self.last_measurement_wrench_monotonic: Optional[float] = None
        self.force_mode_active = False
        self.bag_process: Optional[subprocess.Popen] = None
        self.samples: list[dict] = []
        self.run_t0_monotonic: Optional[float] = None
        self.robot_program_running: Optional[bool] = None

        if bool(self.p("publish_sliding_frame")):
            self.publish_sliding_frame()

    def p(self, name: str):
        return self.get_parameter(name).value

    def publish_sliding_frame(self) -> None:
        """Publish the measured Setznest2 process frame."""
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
        self.static_tf_broadcaster.sendTransform(transform)

    def _robot_program_running_cb(self, msg: Bool) -> None:
        self.robot_program_running = bool(msg.data)

    def _ati_bias_ready_cb(self, msg: Bool) -> None:
        self.ati_bias_ready = bool(msg.data)

    def read_dashboard_program_running(self) -> bool:
        """Read the initial PolyScope program state without changing it."""
        address = (
            str(self.p("robot_dashboard_ip")),
            int(self.p("robot_dashboard_port")),
        )
        timeout = float(self.p("robot_dashboard_timeout_s"))
        if timeout <= 0.0:
            raise ValueError("robot_dashboard_timeout_s must be > 0.")
        with socket.create_connection(address, timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.recv(4096)
            connection.sendall(b"programState\n")
            response = connection.recv(4096).decode(
                "ascii", errors="replace"
            ).strip()
        self.get_logger().info(f"Dashboard program state: {response}")
        return response.startswith("PLAYING ")

    def publish_event(self, name: str, **fields) -> None:
        payload = {
            "event": name,
            "ros_time_s": self.get_clock().now().nanoseconds / 1e9,
            **fields,
        }
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.event_publisher.publish(message)
        self.get_logger().info(f"EVENT {message.data}")

    @staticmethod
    def rotate_vector(
        vector: Sequence[float],
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> tuple[float, float, float]:
        vx, vy, vz = vector
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return (
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        )

    def transform_wrench_force(
        self, msg: WrenchStamped
    ) -> tuple[list[float], list[float], str] | None:
        source = msg.header.frame_id or "tool0"
        try:
            transform = self.tf_buffer.lookup_transform(
                str(self.p("sliding_frame")), source, Time()
            )
        except TransformException:
            return

        force_source = [
            float(msg.wrench.force.x),
            float(msg.wrench.force.y),
            float(msg.wrench.force.z),
        ]
        q = transform.transform.rotation
        force_sliding = self.rotate_vector(
            force_source, q.x, q.y, q.z, q.w
        )
        return (
            force_source,
            [float(value) for value in force_sliding],
            source,
        )

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        transformed = self.transform_wrench_force(msg)
        if transformed is None:
            return
        self.current_force_source = transformed[0]
        self.current_force_sliding = transformed[1]
        self.current_wrench_source_frame = transformed[2]
        self.last_wrench_monotonic = time.monotonic()

    def _measurement_wrench_cb(self, msg: WrenchStamped) -> None:
        transformed = self.transform_wrench_force(msg)
        if transformed is None:
            return
        self.current_measurement_force_source = transformed[0]
        self.current_measurement_force_sliding = transformed[1]
        self.current_measurement_wrench_source_frame = transformed[2]
        self.last_measurement_wrench_monotonic = time.monotonic()

    def wait_interfaces(self) -> None:
        if not self.scaled_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Scaled trajectory action unavailable.")
        if not self.passthrough_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Passthrough trajectory action unavailable.")
        for client, name in (
            (self.switch_client, "switch_controller"),
            (self.list_client, "list_controllers"),
            (self.force_start_client, "start_force_mode"),
            (self.force_stop_client, "stop_force_mode"),
        ):
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f"Service unavailable: {name}")

        if bool(self.p("measurement_wrench_required")):
            bias_timeout = float(self.p("ati_bias_ready_timeout_s"))
            if bias_timeout <= 0.0:
                raise ValueError("ati_bias_ready_timeout_s must be > 0.")
            bias_deadline = time.monotonic() + bias_timeout
            while self.ati_bias_ready is None and time.monotonic() < bias_deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            if self.ati_bias_ready is not True:
                raise RuntimeError(
                    "ATI RTDE bridge software bias is not ready. Keep the "
                    "tool stationary and contact-free, execute "
                    "/ati_ft_rtde_bridge/bias, then restart Phase B. "
                    "No motion was started."
                )

            deadline = time.monotonic() + 10.0
            while (
                self.last_measurement_wrench_monotonic is None
                and time.monotonic() < deadline
            ):
                rclpy.spin_once(self, timeout_sec=0.05)
            if self.last_measurement_wrench_monotonic is None:
                raise RuntimeError(
                    "Required ATI measurement wrench unavailable on "
                    f"{self.p('measurement_wrench_topic')}. Start the "
                    "standalone ATI RDT node and the ATI RTDE bridge, execute "
                    "the bridge software bias, and keep the ATI URCap daemon "
                    "stopped. No motion was started."
                )
            expected_frame = str(self.p("measurement_wrench_expected_frame"))
            if self.current_measurement_wrench_source_frame != expected_frame:
                raise RuntimeError(
                    "ATI wrench has unexpected frame_id: "
                    f"{self.current_measurement_wrench_source_frame!r}; "
                    f"expected {expected_frame!r}. No motion was started."
                )

        if bool(self.p("require_robot_program_running")):
            deadline = time.monotonic() + 2.0
            while (
                self.robot_program_running is None
                and time.monotonic() < deadline
            ):
                rclpy.spin_once(self, timeout_sec=0.05)
            if self.robot_program_running is None:
                try:
                    self.robot_program_running = (
                        self.read_dashboard_program_running()
                    )
                except OSError as exc:
                    raise RuntimeError(
                        "Unable to read the initial PolyScope program state "
                        f"from the Dashboard server: {exc}. No motion was "
                        "started."
                    ) from exc
            if self.robot_program_running is not True:
                raise RuntimeError(
                    "PolyScope ATI/External-Control program is not running. "
                    "External ATI injection and ROS motion are therefore not "
                    "available. No motion was started."
                )

    def controller_states(self) -> dict[str, str]:
        future = self.list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            raise RuntimeError("No controller list received.")
        return {controller.name: controller.state for controller in result.controller}

    def switch(
        self, activate: Sequence[str], deactivate: Sequence[str]
    ) -> None:
        states = self.controller_states()
        activate_needed = [
            name for name in activate if states.get(name) != "active"
        ]
        deactivate_needed = [
            name for name in deactivate if states.get(name) == "active"
        ]
        if not activate_needed and not deactivate_needed:
            return

        request = SwitchController.Request()
        request.activate_controllers = activate_needed
        request.deactivate_controllers = deactivate_needed
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        future = self.switch_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.ok:
            raise RuntimeError(
                "Controller switch failed: "
                f"activate={activate_needed}, deactivate={deactivate_needed}"
            )
        time.sleep(0.5)

    def switch_scaled(self) -> None:
        self.switch(
            ["scaled_joint_trajectory_controller"],
            ["passthrough_trajectory_controller", "force_mode_controller"],
        )

    def switch_force(self) -> None:
        self.switch(
            ["passthrough_trajectory_controller", "force_mode_controller"],
            ["scaled_joint_trajectory_controller"],
        )

    def load_waypoints(self) -> None:
        path = Path(str(self.p("waypoint_yaml_file"))).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Waypoint YAML not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        try:
            params = payload["radial_ik_waypoints"]["ros__parameters"]
            joint_names = [str(value) for value in params["joint_names"]]
            offsets = [
                float(value) for value in params["slide_waypoint_offsets_m"]
            ]
            flat = [
                float(value) for value in params["slide_waypoint_joints_flat"]
            ]
            calibrated = bool(params["slide_waypoints_calibrated"])
            frame = str(params["sliding_frame_used"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid waypoint YAML: {path}") from exc

        if not calibrated:
            raise RuntimeError(
                "slide_waypoints_calibrated is false. Phase A must be "
                "completed and documented before Phase B."
            )
        if joint_names != EXPECTED_JOINT_NAMES:
            raise RuntimeError(
                f"Unexpected joint order: {joint_names}; "
                f"expected {EXPECTED_JOINT_NAMES}."
            )
        if frame != str(self.p("sliding_frame")):
            raise RuntimeError(
                f"Waypoint frame {frame!r} differs from configured "
                f"frame {self.p('sliding_frame')!r}."
            )
        if len(flat) != 6 * len(offsets):
            raise RuntimeError("Waypoint joint list has invalid length.")
        if any(current <= previous for previous, current in zip(offsets, offsets[1:])):
            raise RuntimeError("Waypoint offsets must be strictly increasing.")

        self.joint_names = joint_names
        self.offsets = offsets
        self.waypoints = [flat[index:index + 6] for index in range(0, len(flat), 6)]
        max_index = int(self.p("max_waypoint_index"))
        if max_index < 1 or max_index > len(self.waypoints):
            raise ValueError(
                f"max_waypoint_index must be between 1 and {len(self.waypoints)}."
            )
        self.active_offsets = self.offsets[:max_index]
        self.active_waypoints = self.waypoints[:max_index]
        self.get_logger().info(
            f"Loaded {len(self.waypoints)} validated IK waypoints; "
            f"this run uses points 1..{max_index}, "
            f"scan length={self.active_offsets[-1] * 1000:.3f} mm."
        )

    def lookup_tcp(self):
        deadline = time.monotonic() + float(self.p("tf_timeout_s"))
        last_error = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            try:
                return self.tf_buffer.lookup_transform(
                    str(self.p("sliding_frame")),
                    str(self.p("tcp_frame")),
                    Time(),
                )
            except TransformException as exc:
                last_error = exc
        raise RuntimeError(f"TCP transform unavailable: {last_error}")

    def pose_vector(self) -> tuple[list[float], list[float]]:
        transform = self.lookup_tcp().transform
        return (
            [
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
            ],
            [
                float(transform.rotation.x),
                float(transform.rotation.y),
                float(transform.rotation.z),
                float(transform.rotation.w),
            ],
        )

    def validate_start(self) -> tuple[list[float], list[float]]:
        position, orientation = self.pose_vector()
        target_position = [
            float(self.p("start_x_m")),
            float(self.p("start_y_m")),
            float(self.p("start_z_m")),
        ]
        target_orientation = [
            float(self.p("start_qx")),
            float(self.p("start_qy")),
            float(self.p("start_qz")),
            float(self.p("start_qw")),
        ]
        position_error = math.dist(position, target_position)
        orientation_error = quat_angle(orientation, target_orientation)
        self.get_logger().info(
            "Start validation: "
            f"p_R={[round(value, 6) for value in position]}, "
            f"position error={position_error * 1000:.3f} mm, "
            f"orientation error={math.degrees(orientation_error):.3f} deg."
        )
        if position_error > float(self.p("start_position_tolerance_m")):
            raise RuntimeError("Start position outside tolerance.")
        if orientation_error > float(self.p("start_orientation_tolerance_rad")):
            raise RuntimeError("Start orientation outside tolerance.")
        return position, orientation

    def create_goal(
        self,
        positions: Sequence[Sequence[float]],
        times_s: Sequence[float],
    ) -> FollowJointTrajectory.Goal:
        if len(positions) != len(times_s) or not positions:
            raise ValueError("Invalid trajectory point/time lists.")
        if any(current <= previous for previous, current in zip(times_s, times_s[1:])):
            raise ValueError("Trajectory times must be strictly increasing.")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.joint_names)
        for joints, time_s in zip(positions, times_s):
            if len(joints) != 6:
                raise ValueError("Each trajectory point requires six joints.")
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in joints]
            point.time_from_start = duration_msg(float(time_s))
            goal.trajectory.points.append(point)
        return goal

    def send_goal_without_sampling(
        self,
        client: ActionClient,
        goal: FollowJointTrajectory.Goal,
    ) -> None:
        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Trajectory goal rejected.")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError("Trajectory returned no result.")
        if wrapped.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                "Trajectory controller error: "
                f"code={wrapped.result.error_code}, "
                f"message={wrapped.result.error_string!r}."
            )

    def move_to_start(self) -> None:
        self.joint_names = EXPECTED_JOINT_NAMES
        goal = self.create_goal(
            [[float(value) for value in self.p("start_joints")]],
            [float(self.p("move_to_start_duration_s"))],
        )
        self.send_goal_without_sampling(self.scaled_client, goal)

    def validate_ati_bias(self) -> None:
        """Verify the bridge-bias-corrected ATI wrench while stationary."""
        duration = float(self.p("ati_bias_check_duration_s"))
        limit = float(self.p("ati_bias_max_abs_mean_force_n"))
        if duration <= 0.0 or limit <= 0.0:
            raise ValueError("ATI bias-check duration and limit must be > 0.")

        samples: list[list[float]] = []
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            self.assert_wrench_safe()
            if self.current_measurement_force_sliding is not None:
                samples.append(list(self.current_measurement_force_sliding))
        if not samples:
            raise RuntimeError("No ATI samples available for bias validation.")

        means = [statistics.fmean(row[axis] for row in samples) for axis in range(3)]
        self.get_logger().info(
            "ATI bias validation in sliding_frame: "
            f"mean force={[round(value, 4) for value in means]} N, "
            f"limit={limit:.3f} N per axis."
        )
        if max(abs(value) for value in means) > limit:
            raise RuntimeError(
                "ATI bias validation failed: mean force component exceeds "
                f"{limit:.3f} N. Execute /ati_ft_rtde_bridge/bias while the "
                "robot is stationary and contact-free, then restart Phase B. "
                "No Force Mode was started."
            )
        self.publish_event(
            "ati_bias_validated",
            mean_fx_R_n=means[0],
            mean_fy_R_n=means[1],
            mean_fz_R_n=means[2],
            samples=len(samples),
        )

    def start_force_mode(self) -> None:
        request = SetForceMode.Request()
        request.task_frame.header.frame_id = str(self.p("sliding_frame"))
        request.task_frame.pose.orientation.w = 1.0
        request.selection_vector_z = True
        request.wrench.force.z = float(self.p("normal_force_n"))
        request.type = SetForceMode.Request.NO_TRANSFORM
        request.speed_limits.linear.z = float(self.p("speed_limit_z_mps"))
        request.deviation_limits = [
            float(self.p("deviation_xy_m")),
            float(self.p("deviation_xy_m")),
            float(self.p("deviation_z_m")),
            float(self.p("deviation_angle_rad")),
            float(self.p("deviation_angle_rad")),
            float(self.p("deviation_angle_rad")),
        ]
        request.damping_factor = float(self.p("damping_factor"))
        request.gain_scaling = float(self.p("gain_scaling"))

        future = self.force_start_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError("Force mode start failed.")
        self.force_mode_active = True
        self.publish_event("force_mode_started", normal_force_n=float(self.p("normal_force_n")))

    def stop_force_mode(self) -> None:
        if not self.force_mode_active:
            return
        future = self.force_stop_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        self.force_mode_active = False
        if result is None or not result.success:
            raise RuntimeError("Force mode stop failed.")
        self.publish_event("force_mode_stopped")

    def assert_wrench_safe(self) -> None:
        if (
            bool(self.p("require_robot_program_running"))
            and self.robot_program_running is not True
        ):
            raise RuntimeError(
                "PolyScope program is no longer running; ATI injection is "
                "not guaranteed. Abort without automatic recovery motion."
            )
        if self.ati_bias_ready is not True:
            raise RuntimeError(
                "ATI RTDE bridge software bias is no longer ready. Abort "
                "without automatic recovery motion."
            )
        if self.last_measurement_wrench_monotonic is None:
            raise RuntimeError("No ATI measurement wrench received.")
        measurement_age = time.monotonic() - self.last_measurement_wrench_monotonic
        if measurement_age > float(self.p("measurement_wrench_timeout_s")):
            raise RuntimeError(
                f"ATI measurement wrench stale: age={measurement_age:.3f} s."
            )
        if self.current_measurement_force_sliding is None:
            raise RuntimeError("No transformed ATI force available.")
        normal_force = self.current_measurement_force_sliding[2]
        if abs(normal_force) > float(self.p("max_abs_normal_force_n")):
            raise RuntimeError(
                "ATI normal-force safety limit exceeded: "
                f"Fz_R={normal_force:.3f} N."
            )

    def wait_contact(self) -> None:
        threshold = float(self.p("contact_threshold_n"))
        persistence = float(self.p("contact_persistence_s"))
        deadline = time.monotonic() + float(self.p("contact_timeout_s"))
        above_since: Optional[float] = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.005)
            self.assert_wrench_safe()
            value = abs(float(self.current_measurement_force_sliding[2]))
            if value >= threshold:
                if above_since is None:
                    above_since = time.monotonic()
                elif time.monotonic() - above_since >= persistence:
                    self.publish_event(
                        "contact_detected",
                        ati_force_z_sliding_n=(
                            self.current_measurement_force_sliding[2]
                        ),
                    )
                    return
            else:
                above_since = None
        raise RuntimeError("Contact timeout.")

    def hold_and_sample(
        self,
        duration_s: float,
        phase: str,
        reference_position: Sequence[float],
        reference_orientation: Sequence[float],
    ) -> None:
        deadline = time.monotonic() + duration_s
        sample_period = float(self.p("sample_period_s"))
        next_sample = time.monotonic()
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.01, sample_period))
            self.assert_wrench_safe()
            now = time.monotonic()
            if now >= next_sample:
                self.append_sample(phase, reference_position, reference_orientation)
                next_sample = now + sample_period

    def append_sample(
        self,
        phase: str,
        reference_position: Sequence[float],
        reference_orientation: Sequence[float],
    ) -> dict:
        self.assert_wrench_safe()
        position, orientation = self.pose_vector()
        now = time.monotonic()
        if self.run_t0_monotonic is None:
            self.run_t0_monotonic = now
        force_source = self.current_force_source or [math.nan] * 3
        force_sliding = self.current_force_sliding or [math.nan] * 3
        measurement_source = (
            self.current_measurement_force_source or [math.nan] * 3
        )
        measurement_sliding = (
            self.current_measurement_force_sliding or [math.nan] * 3
        )
        measurement_age = (
            now - self.last_measurement_wrench_monotonic
            if self.last_measurement_wrench_monotonic is not None
            else math.nan
        )
        sample = {
            "time_s": now - self.run_t0_monotonic,
            "phase": phase,
            "wrench_source_frame": self.current_wrench_source_frame,
            "tcp_x_R_m": position[0],
            "tcp_y_R_m": position[1],
            "tcp_z_R_m": position[2],
            "dx_R_m": position[0] - reference_position[0],
            "dy_R_m": position[1] - reference_position[1],
            "dz_R_m": position[2] - reference_position[2],
            "orientation_error_rad": quat_angle(orientation, reference_orientation),
            "force_x_source_N": force_source[0],
            "force_y_source_N": force_source[1],
            "force_z_source_N": force_source[2],
            "force_x_R_N": force_sliding[0],
            "force_y_R_N": force_sliding[1],
            "force_z_R_N": force_sliding[2],
            "force_norm_R_N": math.sqrt(sum(value * value for value in force_sliding)),
            "wrench_age_s": (
                now - self.last_wrench_monotonic
                if self.last_wrench_monotonic is not None
                else math.nan
            ),
            "measurement_wrench_source_frame": (
                self.current_measurement_wrench_source_frame
            ),
            "measurement_force_x_source_N": measurement_source[0],
            "measurement_force_y_source_N": measurement_source[1],
            "measurement_force_z_source_N": measurement_source[2],
            "measurement_force_x_R_N": measurement_sliding[0],
            "measurement_force_y_R_N": measurement_sliding[1],
            "measurement_force_z_R_N": measurement_sliding[2],
            "measurement_force_norm_R_N": math.sqrt(
                sum(value * value for value in measurement_sliding)
            ),
            "measurement_wrench_age_s": measurement_age,
        }
        self.samples.append(sample)
        return sample

    def assert_flat_scan_safe(self, sample: dict, target_offset_m: float) -> None:
        if abs(float(sample["dy_R_m"])) > float(self.p("abort_abs_dy_from_slide_start_m")):
            raise RuntimeError(
                f"Tangential deviation too large: dy_R={sample['dy_R_m'] * 1000:.3f} mm."
            )
        if abs(float(sample["dz_R_m"])) > float(self.p("abort_abs_dz_from_baseline_m")):
            raise RuntimeError(
                f"Normal deviation too large for flat reference: dz_R={sample['dz_R_m'] * 1000:.3f} mm."
            )
        if float(sample["orientation_error_rad"]) > float(self.p("abort_orientation_rad")):
            raise RuntimeError(
                "Orientation deviation too large: "
                f"{math.degrees(sample['orientation_error_rad']):.3f} deg."
            )
        dx = float(sample["dx_R_m"])
        if dx < -float(self.p("abort_radial_undershoot_m")):
            raise RuntimeError(f"Unexpected reverse radial motion: dx_R={dx * 1000:.3f} mm.")
        if dx > target_offset_m + float(self.p("abort_radial_overshoot_m")):
            raise RuntimeError(f"Radial overshoot: dx_R={dx * 1000:.3f} mm.")

    def execute_force_controlled_scan(
        self,
        baseline_position: Sequence[float],
        baseline_orientation: Sequence[float],
    ) -> None:
        speed = float(self.p("radial_speed_mps"))
        if speed <= 0.0:
            raise ValueError("radial_speed_mps must be > 0.")
        goal = self.create_goal(
            self.active_waypoints,
            [offset / speed for offset in self.active_offsets],
        )
        target_offset = self.active_offsets[-1]

        send_future = self.passthrough_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Force-controlled scan trajectory rejected.")

        self.publish_event("slide_started", target_distance_m=target_offset, radial_speed_mps=speed)
        result_future = handle.get_result_async()
        sample_period = float(self.p("sample_period_s"))
        next_sample = time.monotonic()
        contact_loss_since: Optional[float] = None
        try:
            while not result_future.done():
                rclpy.spin_once(self, timeout_sec=min(0.01, sample_period))
                self.assert_wrench_safe()
                now = time.monotonic()
                current_abs_force = abs(
                    float(self.current_measurement_force_sliding[2])
                )
                if current_abs_force < float(self.p("min_abs_contact_force_n")):
                    if contact_loss_since is None:
                        contact_loss_since = now
                    elif now - contact_loss_since >= float(self.p("contact_loss_persistence_s")):
                        raise RuntimeError(
                            "Contact lost during flat reference scan: "
                            f"|Fz_R|={current_abs_force:.3f} N."
                        )
                else:
                    contact_loss_since = None
                if now >= next_sample:
                    sample = self.append_sample(
                        "slide", baseline_position, baseline_orientation
                    )
                    self.assert_flat_scan_safe(sample, target_offset)
                    next_sample = now + sample_period
        except Exception:
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
            raise

        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError("Scan trajectory returned no result.")
        if wrapped.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                "Scan trajectory controller error: "
                f"code={wrapped.result.error_code}, "
                f"message={wrapped.result.error_string!r}."
            )
        self.append_sample("slide", baseline_position, baseline_orientation)
        self.publish_event("slide_finished")

    def lift_at_endpoint(self) -> None:
        """Move from force-displaced endpoint to the nominal IK endpoint."""
        goal = self.create_goal(
            [self.active_waypoints[-1]],
            [float(self.p("lift_to_nominal_endpoint_duration_s"))],
        )
        self.publish_event("endpoint_lift_started")
        self.send_goal_without_sampling(self.scaled_client, goal)
        self.publish_event("endpoint_lift_finished")

    def return_goal(self) -> FollowJointTrajectory.Goal:
        speed = float(self.p("return_speed_mps"))
        if speed <= 0.0:
            raise ValueError("return_speed_mps must be > 0.")

        positions: list[list[float]] = []
        times: list[float] = []
        elapsed = 0.0
        previous_offset = self.active_offsets[-1]
        for offset, joints in zip(
            reversed(self.active_offsets[:-1]), reversed(self.active_waypoints[:-1])
        ):
            elapsed += (previous_offset - offset) / speed
            positions.append(joints)
            times.append(elapsed)
            previous_offset = offset
        elapsed += previous_offset / speed
        positions.append([float(value) for value in self.p("start_joints")])
        times.append(elapsed)
        return self.create_goal(positions, times)

    def start_bag(self) -> Optional[Path]:
        if not bool(self.p("record_bag")):
            return None
        root = Path(str(self.p("bag_root"))).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        path = root / str(self.p("run_name"))
        if path.exists():
            raise FileExistsError(f"Bag already exists: {path}")

        command = [
            "ros2", "bag", "record", "-s", "mcap", "-o", str(path),
            "--qos-profile-overrides-path",
            str(Path(str(self.p("bag_qos_overrides_path"))).expanduser()),
            "--topics",
            "/tf", "/tf_static", "/joint_states",
            "/tcp_pose_broadcaster/pose",
            str(self.p("wrench_topic")),
            str(self.p("measurement_wrench_topic")),
            str(self.p("event_topic")),
        ]
        self.bag_process = subprocess.Popen(command, start_new_session=True)
        discovery_timeout = float(self.p("bag_start_wait_s"))
        if discovery_timeout <= 0.0:
            raise ValueError("bag_start_wait_s must be > 0.")
        deadline = time.monotonic() + discovery_timeout
        subscribed_since: Optional[float] = None
        while time.monotonic() < deadline:
            # Keep ATI, robot-program and TF callbacks alive while rosbag2
            # performs DDS discovery. A blocking sleep here can create a
            # false measurement-stale abort immediately after Force Mode starts.
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.bag_process.poll() is not None:
                raise RuntimeError("ros2 bag record terminated unexpectedly.")
            if self.event_publisher.get_subscription_count() > 0:
                if subscribed_since is None:
                    subscribed_since = time.monotonic()
                elif time.monotonic() - subscribed_since >= 0.5:
                    break
            else:
                subscribed_since = None
        if subscribed_since is None:
            raise RuntimeError(
                "rosbag did not subscribe to the Phase-B event topic before "
                "the discovery timeout. No Force Mode was started."
            )
        self.assert_wrench_safe()
        self.get_logger().info(f"Recording bag: {path}")
        self.publish_event("bag_started", bag_path=str(path))
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

    def compute_metrics(self) -> dict:
        slide = [row for row in self.samples if row["phase"] == "slide"]
        baseline = [row for row in self.samples if row["phase"] == "baseline"]
        if len(slide) < 3 or len(baseline) < 3:
            raise RuntimeError("Too few baseline or slide samples for evaluation.")

        dx = [float(row["dx_R_m"]) for row in slide]
        dy = [float(row["dy_R_m"]) for row in slide]
        dz = [float(row["dz_R_m"]) for row in slide]
        force_z = [float(row["force_z_R_N"]) for row in slide]
        measurement_force_z = [
            float(row["measurement_force_z_R_N"]) for row in slide
        ]
        orientation = [float(row["orientation_error_rad"]) for row in slide]

        slope, intercept = linear_fit(dx, dz)
        residuals = [z - (slope * x + intercept) for x, z in zip(dx, dz)]
        residual_peak_to_peak = max(residuals) - min(residuals)
        radial_error = abs(dx[-1] - self.active_offsets[-1])
        noise_tolerance = float(self.p("monotonic_noise_tolerance_m"))
        negative_steps = [
            current - previous
            for previous, current in zip(dx, dx[1:])
            if current - previous < -noise_tolerance
        ]
        # The separate read-only ATI TCP channel is authoritative for this pilot.
        # The UR broadcaster remains a diagnostic internal-sensor channel.
        mean_force = statistics.fmean(measurement_force_z)
        mean_force_error = mean_force - float(self.p("normal_force_n"))

        provisional_valid = (
            radial_error <= float(self.p("endpoint_radial_tolerance_m"))
            and max(abs(value) for value in dy) <= float(self.p("cross_axis_tolerance_m"))
            and max(orientation) <= float(self.p("orientation_tolerance_rad"))
            and not negative_steps
            and residual_peak_to_peak <= float(self.p("flat_residual_peak_to_peak_limit_m"))
            and abs(mean_force_error) <= float(self.p("mean_force_error_limit_n"))
        )

        return {
            "target_dx_R_m": self.active_offsets[-1],
            "measured_endpoint_dx_R_m": dx[-1],
            "endpoint_radial_error_m": radial_error,
            "max_abs_dy_R_m": max(abs(value) for value in dy),
            "max_abs_dz_R_m": max(abs(value) for value in dz),
            "max_orientation_error_rad": max(orientation),
            "max_orientation_error_deg": math.degrees(max(orientation)),
            "negative_radial_steps_over_tolerance": len(negative_steps),
            "z_profile_slope_dz_dx": slope,
            "z_profile_intercept_m": intercept,
            "z_residual_mean_m": statistics.fmean(residuals),
            "z_residual_std_m": sample_std(residuals),
            "z_residual_peak_to_peak_m": residual_peak_to_peak,
            "ati_force_z_R_mean_N": mean_force,
            "ati_force_z_R_std_N": sample_std(measurement_force_z),
            "ati_force_z_R_min_N": min(measurement_force_z),
            "ati_force_z_R_max_N": max(measurement_force_z),
            "ati_force_z_R_mean_error_N": mean_force_error,
            "ur_diagnostic_force_z_R_mean_N": statistics.fmean(force_z),
            "ur_diagnostic_force_z_R_std_N": sample_std(force_z),
            "measurement_force_z_R_mean_N": statistics.fmean(
                measurement_force_z
            ),
            "measurement_force_z_R_std_N": sample_std(
                measurement_force_z
            ),
            "measurement_force_z_R_min_N": min(measurement_force_z),
            "measurement_force_z_R_max_N": max(measurement_force_z),
            "baseline_ati_force_z_R_mean_N": statistics.fmean(
                [float(row["measurement_force_z_R_N"]) for row in baseline]
            ),
            "baseline_ati_force_z_R_std_N": sample_std(
                [float(row["measurement_force_z_R_N"]) for row in baseline]
            ),
            "n_baseline_samples": len(baseline),
            "n_slide_samples": len(slide),
            "provisional_reference_valid": bool(provisional_valid),
        }

    def save_results(self, status: str, metrics: Optional[dict], error: Optional[str]) -> Path:
        root = Path(str(self.p("result_root"))).expanduser()
        run_dir = root / str(self.p("run_name"))
        run_dir.mkdir(parents=True, exist_ok=True)

        csv_path = run_dir / "flat_reference_time_series.csv"
        fields = [
            "time_s", "phase", "wrench_source_frame",
            "tcp_x_R_m", "tcp_y_R_m", "tcp_z_R_m",
            "dx_R_m", "dy_R_m", "dz_R_m", "orientation_error_rad",
            "force_x_source_N", "force_y_source_N", "force_z_source_N",
            "force_x_R_N", "force_y_R_N", "force_z_R_N",
            "force_norm_R_N", "wrench_age_s",
            "measurement_wrench_source_frame",
            "measurement_force_x_source_N",
            "measurement_force_y_source_N",
            "measurement_force_z_source_N",
            "measurement_force_x_R_N",
            "measurement_force_y_R_N",
            "measurement_force_z_R_N",
            "measurement_force_norm_R_N",
            "measurement_wrench_age_s",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.samples)

        summary = {
            "status": status,
            "error": error,
            "run_name": str(self.p("run_name")),
            "waypoint_yaml_file": str(self.p("waypoint_yaml_file")),
            "normal_force_n": float(self.p("normal_force_n")),
            "radial_speed_mps": float(self.p("radial_speed_mps")),
            "max_waypoint_index": int(self.p("max_waypoint_index")),
            "control_wrench_topic": str(self.p("wrench_topic")),
            "measurement_wrench_topic": str(
                self.p("measurement_wrench_topic")
            ),
            "authoritative_force_source": "ati_rdt_bridge_biased",
            "parameters": {
                "endpoint_radial_tolerance_m": float(self.p("endpoint_radial_tolerance_m")),
                "cross_axis_tolerance_m": float(self.p("cross_axis_tolerance_m")),
                "flat_residual_peak_to_peak_limit_m": float(self.p("flat_residual_peak_to_peak_limit_m")),
                "mean_force_error_limit_n": float(self.p("mean_force_error_limit_n")),
            },
            "metrics": metrics,
        }
        json_path = run_dir / "summary.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.get_logger().info(f"Saved Phase-B results: {csv_path}, {json_path}")
        return run_dir

    def validate_configuration(self) -> None:
        if not bool(self.p("phase_a_validated")):
            raise RuntimeError("phase_a_validated is false.")
        if not bool(self.p("flat_surface_confirmed")):
            raise RuntimeError(
                "flat_surface_confirmed is false. Confirm that the selected "
                "scan path lies on a continuous flat surface without a chamfer."
            )
        if float(self.p("radial_speed_mps")) <= 0.0:
            raise ValueError("radial_speed_mps must be > 0.")

        # Reserve neither path here, but reject collisions before interfaces,
        # prompts or robot motion. Existing experimental artefacts are never
        # overwritten by a retry or an early-abort summary.
        run_name = str(self.p("run_name"))
        result_path = Path(str(self.p("result_root"))).expanduser() / run_name
        if result_path.exists():
            raise FileExistsError(
                f"Result directory already exists: {result_path}. "
                "Choose a unique run_name before starting Phase B."
            )
        if bool(self.p("record_bag")):
            bag_path = Path(str(self.p("bag_root"))).expanduser() / run_name
            if bag_path.exists():
                raise FileExistsError(
                    f"Bag already exists: {bag_path}. Choose a unique "
                    "run_name before starting Phase B."
                )

    def run(self) -> None:
        self.validate_configuration()
        self.wait_interfaces()
        self.load_waypoints()
        self.validate_ati_bias()

        input(
            "\nPHASE B – MOTION APPROVAL\n"
            "The standalone ATI RDT stream and RTDE bridge software bias are "
            "active; the bias-corrected ATI wrench check passed.\n"
            "Press Enter to move to the validated Phase-A start pose, or "
            "Ctrl+C to abort..."
        )
        self.switch_scaled()
        self.move_to_start()
        self.validate_start()

        input(
            "\nPHASE B – FLAT REFERENCE SURFACE\n"
            f"The robot will establish {float(self.p('normal_force_n')):.1f} N "
            f"in z_R and slide {self.active_offsets[-1] * 1000:.1f} mm at "
            f"{float(self.p('radial_speed_mps')) * 1000:.1f} mm/s.\n"
            "Confirm that the ENTIRE path is a continuous flat surface, "
            "reduce robot speed, and keep emergency stop reachable.\n"
            "Press Enter to start ATI-based Force Mode, or Ctrl+C to abort..."
        )

        bag_path: Optional[Path] = None
        metrics: Optional[dict] = None
        status = "FAILED"
        error_text: Optional[str] = None

        try:
            bag_path = self.start_bag()
            self.switch_force()
            self.start_force_mode()
            self.wait_contact()

            contact_position, contact_orientation = self.pose_vector()
            self.hold_and_sample(
                float(self.p("stabilisation_after_contact_s")),
                "stabilisation",
                contact_position,
                contact_orientation,
            )
            baseline_position, baseline_orientation = self.pose_vector()
            self.publish_event("baseline_started")
            self.run_t0_monotonic = time.monotonic()
            self.samples.clear()
            self.hold_and_sample(
                float(self.p("baseline_duration_s")),
                "baseline",
                baseline_position,
                baseline_orientation,
            )
            self.publish_event("baseline_finished")

            self.execute_force_controlled_scan(
                baseline_position, baseline_orientation
            )
            metrics = self.compute_metrics()
            self.publish_event("metrics_computed", **metrics)

            self.stop_force_mode()
            time.sleep(float(self.p("pause_after_force_stop_s")))
            self.switch_scaled()
            self.lift_at_endpoint()
            time.sleep(float(self.p("pause_after_lift_s")))

            self.publish_event("return_started")
            self.send_goal_without_sampling(self.scaled_client, self.return_goal())
            self.publish_event("return_finished")
            self.validate_start()

            status = (
                "COMPLETED_PROVISIONALLY_VALID"
                if metrics["provisional_reference_valid"]
                else "COMPLETED_REVIEW_REQUIRED"
            )
            self.publish_event("phase_b_completed", status=status)
            self.get_logger().info(
                "Phase B completed. "
                f"Provisional reference valid={metrics['provisional_reference_valid']}."
            )
        except Exception as exc:
            status = "INVALID_OR_ABORTED"
            error_text = str(exc)
            try:
                self.publish_event("phase_b_aborted", error=error_text)
            except Exception:
                pass
            raise
        finally:
            try:
                self.stop_force_mode()
            except Exception as exc:
                self.get_logger().error(f"Force-mode cleanup failed: {exc}")
            try:
                self.switch_scaled()
            except Exception as exc:
                self.get_logger().error(f"Controller cleanup failed: {exc}")
            self.stop_bag()
            self.save_results(status, metrics, error_text)
            if bag_path is not None:
                self.get_logger().info(f"Bag finalized: {bag_path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PhaseBFlatReference()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warning("Interrupted by user.")
    except Exception as exc:
        node.get_logger().error(str(exc))
        raise
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

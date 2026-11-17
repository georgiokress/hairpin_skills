#!/usr/bin/env python3
"""
Phase C: Wiederholbare kraftgeregelte Fasenabtastung.

Ziel
----
- Die in Phase A validierte IK-Mehrpunkttrajektorie unter Kraftregelung
  reproduzierbar über die Fase ausführen.
- Pro Lauf TCP-Pose und Kräfte im sliding_frame aufzeichnen.
- Den Lauf kontrolliert beenden, sobald eine sichere normale Absenkung oder
  ein persistenter Kontaktverlust im Fasenbereich erkannt wird.
- Positions-, Kraft- und kombinierte Fasenindikatoren getrennt auswerten.

Wichtig
-------
- Die Online-Stoppbedingung dient ausschließlich der Sicherheit und ist nicht
  gleichbedeutend mit dem geschätzten Fasenbeginn.
- Die eigentliche Detektion wird nach dem Lauf aus gefilterten Signalen und
  einem linearen Referenzprofil der ebenen Vorlaufstrecke berechnet.
- Vor der Zehnerreihe ist ein einzelner Pilotlauf erforderlich.
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
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
from rclpy.time import Time
from std_msgs.msg import String
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


def rolling_median(values: Sequence[float], window_samples: int) -> list[float]:
    """Centered rolling median with truncated windows at the boundaries."""
    if window_samples < 1:
        raise ValueError("window_samples must be >= 1.")
    if window_samples % 2 == 0:
        window_samples += 1
    radius = window_samples // 2
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        output.append(float(statistics.median(values[start:stop])))
    return output


def first_persistent_index(
    condition: Sequence[bool],
    times_s: Sequence[float],
    persistence_s: float,
) -> Optional[int]:
    """Return first index where condition stays true for persistence_s."""
    if len(condition) != len(times_s):
        raise ValueError("condition and times_s must have equal length.")
    start_index: Optional[int] = None
    for index, active in enumerate(condition):
        if active:
            if start_index is None:
                start_index = index
            if times_s[index] - times_s[start_index] >= persistence_s:
                return start_index
        else:
            start_index = None
    return None


def shortest_angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


class PhaseCChamferScan(Node):
    def __init__(self) -> None:
        super().__init__("phase_c_chamfer_scan")

        # Freigaben
        self.declare_parameter("phase_a_validated", False)
        self.declare_parameter("phase_b_reference_validated", False)
        self.declare_parameter("chamfer_path_confirmed", False)

        # Frames
        self.declare_parameter("sliding_frame", "sliding_frame")
        self.declare_parameter("tcp_frame", "tool0_controller")
        self.declare_parameter("tf_timeout_s", 10.0)
        self.declare_parameter("publish_sliding_frame", True)
        self.declare_parameter("sliding_parent_frame", "base")
        self.declare_parameter("sliding_tf_x_m", -0.22737)
        self.declare_parameter("sliding_tf_y_m", -0.40030)
        self.declare_parameter("sliding_tf_z_m", 0.03517)
        self.declare_parameter("sliding_tf_qx", -0.029727)
        self.declare_parameter("sliding_tf_qy", 0.047563)
        self.declare_parameter("sliding_tf_qz", -0.975264)
        self.declare_parameter("sliding_tf_qw", -0.213808)

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
        self.declare_parameter("max_waypoint_index", 5)
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

        # Glatte Zeitparametrisierung der IK-Wegpunkte
        self.declare_parameter("use_waypoint_velocities", True)
        self.declare_parameter("use_waypoint_accelerations", True)
        self.declare_parameter("endpoint_velocity_zero", True)

        # Online-Sicherheits- und kontrollierte Stoppbedingungen
        self.declare_parameter("abort_abs_dy_from_slide_start_m", 0.0015)
        self.declare_parameter("hard_abort_abs_dz_from_baseline_m", 0.0020)
        self.declare_parameter("controlled_stop_abs_dz_m", 0.00120)
        self.declare_parameter("controlled_stop_min_x_m", 0.0075)
        self.declare_parameter("controlled_stop_persistence_s", 0.06)
        self.declare_parameter(
            "abort_orientation_rad", math.radians(6.0)
        )
        self.declare_parameter("abort_radial_undershoot_m", 0.0005)
        self.declare_parameter("abort_radial_overshoot_m", 0.0010)
        self.declare_parameter("early_contact_loss_max_x_m", 0.0070)

        # Offline-Auswertung: ebener Referenzbereich und Detektoren
        self.declare_parameter("reference_fit_x_min_m", 0.0010)
        self.declare_parameter("reference_fit_x_max_m", 0.0070)
        self.declare_parameter("detector_min_x_m", 0.0070)
        self.declare_parameter("filter_window_s", 0.15)
        self.declare_parameter("detector_persistence_s", 0.12)
        self.declare_parameter("position_residual_threshold_m", -0.00025)
        self.declare_parameter("force_relief_threshold_n", 1.5)
        self.declare_parameter("combined_position_threshold_m", -0.00020)
        self.declare_parameter("combined_force_relief_threshold_n", 1.0)

        # Abheben und Rückfahrt
        self.declare_parameter("lift_to_nominal_endpoint_duration_s", 2.0)
        self.declare_parameter("return_speed_mps", 0.001)
        self.declare_parameter("pause_after_force_stop_s", 0.25)
        self.declare_parameter("pause_after_lift_s", 0.25)

        # Abtastung und Ausgabe
        self.declare_parameter("sample_period_s", 0.02)
        self.declare_parameter(
            "result_root",
            str(Path.home() / "masterarbeit" / "results" / "phase_c"),
        )
        self.declare_parameter("run_name", "phase_c_chamfer_pilot_01")
        self.declare_parameter(
            "bag_root", str(Path.home() / "masterarbeit" / "bags")
        )
        self.declare_parameter("record_bag", True)
        self.declare_parameter("bag_start_wait_s", 2.0)
        self.declare_parameter("tare_wait_s", 1.0)
        self.declare_parameter(
            "event_topic", "/phase_c_chamfer_scan/events"
        )

        # ROS-Schnittstellen
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
        self.zero_client = self.create_client(
            Trigger, "/io_and_status_controller/zero_ftsensor"
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
        self.force_mode_active = False
        self.bag_process: Optional[subprocess.Popen] = None
        self.samples: list[dict] = []
        self.run_t0_monotonic: Optional[float] = None
        self.stop_reason: str = "not_started"
        self.stop_sample: Optional[dict] = None

        if bool(self.p("publish_sliding_frame")):
            self.publish_sliding_frame()

    def p(self, name: str):
        return self.get_parameter(name).value

    def publish_sliding_frame(self) -> None:
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

    def _wrench_cb(self, msg: WrenchStamped) -> None:
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

        self.current_force_source = force_source
        self.current_force_sliding = [float(value) for value in force_sliding]
        self.current_wrench_source_frame = source
        self.last_wrench_monotonic = time.monotonic()

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
                "completed and documented before Phase C."
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
        velocities: Optional[Sequence[Sequence[float]]] = None,
        accelerations: Optional[Sequence[Sequence[float]]] = None,
    ) -> FollowJointTrajectory.Goal:
        if len(positions) != len(times_s) or not positions:
            raise ValueError("Invalid trajectory point/time lists.")
        if any(current <= previous for previous, current in zip(times_s, times_s[1:])):
            raise ValueError("Trajectory times must be strictly increasing.")
        if velocities is not None and len(velocities) != len(positions):
            raise ValueError("Velocity list length differs from positions.")
        if accelerations is not None and len(accelerations) != len(positions):
            raise ValueError("Acceleration list length differs from positions.")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.joint_names)
        for index, (joints, time_s) in enumerate(zip(positions, times_s)):
            if len(joints) != 6:
                raise ValueError("Each trajectory point requires six joints.")
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in joints]
            if velocities is not None:
                if len(velocities[index]) != 6:
                    raise ValueError("Each velocity point requires six joints.")
                point.velocities = [float(value) for value in velocities[index]]
            if accelerations is not None:
                if len(accelerations[index]) != 6:
                    raise ValueError("Each acceleration point requires six joints.")
                point.accelerations = [
                    float(value) for value in accelerations[index]
                ]
            point.time_from_start = duration_msg(float(time_s))
            goal.trajectory.points.append(point)
        return goal

    def create_scan_goal(self) -> FollowJointTrajectory.Goal:
        """Create scan goal with finite-difference waypoint velocities."""
        speed = float(self.p("radial_speed_mps"))
        if speed <= 0.0:
            raise ValueError("radial_speed_mps must be > 0.")

        positions = self.active_waypoints
        times = [offset / speed for offset in self.active_offsets]
        if not bool(self.p("use_waypoint_velocities")):
            return self.create_goal(positions, times)

        all_positions = [
            [float(value) for value in self.p("start_joints")],
            *[list(point) for point in self.active_waypoints],
        ]
        all_times = [0.0, *times]
        velocities: list[list[float]] = []

        for active_index in range(1, len(all_positions)):
            is_last = active_index == len(all_positions) - 1
            if is_last and bool(self.p("endpoint_velocity_zero")):
                velocities.append([0.0] * 6)
                continue

            previous_index = max(0, active_index - 1)
            next_index = min(len(all_positions) - 1, active_index + 1)
            dt = all_times[next_index] - all_times[previous_index]
            if dt <= 0.0:
                raise RuntimeError("Invalid time spacing for velocity calculation.")
            velocity = [
                shortest_angle_delta(
                    all_positions[next_index][joint_index],
                    all_positions[previous_index][joint_index],
                ) / dt
                for joint_index in range(6)
            ]
            velocities.append(velocity)

        accelerations = None
        if bool(self.p("use_waypoint_accelerations")):
            accelerations = [[0.0] * 6 for _ in positions]

        return self.create_goal(
            positions,
            times,
            velocities=velocities,
            accelerations=accelerations,
        )

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

    def zero_sensor(self) -> None:
        future = self.zero_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError("FT sensor tare failed.")
        time.sleep(float(self.p("tare_wait_s")))
        self.publish_event("sensor_tared")

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
        if self.last_wrench_monotonic is None:
            raise RuntimeError("No wrench data received.")
        age = time.monotonic() - self.last_wrench_monotonic
        if age > float(self.p("wrench_timeout_s")):
            raise RuntimeError(f"Wrench data stale: age={age:.3f} s.")
        if self.current_force_sliding is None:
            raise RuntimeError("No transformed force available.")
        normal_force = self.current_force_sliding[2]
        if abs(normal_force) > float(self.p("max_abs_normal_force_n")):
            raise RuntimeError(
                f"Normal-force safety limit exceeded: Fz_R={normal_force:.3f} N."
            )

    def wait_contact(self) -> None:
        threshold = float(self.p("contact_threshold_n"))
        persistence = float(self.p("contact_persistence_s"))
        deadline = time.monotonic() + float(self.p("contact_timeout_s"))
        above_since: Optional[float] = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.005)
            self.assert_wrench_safe()
            value = abs(float(self.current_force_sliding[2]))
            if value >= threshold:
                if above_since is None:
                    above_since = time.monotonic()
                elif time.monotonic() - above_since >= persistence:
                    self.publish_event(
                        "contact_detected",
                        force_z_sliding_n=self.current_force_sliding[2],
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
            "wrench_age_s": now - float(self.last_wrench_monotonic),
        }
        self.samples.append(sample)
        return sample

    def assert_chamfer_scan_safe(
        self, sample: dict, target_offset_m: float
    ) -> None:
        if abs(float(sample["dy_R_m"])) > float(
            self.p("abort_abs_dy_from_slide_start_m")
        ):
            raise RuntimeError(
                f"Tangential deviation too large: "
                f"dy_R={sample['dy_R_m'] * 1000:.3f} mm."
            )
        if abs(float(sample["dz_R_m"])) > float(
            self.p("hard_abort_abs_dz_from_baseline_m")
        ):
            raise RuntimeError(
                f"Hard normal-deviation limit exceeded: "
                f"dz_R={sample['dz_R_m'] * 1000:.3f} mm."
            )
        if float(sample["orientation_error_rad"]) > float(
            self.p("abort_orientation_rad")
        ):
            raise RuntimeError(
                "Orientation deviation too large: "
                f"{math.degrees(sample['orientation_error_rad']):.3f} deg."
            )
        dx = float(sample["dx_R_m"])
        if dx < -float(self.p("abort_radial_undershoot_m")):
            raise RuntimeError(
                f"Unexpected reverse radial motion: dx_R={dx * 1000:.3f} mm."
            )
        if dx > target_offset_m + float(self.p("abort_radial_overshoot_m")):
            raise RuntimeError(
                f"Radial overshoot: dx_R={dx * 1000:.3f} mm."
            )

    def cancel_goal(self, handle) -> None:
        cancel_future = handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)

    def execute_force_controlled_scan(
        self,
        baseline_position: Sequence[float],
        baseline_orientation: Sequence[float],
    ) -> None:
        goal = self.create_scan_goal()
        speed = float(self.p("radial_speed_mps"))
        target_offset = self.active_offsets[-1]

        send_future = self.passthrough_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Force-controlled chamfer trajectory rejected.")

        self.stop_reason = "running"
        self.stop_sample = None
        self.publish_event(
            "slide_started",
            target_distance_m=target_offset,
            radial_speed_mps=speed,
            smoothed_velocities=bool(self.p("use_waypoint_velocities")),
            smoothed_accelerations=bool(self.p("use_waypoint_accelerations")),
        )
        result_future = handle.get_result_async()
        sample_period = float(self.p("sample_period_s"))
        next_sample = time.monotonic()
        contact_loss_since: Optional[float] = None
        descent_since: Optional[float] = None
        expected_stop = False

        try:
            while not result_future.done():
                rclpy.spin_once(self, timeout_sec=min(0.01, sample_period))
                self.assert_wrench_safe()
                now = time.monotonic()
                if now < next_sample:
                    continue

                sample = self.append_sample(
                    "slide", baseline_position, baseline_orientation
                )
                self.assert_chamfer_scan_safe(sample, target_offset)
                next_sample = now + sample_period

                dx = float(sample["dx_R_m"])
                dz = float(sample["dz_R_m"])
                abs_force = abs(float(sample["force_z_R_N"]))

                # Persistenter Kontaktverlust: vor der Fase Fehler, danach
                # kontrolliertes Ende eines verwertbaren Fasenlaufs.
                if abs_force < float(self.p("min_abs_contact_force_n")):
                    if contact_loss_since is None:
                        contact_loss_since = now
                    elif now - contact_loss_since >= float(
                        self.p("contact_loss_persistence_s")
                    ):
                        if dx < float(self.p("early_contact_loss_max_x_m")):
                            raise RuntimeError(
                                "Contact lost before chamfer region: "
                                f"x_R={dx * 1000:.3f} mm, "
                                f"|Fz_R|={abs_force:.3f} N."
                            )
                        self.stop_reason = "persistent_contact_loss"
                        self.stop_sample = sample
                        expected_stop = True
                else:
                    contact_loss_since = None

                # Kontrollierter Stopp nach ausreichend großer Absenkung im
                # bekannten Fasenbereich. Dies ist eine Sicherheitsbedingung,
                # nicht die spätere Schätzung des Fasenbeginns.
                if (
                    dx >= float(self.p("controlled_stop_min_x_m"))
                    and dz <= -abs(float(self.p("controlled_stop_abs_dz_m")))
                ):
                    if descent_since is None:
                        descent_since = now
                    elif now - descent_since >= float(
                        self.p("controlled_stop_persistence_s")
                    ):
                        self.stop_reason = "controlled_normal_descent"
                        self.stop_sample = sample
                        expected_stop = True
                else:
                    descent_since = None

                if expected_stop:
                    self.publish_event(
                        "controlled_scan_stop",
                        reason=self.stop_reason,
                        x_R_m=dx,
                        dz_R_m=dz,
                        force_z_R_N=float(sample["force_z_R_N"]),
                    )
                    self.cancel_goal(handle)
                    break
        except Exception:
            self.cancel_goal(handle)
            raise

        if not expected_stop:
            wrapped = result_future.result()
            if wrapped is None:
                raise RuntimeError("Chamfer trajectory returned no result.")
            if wrapped.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                raise RuntimeError(
                    "Chamfer trajectory controller error: "
                    f"code={wrapped.result.error_code}, "
                    f"message={wrapped.result.error_string!r}."
                )
            final_sample = self.append_sample(
                "slide", baseline_position, baseline_orientation
            )
            self.stop_reason = "trajectory_completed"
            self.stop_sample = final_sample

        self.publish_event("slide_finished", stop_reason=self.stop_reason)

    def nominal_joints_at_offset(self, offset_m: float) -> list[float]:
        offsets = [0.0, *self.offsets]
        joints = [
            [float(value) for value in self.p("start_joints")],
            *[list(point) for point in self.waypoints],
        ]
        clamped = max(0.0, min(float(offset_m), offsets[-1]))
        if clamped <= 0.0:
            return joints[0]
        for index in range(1, len(offsets)):
            if clamped <= offsets[index]:
                lower_offset = offsets[index - 1]
                upper_offset = offsets[index]
                alpha = (clamped - lower_offset) / (upper_offset - lower_offset)
                return [
                    joints[index - 1][joint_index]
                    + alpha
                    * shortest_angle_delta(
                        joints[index][joint_index],
                        joints[index - 1][joint_index],
                    )
                    for joint_index in range(6)
                ]
        return joints[-1]

    def lift_at_stop_position(self) -> float:
        """Return from force-displaced pose to nominal IK pose at current x."""
        if self.stop_sample is None:
            raise RuntimeError("No stop sample available for recovery.")
        stop_offset = max(0.0, float(self.stop_sample["dx_R_m"]))
        nominal_joints = self.nominal_joints_at_offset(stop_offset)
        goal = self.create_goal(
            [nominal_joints],
            [float(self.p("lift_to_nominal_endpoint_duration_s"))],
        )
        self.publish_event(
            "stop_position_lift_started", stop_offset_m=stop_offset
        )
        self.send_goal_without_sampling(self.scaled_client, goal)
        self.publish_event("stop_position_lift_finished")
        return stop_offset

    def return_goal_from_offset(self, stop_offset_m: float) -> FollowJointTrajectory.Goal:
        speed = float(self.p("return_speed_mps"))
        if speed <= 0.0:
            raise ValueError("return_speed_mps must be > 0.")

        candidate_pairs = [
            (offset, joints)
            for offset, joints in zip(self.offsets, self.waypoints)
            if offset < stop_offset_m - 1e-9
        ]
        positions: list[list[float]] = []
        times: list[float] = []
        elapsed = 0.0
        previous_offset = stop_offset_m
        for offset, joints in reversed(candidate_pairs):
            elapsed += max(0.001, (previous_offset - offset) / speed)
            positions.append(list(joints))
            times.append(elapsed)
            previous_offset = offset
        elapsed += max(0.001, previous_offset / speed)
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
            "--topics",
            "/tf", "/tf_static", "/joint_states",
            "/tcp_pose_broadcaster/pose",
            str(self.p("wrench_topic")),
            str(self.p("event_topic")),
        ]
        self.bag_process = subprocess.Popen(command, start_new_session=True)
        time.sleep(float(self.p("bag_start_wait_s")))
        if self.bag_process.poll() is not None:
            raise RuntimeError("ros2 bag record terminated unexpectedly.")
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
        if len(slide) < 20 or len(baseline) < 3:
            raise RuntimeError("Too few baseline or slide samples for evaluation.")

        times = [float(row["time_s"]) for row in slide]
        dx = [float(row["dx_R_m"]) for row in slide]
        dy = [float(row["dy_R_m"]) for row in slide]
        dz = [float(row["dz_R_m"]) for row in slide]
        force_z = [float(row["force_z_R_N"]) for row in slide]
        orientation = [float(row["orientation_error_rad"]) for row in slide]

        fit_min = float(self.p("reference_fit_x_min_m"))
        fit_max = float(self.p("reference_fit_x_max_m"))
        reference_indices = [
            index for index, x_value in enumerate(dx)
            if fit_min <= x_value <= fit_max
        ]
        if len(reference_indices) < 20:
            raise RuntimeError(
                "Too few samples in the configured flat reference interval: "
                f"[{fit_min * 1000:.3f}, {fit_max * 1000:.3f}] mm."
            )

        reference_x = [dx[index] for index in reference_indices]
        reference_z = [dz[index] for index in reference_indices]
        slope, intercept = linear_fit(reference_x, reference_z)
        z_reference = [slope * x_value + intercept for x_value in dx]
        residual = [
            z_value - reference_value
            for z_value, reference_value in zip(dz, z_reference)
        ]

        window_samples = max(
            1,
            int(round(
                float(self.p("filter_window_s"))
                / float(self.p("sample_period_s"))
            )),
        )
        if window_samples % 2 == 0:
            window_samples += 1
        residual_filtered = rolling_median(residual, window_samples)
        force_filtered = rolling_median(force_z, window_samples)

        flat_force_values = [force_filtered[index] for index in reference_indices]
        flat_force_mean = statistics.fmean(flat_force_values)
        flat_force_std = sample_std(flat_force_values)
        force_relief = [value - flat_force_mean for value in force_filtered]

        for row, reference_value, raw_residual, filtered_residual, filtered_force, relief in zip(
            slide,
            z_reference,
            residual,
            residual_filtered,
            force_filtered,
            force_relief,
        ):
            row["z_reference_m"] = reference_value
            row["z_residual_m"] = raw_residual
            row["z_residual_filtered_m"] = filtered_residual
            row["force_z_filtered_N"] = filtered_force
            row["force_relief_N"] = relief

        detector_min_x = float(self.p("detector_min_x_m"))
        persistence = float(self.p("detector_persistence_s"))
        position_threshold = float(self.p("position_residual_threshold_m"))
        force_threshold = float(self.p("force_relief_threshold_n"))
        combined_position_threshold = float(
            self.p("combined_position_threshold_m")
        )
        combined_force_threshold = float(
            self.p("combined_force_relief_threshold_n")
        )

        position_condition = [
            x_value >= detector_min_x
            and residual_value <= position_threshold
            for x_value, residual_value in zip(dx, residual_filtered)
        ]
        force_condition = [
            x_value >= detector_min_x and relief >= force_threshold
            for x_value, relief in zip(dx, force_relief)
        ]
        combined_condition = [
            x_value >= detector_min_x
            and residual_value <= combined_position_threshold
            and relief >= combined_force_threshold
            for x_value, residual_value, relief in zip(
                dx, residual_filtered, force_relief
            )
        ]

        position_index = first_persistent_index(
            position_condition, times, persistence
        )
        force_index = first_persistent_index(force_condition, times, persistence)
        combined_index = first_persistent_index(
            combined_condition, times, persistence
        )

        def detector_metrics(prefix: str, index: Optional[int]) -> dict:
            if index is None:
                return {
                    f"{prefix}_detected": False,
                    f"{prefix}_detection_x_R_m": None,
                    f"{prefix}_detection_time_s": None,
                    f"{prefix}_z_residual_filtered_m": None,
                    f"{prefix}_force_z_filtered_N": None,
                    f"{prefix}_force_relief_N": None,
                }
            return {
                f"{prefix}_detected": True,
                f"{prefix}_detection_x_R_m": dx[index],
                f"{prefix}_detection_time_s": times[index],
                f"{prefix}_z_residual_filtered_m": residual_filtered[index],
                f"{prefix}_force_z_filtered_N": force_filtered[index],
                f"{prefix}_force_relief_N": force_relief[index],
            }

        # Frühtrigger ohne räumliche Sperre zur Beurteilung der Robustheit.
        position_early_index = first_persistent_index(
            [value <= position_threshold for value in residual_filtered],
            times,
            persistence,
        )
        force_early_index = first_persistent_index(
            [value >= force_threshold for value in force_relief],
            times,
            persistence,
        )
        combined_early_index = first_persistent_index(
            [
                residual_value <= combined_position_threshold
                and relief >= combined_force_threshold
                for residual_value, relief in zip(
                    residual_filtered, force_relief
                )
            ],
            times,
            persistence,
        )

        noise_tolerance = float(self.p("monotonic_noise_tolerance_m"))
        negative_steps = [
            current - previous
            for previous, current in zip(dx, dx[1:])
            if current - previous < -noise_tolerance
        ]

        metrics = {
            "target_dx_R_m": self.active_offsets[-1],
            "measured_stop_dx_R_m": dx[-1],
            "stop_reason": self.stop_reason,
            "stop_dz_R_m": dz[-1],
            "stop_force_z_R_N": force_z[-1],
            "max_abs_dy_R_m": max(abs(value) for value in dy),
            "min_dz_R_m": min(dz),
            "max_orientation_error_rad": max(orientation),
            "max_orientation_error_deg": math.degrees(max(orientation)),
            "negative_radial_steps_over_tolerance": len(negative_steps),
            "reference_fit_x_min_m": fit_min,
            "reference_fit_x_max_m": fit_max,
            "reference_z_slope_dz_dx": slope,
            "reference_z_intercept_m": intercept,
            "reference_residual_std_m": sample_std(
                [residual[index] for index in reference_indices]
            ),
            "reference_residual_peak_to_peak_m": (
                max(residual[index] for index in reference_indices)
                - min(residual[index] for index in reference_indices)
            ),
            "flat_force_z_R_mean_N": flat_force_mean,
            "flat_force_z_R_std_N": flat_force_std,
            "slide_force_z_R_mean_N": statistics.fmean(force_z),
            "slide_force_z_R_std_N": sample_std(force_z),
            "slide_force_z_R_min_N": min(force_z),
            "slide_force_z_R_max_N": max(force_z),
            "filter_window_samples": window_samples,
            "n_reference_samples": len(reference_indices),
            "n_slide_samples": len(slide),
            "position_false_trigger_before_detector_min_x": bool(
                position_early_index is not None
                and dx[position_early_index] < detector_min_x
            ),
            "force_false_trigger_before_detector_min_x": bool(
                force_early_index is not None
                and dx[force_early_index] < detector_min_x
            ),
            "combined_false_trigger_before_detector_min_x": bool(
                combined_early_index is not None
                and dx[combined_early_index] < detector_min_x
            ),
            "usable_for_statistics": bool(
                dx[-1] >= float(self.p("controlled_stop_min_x_m"))
                and len(reference_indices) >= 20
            ),
        }
        metrics.update(detector_metrics("position", position_index))
        metrics.update(detector_metrics("force", force_index))
        metrics.update(detector_metrics("combined", combined_index))
        return metrics

    def save_results(
        self,
        status: str,
        metrics: Optional[dict],
        error: Optional[str],
    ) -> Path:
        root = Path(str(self.p("result_root"))).expanduser()
        run_dir = root / str(self.p("run_name"))
        run_dir.mkdir(parents=True, exist_ok=True)

        csv_path = run_dir / "chamfer_time_series.csv"
        fields = [
            "time_s", "phase", "wrench_source_frame",
            "tcp_x_R_m", "tcp_y_R_m", "tcp_z_R_m",
            "dx_R_m", "dy_R_m", "dz_R_m", "orientation_error_rad",
            "force_x_source_N", "force_y_source_N", "force_z_source_N",
            "force_x_R_N", "force_y_R_N", "force_z_R_N",
            "force_norm_R_N", "wrench_age_s",
            "z_reference_m", "z_residual_m", "z_residual_filtered_m",
            "force_z_filtered_N", "force_relief_N",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            for row in self.samples:
                completed = {field: row.get(field, math.nan) for field in fields}
                writer.writerow(completed)

        summary = {
            "status": status,
            "error": error,
            "run_name": str(self.p("run_name")),
            "waypoint_yaml_file": str(self.p("waypoint_yaml_file")),
            "normal_force_n": float(self.p("normal_force_n")),
            "radial_speed_mps": float(self.p("radial_speed_mps")),
            "max_waypoint_index": int(self.p("max_waypoint_index")),
            "parameters": {
                "controlled_stop_abs_dz_m": float(
                    self.p("controlled_stop_abs_dz_m")
                ),
                "hard_abort_abs_dz_from_baseline_m": float(
                    self.p("hard_abort_abs_dz_from_baseline_m")
                ),
                "reference_fit_x_min_m": float(
                    self.p("reference_fit_x_min_m")
                ),
                "reference_fit_x_max_m": float(
                    self.p("reference_fit_x_max_m")
                ),
                "detector_min_x_m": float(self.p("detector_min_x_m")),
                "filter_window_s": float(self.p("filter_window_s")),
                "detector_persistence_s": float(
                    self.p("detector_persistence_s")
                ),
                "position_residual_threshold_m": float(
                    self.p("position_residual_threshold_m")
                ),
                "force_relief_threshold_n": float(
                    self.p("force_relief_threshold_n")
                ),
                "combined_position_threshold_m": float(
                    self.p("combined_position_threshold_m")
                ),
                "combined_force_relief_threshold_n": float(
                    self.p("combined_force_relief_threshold_n")
                ),
                "use_waypoint_velocities": bool(
                    self.p("use_waypoint_velocities")
                ),
                "use_waypoint_accelerations": bool(
                    self.p("use_waypoint_accelerations")
                ),
            },
            "metrics": metrics,
        }
        json_path = run_dir / "summary.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.get_logger().info(
            f"Saved Phase-C results: {csv_path}, {json_path}"
        )
        return run_dir

    def validate_configuration(self) -> None:
        if not bool(self.p("phase_a_validated")):
            raise RuntimeError("phase_a_validated is false.")
        if not bool(self.p("phase_b_reference_validated")):
            raise RuntimeError("phase_b_reference_validated is false.")
        if not bool(self.p("chamfer_path_confirmed")):
            raise RuntimeError(
                "chamfer_path_confirmed is false. Confirm that the selected "
                "path contains the intended flat approach and chamfer."
            )
        if float(self.p("radial_speed_mps")) <= 0.0:
            raise ValueError("radial_speed_mps must be > 0.")
        if int(self.p("max_waypoint_index")) < 4:
            raise ValueError(
                "Phase C requires at least waypoint 4 so the chamfer region "
                "can be reached."
            )
        if float(self.p("reference_fit_x_max_m")) >= float(
            self.p("detector_min_x_m")
        ):
            raise ValueError(
                "reference_fit_x_max_m must be smaller than detector_min_x_m."
            )

    def run(self) -> None:
        self.validate_configuration()
        self.wait_interfaces()
        self.load_waypoints()
        self.switch_scaled()
        self.move_to_start()
        self.validate_start()

        input(
            "\nPHASE C – CHAMFER SCAN\n"
            f"The robot will establish {float(self.p('normal_force_n')):.1f} N "
            f"in z_R and scan up to {self.active_offsets[-1] * 1000:.1f} mm "
            f"at {float(self.p('radial_speed_mps')) * 1000:.1f} mm/s.\n"
            "The path must contain the validated flat approach followed by "
            "the intended chamfer. The scan may stop automatically after "
            "sufficient normal descent or persistent contact loss.\n"
            "Reduce robot speed and keep emergency stop reachable.\n"
            "Press Enter to tare and start, or Ctrl+C to abort..."
        )

        bag_path: Optional[Path] = None
        metrics: Optional[dict] = None
        status = "FAILED"
        error_text: Optional[str] = None

        try:
            self.zero_sensor()
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
            stop_offset = self.lift_at_stop_position()
            time.sleep(float(self.p("pause_after_lift_s")))

            self.publish_event("return_started", stop_offset_m=stop_offset)
            self.send_goal_without_sampling(
                self.scaled_client,
                self.return_goal_from_offset(stop_offset),
            )
            self.publish_event("return_finished")
            self.validate_start()

            status = (
                "COMPLETED_USABLE_FOR_STATISTICS"
                if metrics["usable_for_statistics"]
                else "COMPLETED_REVIEW_REQUIRED"
            )
            self.publish_event("phase_c_completed", status=status)
            self.get_logger().info(
                "Phase C completed. "
                f"usable_for_statistics={metrics['usable_for_statistics']}, "
                f"stop_reason={metrics['stop_reason']}, "
                f"position_detected={metrics['position_detected']}, "
                f"force_detected={metrics['force_detected']}, "
                f"combined_detected={metrics['combined_detected']}."
            )
        except Exception as exc:
            status = "INVALID_OR_ABORTED"
            error_text = str(exc)
            try:
                self.publish_event("phase_c_aborted", error=error_text)
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
    node = PhaseCChamferScan()
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

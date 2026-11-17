#!/usr/bin/env python3
"""
Phase A: Kontaktfreie Validierung einer radialen IK-Mehrpunkttrajektorie.

Der Node:
1. fährt mit dem scaled_joint_trajectory_controller zur kalibrierten Startpose,
2. lädt die mit MoveIt erzeugten IK-Wegpunkte aus einer YAML-Datei,
3. führt zunächst nur die ersten N Wegpunkte als EIN FollowJointTrajectory-Goal aus,
4. misst die reale TCP-Bahn fortlaufend im sliding_frame,
5. bricht bei groben Quer-/Orientierungsabweichungen ab,
6. validiert Endpunkt, Querabweichung, Orientierung und Monotonie,
7. fährt nach erfolgreicher Validierung entlang derselben Wegpunkte zurück,
8. speichert Rohdaten als CSV und Kennwerte als JSON.

Kein Force Mode. Keine Kontaktfahrt.
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Sequence

import rclpy
import yaml
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint


DEFAULT_JOINT_NAMES = [
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
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


class PhaseARadialIKValidator(Node):
    def __init__(self) -> None:
        super().__init__("phase_a_radial_ik_validator")

        # Frames
        self.declare_parameter("sliding_frame", "sliding_frame")
        self.declare_parameter("tcp_frame", "tool0_controller")

        # Kalibrierte Startpose aus dem 3-mm-Test
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

        # IK-Wegpunkte
        self.declare_parameter(
            "waypoint_yaml_file",
            str(Path.home() / "masterarbeit" / "radial_ik_waypoints.yaml"),
        )
        self.declare_parameter("max_waypoint_index", 1)
        self.declare_parameter("radial_speed_mps", 0.001)

        # Start- und Bahnvalidierung
        self.declare_parameter("start_position_tolerance_m", 0.001)
        self.declare_parameter(
            "start_orientation_tolerance_rad", math.radians(3.0)
        )
        self.declare_parameter("endpoint_radial_tolerance_m", 0.0005)
        self.declare_parameter("cross_axis_tolerance_m", 0.0005)
        self.declare_parameter(
            "orientation_tolerance_rad", math.radians(3.0)
        )
        self.declare_parameter("monotonic_noise_tolerance_m", 0.00010)

        # Harte Online-Abbruchgrenzen, bewusst größer als Auswertungstoleranzen
        self.declare_parameter("abort_cross_axis_m", 0.0015)
        self.declare_parameter(
            "abort_orientation_rad", math.radians(6.0)
        )
        self.declare_parameter("abort_radial_undershoot_m", 0.0005)
        self.declare_parameter("abort_radial_overshoot_m", 0.0010)
        self.declare_parameter("tf_timeout_s", 0.5)
        self.declare_parameter("sample_period_s", 0.02)

        # Rückfahrt
        self.declare_parameter("return_to_start_after_success", True)
        self.declare_parameter("pause_before_return_s", 0.5)

        # Ausgabe
        self.declare_parameter(
            "result_root",
            str(Path.home() / "masterarbeit" / "results" / "phase_a"),
        )
        self.declare_parameter("run_name", "phase_a_ik_wp01")

        # Optionales rosbag
        self.declare_parameter("record_bag", True)
        self.declare_parameter(
            "bag_root",
            str(Path.home() / "masterarbeit" / "bags"),
        )
        self.declare_parameter("bag_start_wait_s", 1.5)

        self.scaled_action = (
            "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )
        self.scaled_client = ActionClient(
            self, FollowJointTrajectory, self.scaled_action
        )
        self.switch_client = self.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        self.list_client = self.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bag_process: subprocess.Popen | None = None
        self.samples: list[dict] = []
        self.run_t0_monotonic: float | None = None

        self.joint_names: list[str] = []
        self.offsets: list[float] = []
        self.waypoints: list[list[float]] = []

    def p(self, name: str):
        return self.get_parameter(name).value

    def wait_interfaces(self) -> None:
        if not self.scaled_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(
                f"Action unavailable: {self.scaled_action}"
            )
        for client, name in (
            (self.switch_client, "switch_controller"),
            (self.list_client, "list_controllers"),
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

    def activate_scaled_controller(self) -> None:
        states = self.controller_states()
        activate = []
        deactivate = []

        if states.get("scaled_joint_trajectory_controller") != "active":
            activate.append("scaled_joint_trajectory_controller")

        for name in (
            "passthrough_trajectory_controller",
            "force_mode_controller",
        ):
            if states.get(name) == "active":
                deactivate.append(name)

        if not activate and not deactivate:
            return

        request = SwitchController.Request()
        request.activate_controllers = activate
        request.deactivate_controllers = deactivate
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True

        future = self.switch_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.ok:
            raise RuntimeError(
                "Controller switch failed: "
                f"activate={activate}, deactivate={deactivate}"
            )
        time.sleep(0.5)

    def load_waypoints(self) -> None:
        path = Path(str(self.p("waypoint_yaml_file"))).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Waypoint YAML not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        try:
            params = payload["radial_ik_waypoints"]["ros__parameters"]
            joint_names = [str(x) for x in params["joint_names"]]
            offsets = [float(x) for x in params["slide_waypoint_offsets_m"]]
            flat = [float(x) for x in params["slide_waypoint_joints_flat"]]
            group = str(params["planning_group_used"])
            ik_link = str(params["ik_link_name_used"])
            frame = str(params["sliding_frame_used"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid waypoint YAML structure: {path}"
            ) from exc

        if joint_names != DEFAULT_JOINT_NAMES:
            raise RuntimeError(
                "Joint order differs from the validated controller order.\n"
                f"Expected: {DEFAULT_JOINT_NAMES}\n"
                f"YAML:     {joint_names}"
            )
        if len(flat) != 6 * len(offsets):
            raise RuntimeError(
                f"Expected {6 * len(offsets)} joint values, got {len(flat)}."
            )
        if any(b <= a for a, b in zip(offsets, offsets[1:])):
            raise RuntimeError("Waypoint offsets must be strictly increasing.")
        if offsets[0] <= 0.0:
            raise RuntimeError("First waypoint offset must be > 0.")
        if frame != str(self.p("sliding_frame")):
            raise RuntimeError(
                f"Waypoint frame '{frame}' differs from configured "
                f"sliding_frame '{self.p('sliding_frame')}'."
            )

        self.joint_names = joint_names
        self.offsets = offsets
        self.waypoints = [
            flat[i : i + 6] for i in range(0, len(flat), 6)
        ]

        max_index = int(self.p("max_waypoint_index"))
        if max_index < 1 or max_index > len(self.waypoints):
            raise ValueError(
                f"max_waypoint_index must be between 1 and "
                f"{len(self.waypoints)}."
            )

        self.get_logger().info(
            f"Loaded {len(self.waypoints)} IK waypoints from {path}. "
            f"Planning group={group}, IK link={ik_link}. "
            f"This run executes waypoints 1..{max_index}, "
            f"target offset={self.offsets[max_index - 1]*1000:.3f} mm."
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
            f"p_R={[round(v, 6) for v in position]}, "
            f"position error={position_error*1000:.3f} mm, "
            f"orientation error={math.degrees(orientation_error):.3f} deg."
        )

        if position_error > float(self.p("start_position_tolerance_m")):
            raise RuntimeError("Start position outside tolerance.")
        if orientation_error > float(
            self.p("start_orientation_tolerance_rad")
        ):
            raise RuntimeError("Start orientation outside tolerance.")

        return position, orientation

    def create_goal(
        self,
        positions: Sequence[Sequence[float]],
        times_s: Sequence[float],
    ) -> FollowJointTrajectory.Goal:
        if len(positions) != len(times_s):
            raise ValueError("positions and times_s must have equal length.")
        if not positions:
            raise ValueError("Trajectory must contain at least one point.")
        if any(b <= a for a, b in zip(times_s, times_s[1:])):
            raise ValueError("Trajectory times must be strictly increasing.")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.joint_names)

        for joints, time_s in zip(positions, times_s):
            if len(joints) != 6:
                raise ValueError("Each waypoint must contain six joints.")
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in joints]
            point.time_from_start = duration_msg(float(time_s))
            goal.trajectory.points.append(point)

        return goal

    def append_sample(
        self,
        phase: str,
        start_position: Sequence[float],
        start_orientation: Sequence[float],
    ) -> dict:
        position, orientation = self.pose_vector()
        now = time.monotonic()
        if self.run_t0_monotonic is None:
            self.run_t0_monotonic = now

        sample = {
            "time_s": now - self.run_t0_monotonic,
            "phase": phase,
            "tcp_x_R_m": position[0],
            "tcp_y_R_m": position[1],
            "tcp_z_R_m": position[2],
            "dx_R_m": position[0] - start_position[0],
            "dy_R_m": position[1] - start_position[1],
            "dz_R_m": position[2] - start_position[2],
            "orientation_error_rad": quat_angle(
                orientation, start_orientation
            ),
        }
        self.samples.append(sample)
        return sample

    def assert_online_path_safe(
        self,
        sample: dict,
        target_offset_m: float,
    ) -> None:
        cross = max(abs(sample["dy_R_m"]), abs(sample["dz_R_m"]))
        if cross > float(self.p("abort_cross_axis_m")):
            raise RuntimeError(
                "Online path abort: cross-axis deviation "
                f"{cross*1000:.3f} mm exceeds "
                f"{float(self.p('abort_cross_axis_m'))*1000:.3f} mm."
            )

        orientation_error = float(sample["orientation_error_rad"])
        if orientation_error > float(self.p("abort_orientation_rad")):
            raise RuntimeError(
                "Online path abort: orientation deviation "
                f"{math.degrees(orientation_error):.3f} deg."
            )

        dx = float(sample["dx_R_m"])
        if dx < -float(self.p("abort_radial_undershoot_m")):
            raise RuntimeError(
                f"Online path abort: radial motion reversed to "
                f"{dx*1000:.3f} mm."
            )
        if dx > target_offset_m + float(
            self.p("abort_radial_overshoot_m")
        ):
            raise RuntimeError(
                f"Online path abort: radial overshoot, "
                f"dx_R={dx*1000:.3f} mm."
            )

    def execute_goal_and_sample(
        self,
        goal: FollowJointTrajectory.Goal,
        phase: str,
        start_position: Sequence[float],
        start_orientation: Sequence[float],
        target_offset_m: float,
        monitor_path: bool,
    ) -> None:
        send_future = self.scaled_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Trajectory goal rejected.")

        result_future = handle.get_result_async()
        sample_period = float(self.p("sample_period_s"))
        next_sample = time.monotonic()

        try:
            while not result_future.done():
                rclpy.spin_once(self, timeout_sec=min(0.01, sample_period))
                now = time.monotonic()
                if now >= next_sample:
                    sample = self.append_sample(
                        phase, start_position, start_orientation
                    )
                    if monitor_path:
                        self.assert_online_path_safe(
                            sample, target_offset_m
                        )
                    next_sample = now + sample_period
        except Exception:
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(
                self, cancel_future, timeout_sec=2.0
            )
            raise

        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError("Trajectory returned no result.")
        result = wrapped.result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                "Trajectory controller reported failure: "
                f"error_code={result.error_code}, "
                f"error_string='{result.error_string}'."
            )

        self.append_sample(phase, start_position, start_orientation)

    def move_to_start(self) -> None:
        goal = self.create_goal(
            [[float(x) for x in self.p("start_joints")]],
            [float(self.p("move_to_start_duration_s"))],
        )
        # No phase-A path monitoring during the known approach.
        position, orientation = self.pose_vector()
        self.execute_goal_and_sample(
            goal=goal,
            phase="move_to_start",
            start_position=position,
            start_orientation=orientation,
            target_offset_m=1.0,
            monitor_path=False,
        )

    def outbound_goal(self, max_index: int):
        selected_positions = self.waypoints[:max_index]
        speed = float(self.p("radial_speed_mps"))
        if speed <= 0.0:
            raise ValueError("radial_speed_mps must be > 0.")
        times = [offset / speed for offset in self.offsets[:max_index]]
        return self.create_goal(selected_positions, times)

    def return_goal(self, max_index: int):
        speed = float(self.p("radial_speed_mps"))
        selected_offsets = self.offsets[:max_index]
        selected_waypoints = self.waypoints[:max_index]
        endpoint_offset = selected_offsets[-1]

        positions: list[list[float]] = []
        times: list[float] = []

        elapsed = 0.0
        previous_offset = endpoint_offset

        # Wegpunkte unterhalb des Endpunkts in umgekehrter Reihenfolge.
        for offset, joints in zip(
            reversed(selected_offsets[:-1]),
            reversed(selected_waypoints[:-1]),
        ):
            elapsed += (previous_offset - offset) / speed
            positions.append(joints)
            times.append(elapsed)
            previous_offset = offset

        elapsed += previous_offset / speed
        positions.append([float(x) for x in self.p("start_joints")])
        times.append(elapsed)

        return self.create_goal(positions, times)

    def validate_outbound(
        self,
        start_position: Sequence[float],
        target_offset_m: float,
    ) -> dict:
        outbound = [
            sample for sample in self.samples
            if sample["phase"] == "outbound"
        ]
        if len(outbound) < 3:
            raise RuntimeError("Too few outbound TCP samples.")

        dx = [float(row["dx_R_m"]) for row in outbound]
        dy = [float(row["dy_R_m"]) for row in outbound]
        dz = [float(row["dz_R_m"]) for row in outbound]
        orientation = [
            float(row["orientation_error_rad"]) for row in outbound
        ]

        endpoint_dx = dx[-1]
        radial_error = abs(endpoint_dx - target_offset_m)
        max_abs_dy = max(abs(v) for v in dy)
        max_abs_dz = max(abs(v) for v in dz)
        max_orientation = max(orientation)

        noise_tol = float(self.p("monotonic_noise_tolerance_m"))
        negative_steps = [
            current - previous
            for previous, current in zip(dx, dx[1:])
            if current - previous < -noise_tol
        ]
        max_negative_step = (
            min(negative_steps) if negative_steps else 0.0
        )

        valid = (
            radial_error
            <= float(self.p("endpoint_radial_tolerance_m"))
            and max_abs_dy
            <= float(self.p("cross_axis_tolerance_m"))
            and max_abs_dz
            <= float(self.p("cross_axis_tolerance_m"))
            and max_orientation
            <= float(self.p("orientation_tolerance_rad"))
            and not negative_steps
        )

        result = {
            "target_dx_R_m": target_offset_m,
            "measured_endpoint_dx_R_m": endpoint_dx,
            "endpoint_radial_error_m": radial_error,
            "max_abs_dy_R_m": max_abs_dy,
            "max_abs_dz_R_m": max_abs_dz,
            "max_orientation_error_rad": max_orientation,
            "max_orientation_error_deg": math.degrees(max_orientation),
            "negative_radial_steps_over_tolerance": len(negative_steps),
            "largest_negative_radial_step_m": max_negative_step,
            "n_outbound_samples": len(outbound),
            "valid": bool(valid),
        }

        self.get_logger().info(
            "Phase-A outbound metrics: "
            f"target dx={target_offset_m*1000:.3f} mm, "
            f"measured dx={endpoint_dx*1000:.3f} mm, "
            f"radial error={radial_error*1000:.3f} mm, "
            f"max |dy|={max_abs_dy*1000:.3f} mm, "
            f"max |dz|={max_abs_dz*1000:.3f} mm, "
            f"max orientation={math.degrees(max_orientation):.3f} deg, "
            f"negative steps={len(negative_steps)}, valid={valid}."
        )
        return result

    def start_bag(self) -> Path | None:
        if not bool(self.p("record_bag")):
            return None

        root = Path(str(self.p("bag_root"))).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        path = root / str(self.p("run_name"))

        if path.exists():
            raise FileExistsError(f"Bag already exists: {path}")

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
            "-o",
            str(path),
        ]
        self.bag_process = subprocess.Popen(
            command, start_new_session=True
        )
        time.sleep(float(self.p("bag_start_wait_s")))
        if self.bag_process.poll() is not None:
            raise RuntimeError("ros2 bag record terminated unexpectedly.")

        self.get_logger().info(f"Recording bag: {path}")
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

    def save_results(self, metrics: dict | None, status: str) -> Path:
        root = Path(str(self.p("result_root"))).expanduser()
        run_dir = root / str(self.p("run_name"))
        run_dir.mkdir(parents=True, exist_ok=True)

        csv_path = run_dir / "tcp_path_sliding_frame.csv"
        fieldnames = [
            "time_s",
            "phase",
            "tcp_x_R_m",
            "tcp_y_R_m",
            "tcp_z_R_m",
            "dx_R_m",
            "dy_R_m",
            "dz_R_m",
            "orientation_error_rad",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)

        summary = {
            "status": status,
            "run_name": str(self.p("run_name")),
            "waypoint_yaml_file": str(self.p("waypoint_yaml_file")),
            "max_waypoint_index": int(self.p("max_waypoint_index")),
            "radial_speed_mps": float(self.p("radial_speed_mps")),
            "metrics": metrics,
        }
        json_path = run_dir / "summary.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        self.get_logger().info(
            f"Saved Phase-A results: {csv_path}, {json_path}"
        )
        return run_dir

    def run(self) -> None:
        self.wait_interfaces()
        self.activate_scaled_controller()
        self.load_waypoints()

        self.get_logger().info(
            "Moving to the already calibrated sliding start pose."
        )
        self.move_to_start()
        start_position, start_orientation = self.validate_start()

        max_index = int(self.p("max_waypoint_index"))
        target_offset = self.offsets[max_index - 1]

        input(
            "\nPHASE A – CONTACT-FREE TEST\n"
            f"About to execute waypoints 1..{max_index} to "
            f"{target_offset*1000:.3f} mm in +x_R.\n"
            "Ensure the entire path is free, reduce robot speed, "
            "and keep emergency stop reachable.\n"
            "Press Enter to start, or Ctrl+C to abort..."
        )

        bag_path = self.start_bag()
        metrics = None
        status = "FAILED"
        try:
            self.run_t0_monotonic = time.monotonic()
            self.samples.clear()
            self.append_sample(
                "outbound", start_position, start_orientation
            )

            outbound = self.outbound_goal(max_index)
            self.execute_goal_and_sample(
                goal=outbound,
                phase="outbound",
                start_position=start_position,
                start_orientation=start_orientation,
                target_offset_m=target_offset,
                monitor_path=True,
            )

            metrics = self.validate_outbound(
                start_position, target_offset
            )
            if not metrics["valid"]:
                raise RuntimeError(
                    "Phase-A path failed the validation criteria. "
                    "No automatic return is executed."
                )

            status = "VALID"
            if bool(self.p("return_to_start_after_success")):
                time.sleep(float(self.p("pause_before_return_s")))
                self.get_logger().info(
                    "Outbound path valid. Returning along the same "
                    "IK waypoint path."
                )
                return_goal = self.return_goal(max_index)
                self.execute_goal_and_sample(
                    goal=return_goal,
                    phase="return",
                    start_position=start_position,
                    start_orientation=start_orientation,
                    target_offset_m=target_offset,
                    monitor_path=False,
                )
                self.validate_start()

            self.get_logger().info(
                f"Phase A succeeded for waypoint index {max_index}."
            )
        except Exception:
            status = "INVALID_OR_ABORTED"
            raise
        finally:
            self.stop_bag()
            self.save_results(metrics, status)
            if bag_path is not None:
                self.get_logger().info(f"Bag finalized: {bag_path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PhaseARadialIKValidator()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warning("Interrupted by user.")
    except Exception as exc:
        node.get_logger().error(str(exc))
        raise
    finally:
        try:
            node.stop_bag()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

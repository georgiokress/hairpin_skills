#!/usr/bin/env python3
"""Phase B with direct ATI feedback and Cartesian velocity control.

The external ATI TCP wrench is the only feedback signal used by this
controller.  UR Force Mode and the robot's internal F/T estimate are not used
for control.  A damped geometric Jacobian maps the requested Cartesian
velocity in ``sliding_frame`` to the UR driver's joint velocity controller.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rclpy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hairpin_skills.phase_b_flat_reference import (
    PhaseBFlatReference,
    duration_msg,
    quat_angle,
)


VELOCITY_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

JOINT_CHILD_FRAMES = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class PhaseBAtiVelocity(PhaseBFlatReference):
    """Execute the flat 5-mm reference scan using direct ATI feedback."""

    def __init__(self) -> None:
        super().__init__("phase_b_ati_velocity")

        self.declare_parameter(
            "velocity_command_topic",
            "/forward_velocity_controller/commands",
        )
        self.declare_parameter("control_rate_hz", 125.0)
        self.declare_parameter("command_horizon_s", 0.04)
        self.declare_parameter("joint_state_timeout_s", 0.10)
        self.declare_parameter("force_filter_alpha", 0.20)
        self.declare_parameter("force_kp_mps_per_n", 0.00025)
        self.declare_parameter("approach_direction_sign", -1.0)
        self.declare_parameter("max_normal_velocity_mps", 0.0005)
        self.declare_parameter("max_joint_velocity_radps", 0.15)
        self.declare_parameter("jacobian_damping", 0.02)
        self.declare_parameter("max_jacobian_condition", 250.0)
        self.declare_parameter("max_approach_displacement_m", 0.0030)
        self.declare_parameter("max_release_displacement_m", 0.0015)
        self.declare_parameter("release_velocity_mps", 0.0005)
        self.declare_parameter("release_force_threshold_n", 0.20)
        self.declare_parameter("release_clearance_m", 0.0005)
        self.declare_parameter("max_scan_duration_factor", 2.0)

        self.velocity_publisher = self.create_publisher(
            Float64MultiArray,
            str(self.p("velocity_command_topic")),
            10,
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 50)
        # Isolate joint feedback from the always-ready 500-Hz wrench
        # subscriptions on the main node. The control loop spins this node
        # explicitly before every trajectory command.
        self.joint_feedback_node = rclpy.create_node(
            "phase_b_ati_joint_feedback"
        )
        self.joint_feedback_subscription = (
            self.joint_feedback_node.create_subscription(
                JointState, "/joint_states", self._joint_cb, 50
            )
        )
        self.streaming_active = False
        self.current_joint_positions: Optional[np.ndarray] = None
        self.last_joint_state_monotonic: Optional[float] = None
        self.filtered_force_z: Optional[float] = None
        self.last_command = np.zeros(6, dtype=float)

    def _joint_cb(self, message: JointState) -> None:
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in VELOCITY_JOINT_NAMES):
            return
        self.current_joint_positions = np.asarray(
            [positions[name] for name in VELOCITY_JOINT_NAMES],
            dtype=float,
        )
        self.last_joint_state_monotonic = time.monotonic()

    def validate_configuration(self) -> None:
        super().validate_configuration()
        if float(self.p("normal_force_n")) <= 0.0:
            raise ValueError(
                "This pilot requires a positive ATI target Fz_R."
            )
        if float(self.p("approach_direction_sign")) != -1.0:
            raise ValueError(
                "Setznest2 approach_direction_sign must remain -1.0 "
                "because -Z_R points toward the surface."
            )
        if not 20.0 <= float(self.p("control_rate_hz")) <= 250.0:
            raise ValueError("control_rate_hz must be in [20, 250] Hz.")
        if not 0.02 <= float(self.p("command_horizon_s")) <= 0.10:
            raise ValueError("command_horizon_s must be in [0.02, 0.10] s.")
        if not 0.0 < float(self.p("force_filter_alpha")) <= 1.0:
            raise ValueError("force_filter_alpha must be in (0, 1].")
        if not 0.0 < float(self.p("force_kp_mps_per_n")) <= 0.001:
            raise ValueError("force_kp_mps_per_n must be in (0, 0.001].")
        if not 0.0 < float(self.p("max_normal_velocity_mps")) <= 0.001:
            raise ValueError(
                "max_normal_velocity_mps must be in (0, 0.001] m/s."
            )
        if not 0.0 < float(self.p("max_joint_velocity_radps")) <= 0.25:
            raise ValueError(
                "max_joint_velocity_radps must be in (0, 0.25] rad/s."
            )
        if float(self.p("radial_speed_mps")) > 0.001:
            raise ValueError("radial_speed_mps must not exceed 0.001 m/s.")
        if float(self.p("max_abs_normal_force_n")) > 6.0:
            raise ValueError("ATI hard force limit must not exceed 6 N.")

        result_path = (
            Path(str(self.p("result_root"))).expanduser()
            / str(self.p("run_name"))
        )
        bag_path = (
            Path(str(self.p("bag_root"))).expanduser()
            / str(self.p("run_name"))
        )
        if result_path.exists():
            raise FileExistsError(f"Result already exists: {result_path}")
        if bool(self.p("record_bag")) and bag_path.exists():
            raise FileExistsError(f"Bag already exists: {bag_path}")

    def wait_interfaces(self) -> None:
        """Wait only for interfaces used by the direct ATI controller."""
        if not self.scaled_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Scaled trajectory action unavailable.")
        for client, name in (
            (self.switch_client, "switch_controller"),
            (self.list_client, "list_controllers"),
        ):
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f"Service unavailable: {name}")

        deadline = time.monotonic() + 10.0
        while (
            self.last_measurement_wrench_monotonic is None
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.last_measurement_wrench_monotonic is None:
            raise RuntimeError(
                "ATI TCP wrench unavailable. No motion was started."
            )
        expected_frame = str(self.p("measurement_wrench_expected_frame"))
        if self.current_measurement_wrench_source_frame != expected_frame:
            raise RuntimeError(
                "ATI wrench frame mismatch: received "
                f"{self.current_measurement_wrench_source_frame!r}, expected "
                f"{expected_frame!r}. No motion was started."
            )

        if bool(self.p("require_robot_program_running")):
            program_deadline = time.monotonic() + 2.0
            while (
                self.robot_program_running is None
                and time.monotonic() < program_deadline
            ):
                rclpy.spin_once(self, timeout_sec=0.05)
            if self.robot_program_running is None:
                self.robot_program_running = self.read_dashboard_program_running()
            if self.robot_program_running is not True:
                raise RuntimeError(
                    "PolyScope External Control program is not running. "
                    "No motion was started."
                )

        joint_deadline = time.monotonic() + 2.0
        while (
            self.current_joint_positions is None
            and time.monotonic() < joint_deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.current_joint_positions is None:
            raise RuntimeError("Joint states unavailable. No motion was started.")

        states = self.controller_states()
        if states.get("scaled_joint_trajectory_controller") != "active":
            raise RuntimeError(
                "scaled_joint_trajectory_controller is not active."
            )

    def publish_joint_velocity(self, values: Sequence[float]) -> None:
        command = np.asarray(values, dtype=float)
        if command.shape != (6,) or not np.all(np.isfinite(command)):
            raise RuntimeError("Invalid joint velocity command.")
        limit = float(self.p("max_joint_velocity_radps"))
        peak = float(np.max(np.abs(command)))
        if peak > limit:
            command *= limit / peak
        if (
            self.current_joint_positions is None
            or self.last_joint_state_monotonic is None
        ):
            raise RuntimeError("Joint state unavailable.")
        age = time.monotonic() - self.last_joint_state_monotonic
        if age > float(self.p("joint_state_timeout_s")):
            raise RuntimeError(f"Joint state stale: age={age:.3f} s.")
        message = Float64MultiArray()
        message.data = [float(value) for value in command]
        self.velocity_publisher.publish(message)
        self.last_command = command

    def publish_hold(self) -> None:
        message = Float64MultiArray()
        message.data = [0.0] * 6
        for _ in range(5):
            self.velocity_publisher.publish(message)
            time.sleep(0.004)
        self.last_command = np.zeros(6, dtype=float)

    def stop_velocity_control(self) -> None:
        if not self.streaming_active:
            return
        self.publish_hold()
        self.streaming_active = False
        self.switch_scaled()

    def switch_velocity(self) -> None:
        self.switch(
            ["forward_velocity_controller"],
            [
                "scaled_joint_trajectory_controller",
                "passthrough_trajectory_controller",
                "force_mode_controller",
            ],
        )

    def switch_scaled(self) -> None:
        self.switch(
            ["scaled_joint_trajectory_controller"],
            [
                "forward_velocity_controller",
                "passthrough_trajectory_controller",
                "force_mode_controller",
            ],
        )

    def geometric_jacobian(self) -> np.ndarray:
        """Return the 6x6 geometric Jacobian expressed in sliding_frame."""
        tcp_position, _ = self.pose_vector()
        tcp = np.asarray(tcp_position, dtype=float)
        jacobian = np.zeros((6, 6), dtype=float)
        for index, child_frame in enumerate(JOINT_CHILD_FRAMES):
            transform = self.tf_buffer.lookup_transform(
                str(self.p("sliding_frame")),
                child_frame,
                Time(),
            ).transform
            joint_position = np.array(
                [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                ],
                dtype=float,
            )
            q = transform.rotation
            axis = np.asarray(
                self.rotate_vector([0.0, 0.0, 1.0], q.x, q.y, q.z, q.w),
                dtype=float,
            )
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-9:
                raise RuntimeError(f"Invalid joint axis for {child_frame}.")
            axis /= axis_norm
            jacobian[:3, index] = np.cross(axis, tcp - joint_position)
            jacobian[3:, index] = axis
        return jacobian

    def cartesian_to_joint_velocity(
        self,
        velocity_x_R: float,
        velocity_z_R: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        jacobian = self.geometric_jacobian()
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        condition = float(
            singular_values[0] / max(singular_values[-1], 1e-12)
        )
        if condition > float(self.p("max_jacobian_condition")):
            raise RuntimeError(
                f"Jacobian condition too high: {condition:.1f}."
            )
        twist = np.array(
            [velocity_x_R, 0.0, velocity_z_R, 0.0, 0.0, 0.0],
            dtype=float,
        )
        damping = float(self.p("jacobian_damping"))
        system = jacobian @ jacobian.T + (damping ** 2) * np.eye(6)
        joint_velocity = jacobian.T @ np.linalg.solve(system, twist)
        achieved_twist = jacobian @ joint_velocity
        return joint_velocity, condition, achieved_twist

    def filtered_ati_force_z(self) -> float:
        self.assert_wrench_safe()
        measured = float(self.current_measurement_force_sliding[2])
        alpha = float(self.p("force_filter_alpha"))
        if self.filtered_force_z is None:
            self.filtered_force_z = measured
        else:
            self.filtered_force_z = (
                alpha * measured + (1.0 - alpha) * self.filtered_force_z
            )
        return self.filtered_force_z

    def normal_control_velocity(self) -> tuple[float, float]:
        force_z = self.filtered_ati_force_z()
        error = float(self.p("normal_force_n")) - force_z
        velocity = (
            float(self.p("approach_direction_sign"))
            * float(self.p("force_kp_mps_per_n"))
            * error
        )
        limit = float(self.p("max_normal_velocity_mps"))
        return clamp(velocity, -limit, limit), force_z

    def command_cartesian_velocity(
        self,
        velocity_x_R: float,
        velocity_z_R: float,
    ) -> tuple[float, np.ndarray]:
        joint_velocity, condition, achieved = self.cartesian_to_joint_velocity(
            velocity_x_R,
            velocity_z_R,
        )
        self.publish_joint_velocity(joint_velocity)
        return condition, achieved

    def assert_pose_safe(
        self,
        start_position: Sequence[float],
        start_orientation: Sequence[float],
        target_x_m: float,
    ) -> tuple[list[float], list[float]]:
        position, orientation = self.pose_vector()
        dx = position[0] - start_position[0]
        dy = position[1] - start_position[1]
        dz = position[2] - start_position[2]
        if abs(dy) > float(self.p("abort_abs_dy_from_slide_start_m")):
            raise RuntimeError(f"Y_R deviation too large: {dy * 1000:.3f} mm.")
        if dx < -float(self.p("abort_radial_undershoot_m")):
            raise RuntimeError(f"Unexpected -X_R motion: {dx * 1000:.3f} mm.")
        if dx > target_x_m + float(self.p("abort_radial_overshoot_m")):
            raise RuntimeError(f"X_R overshoot: {dx * 1000:.3f} mm.")
        if dz < -float(self.p("max_approach_displacement_m")):
            raise RuntimeError(f"Approach travel exceeded: {dz * 1000:.3f} mm.")
        angle = quat_angle(orientation, start_orientation)
        if angle > float(self.p("abort_orientation_rad")):
            raise RuntimeError(
                f"Orientation deviation too large: {math.degrees(angle):.3f} deg."
            )
        return position, orientation

    def control_step(
        self,
        phase: str,
        velocity_x_R: float,
        start_position: Sequence[float],
        start_orientation: Sequence[float],
        target_x_m: float,
        sample_reference: Sequence[float],
        sample_orientation: Sequence[float],
    ) -> tuple[dict, float]:
        # Both ATI and joint states arrive at about 500 Hz. Processing only
        # one ready callback per 125-Hz control cycle can repeatedly select
        # the wrench subscription and make an otherwise healthy joint state
        # appear stale. Drain callbacks until both feedback streams have
        # delivered a fresh sample; retain the independent stale-data aborts.
        rclpy.spin_once(self.joint_feedback_node, timeout_sec=0.01)
        previous_wrench_stamp = self.last_measurement_wrench_monotonic
        callback_deadline = time.monotonic() + 0.006
        while time.monotonic() < callback_deadline:
            rclpy.spin_once(self, timeout_sec=0.001)
            wrench_fresh = (
                self.last_measurement_wrench_monotonic is not None
                and self.last_measurement_wrench_monotonic
                != previous_wrench_stamp
            )
            if wrench_fresh:
                break
        velocity_z_R, force_z = self.normal_control_velocity()
        self.assert_pose_safe(
            start_position,
            start_orientation,
            target_x_m,
        )
        condition, achieved = self.command_cartesian_velocity(
            velocity_x_R,
            velocity_z_R,
        )
        sample = self.append_sample(
            phase,
            sample_reference,
            sample_orientation,
        )
        sample["command_vx_R_mps"] = velocity_x_R
        sample["command_vz_R_mps"] = velocity_z_R
        sample["achieved_vx_R_mps"] = float(achieved[0])
        sample["achieved_vz_R_mps"] = float(achieved[2])
        sample["jacobian_condition"] = condition
        sample["filtered_ati_force_z_R_N"] = force_z
        return sample, force_z

    def wait_period(self, started: float) -> None:
        period = 1.0 / float(self.p("control_rate_hz"))
        remaining = period - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)

    def approach_contact(
        self,
        start_position: Sequence[float],
        start_orientation: Sequence[float],
        target_x_m: float,
    ) -> tuple[list[float], list[float]]:
        deadline = time.monotonic() + float(self.p("contact_timeout_s"))
        threshold = float(self.p("contact_threshold_n"))
        persistence = float(self.p("contact_persistence_s"))
        above_since: Optional[float] = None
        self.publish_event("ati_velocity_approach_started")
        while time.monotonic() < deadline:
            iteration = time.monotonic()
            _, force_z = self.control_step(
                "approach",
                0.0,
                start_position,
                start_orientation,
                target_x_m,
                start_position,
                start_orientation,
            )
            if force_z >= threshold:
                if above_since is None:
                    above_since = time.monotonic()
                elif time.monotonic() - above_since >= persistence:
                    self.publish_event(
                        "ati_contact_detected",
                        ati_force_z_R_n=force_z,
                    )
                    self.publish_hold()
                    return self.pose_vector()
            else:
                above_since = None
            self.wait_period(iteration)
        raise RuntimeError("ATI contact timeout.")

    def controlled_hold(
        self,
        duration_s: float,
        phase: str,
        start_position: Sequence[float],
        start_orientation: Sequence[float],
        target_x_m: float,
        sample_reference: Sequence[float],
        sample_orientation: Sequence[float],
    ) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            iteration = time.monotonic()
            self.control_step(
                phase,
                0.0,
                start_position,
                start_orientation,
                target_x_m,
                sample_reference,
                sample_orientation,
            )
            self.wait_period(iteration)

    def controlled_slide(
        self,
        start_position: Sequence[float],
        start_orientation: Sequence[float],
        baseline_position: Sequence[float],
        baseline_orientation: Sequence[float],
        target_x_m: float,
    ) -> None:
        speed = float(self.p("radial_speed_mps"))
        duration = target_x_m / speed
        deadline = (
            time.monotonic()
            + duration * float(self.p("max_scan_duration_factor"))
            + 1.0
        )
        contact_loss_since: Optional[float] = None
        self.publish_event(
            "ati_velocity_slide_started",
            target_distance_m=target_x_m,
            radial_speed_mps=speed,
        )
        while time.monotonic() < deadline:
            iteration = time.monotonic()
            current_position, _ = self.pose_vector()
            dx = current_position[0] - baseline_position[0]
            remaining = target_x_m - dx
            if remaining <= float(self.p("endpoint_radial_tolerance_m")):
                self.publish_hold()
                self.append_sample(
                    "slide",
                    baseline_position,
                    baseline_orientation,
                )
                self.publish_event("ati_velocity_slide_finished")
                return
            velocity_x = min(speed, max(0.00005, 4.0 * remaining))
            sample, force_z = self.control_step(
                "slide",
                velocity_x,
                start_position,
                start_orientation,
                target_x_m,
                baseline_position,
                baseline_orientation,
            )
            if abs(float(sample["dz_R_m"])) > float(
                self.p("abort_abs_dz_from_baseline_m")
            ):
                raise RuntimeError(
                    "Flat-reference Z_R deviation too large: "
                    f"{float(sample['dz_R_m']) * 1000:.3f} mm."
                )
            if force_z < float(self.p("min_abs_contact_force_n")):
                if contact_loss_since is None:
                    contact_loss_since = time.monotonic()
                elif (
                    time.monotonic() - contact_loss_since
                    >= float(self.p("contact_loss_persistence_s"))
                ):
                    raise RuntimeError(
                        f"ATI contact lost during slide: Fz_R={force_z:.3f} N."
                    )
            else:
                contact_loss_since = None
            self.wait_period(iteration)
        raise RuntimeError("ATI velocity slide timeout.")

    def release_contact(
        self,
        start_position: Sequence[float],
        start_orientation: Sequence[float],
        target_x_m: float,
    ) -> None:
        release_start, _ = self.pose_vector()
        released_z: Optional[float] = None
        max_distance = float(self.p("max_release_displacement_m"))
        clearance = float(self.p("release_clearance_m"))
        speed = float(self.p("release_velocity_mps"))
        threshold = float(self.p("release_force_threshold_n"))
        self.publish_event("ati_velocity_release_started")
        while True:
            iteration = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.0)
            force_z = self.filtered_ati_force_z()
            position, _ = self.assert_pose_safe(
                start_position,
                start_orientation,
                target_x_m,
            )
            travelled = position[2] - release_start[2]
            if travelled > max_distance:
                raise RuntimeError(
                    "Release displacement exceeded before contact-free state: "
                    f"{travelled * 1000:.3f} mm."
                )
            if force_z <= threshold and released_z is None:
                released_z = position[2]
            if released_z is not None and position[2] - released_z >= clearance:
                self.publish_hold()
                self.publish_event(
                    "ati_velocity_release_finished",
                    release_distance_m=travelled,
                )
                return
            self.command_cartesian_velocity(0.0, speed)
            self.append_sample(
                "release",
                release_start,
                start_orientation,
            )
            self.wait_period(iteration)

    def save_results(
        self,
        status: str,
        metrics: Optional[dict],
        error: Optional[str],
    ) -> Path:
        root = Path(str(self.p("result_root"))).expanduser()
        run_dir = root / str(self.p("run_name"))
        run_dir.mkdir(parents=True, exist_ok=False)
        csv_path = run_dir / "flat_reference_time_series.csv"
        fields: list[str] = []
        for row in self.samples:
            for field in row:
                if field not in fields:
                    fields.append(field)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            if fields:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.samples)

        summary = {
            "status": status,
            "error": error,
            "run_name": str(self.p("run_name")),
            "control_architecture": "direct_ati_short_horizon_trajectory",
            "control_wrench_topic": str(self.p("measurement_wrench_topic")),
            "authoritative_force_source": "ati_netft_tcp_readft",
            "measurement_wrench_expected_frame": str(
                self.p("measurement_wrench_expected_frame")
            ),
            "ur_force_mode_used": False,
            "trajectory_controller": "scaled_joint_trajectory_controller",
            "command_horizon_s": float(self.p("command_horizon_s")),
            "approach_direction": "-Z_R",
            "automatic_return_to_start": False,
            "normal_force_n": float(self.p("normal_force_n")),
            "radial_speed_mps": float(self.p("radial_speed_mps")),
            "target_distance_m": (
                float(self.active_offsets[-1]) if self.active_offsets else None
            ),
            "force_kp_mps_per_n": float(self.p("force_kp_mps_per_n")),
            "max_normal_velocity_mps": float(
                self.p("max_normal_velocity_mps")
            ),
            "max_joint_velocity_radps": float(
                self.p("max_joint_velocity_radps")
            ),
            "max_abs_normal_force_n": float(
                self.p("max_abs_normal_force_n")
            ),
            "metrics": metrics,
        }
        summary_path = run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.get_logger().info(
            f"Saved ATI Phase-B results: {csv_path}, {summary_path}"
        )
        return run_dir

    def run(self) -> None:
        self.validate_configuration()
        self.wait_interfaces()
        self.load_waypoints()
        self.validate_ati_bias()
        target_x_m = float(self.active_offsets[-1])

        input(
            "\nPHASE B ATI – STARTPOSE APPROVAL\n"
            "Press Enter to move to the validated Phase-A start pose, or "
            "Ctrl+C to abort..."
        )
        self.switch_scaled()
        self.move_to_start()
        start_position, start_orientation = self.validate_start()

        input(
            "\nPHASE B ATI – MOTION APPROVAL\n"
            f"Direct ATI control will approach in -Z_R to "
            f"Fz_R={float(self.p('normal_force_n')):.1f} N, slide "
            f"{target_x_m * 1000:.1f} mm in +X_R at "
            f"{float(self.p('radial_speed_mps')) * 1000:.1f} mm/s, then "
            "release in +Z_R. UR Force Mode is NOT used.\n"
            f"Limits: |Fz_R| <= {float(self.p('max_abs_normal_force_n')):.1f} N, "
            f"|vz_R| <= {float(self.p('max_normal_velocity_mps')) * 1000:.1f} mm/s.\n"
            "Confirm the complete path is flat and free, speed slider is "
            "100%, and emergency stop is reachable. Press Enter to start, "
            "or Ctrl+C to abort..."
        )

        bag_path: Optional[Path] = None
        metrics: Optional[dict] = None
        status = "FAILED"
        error_text: Optional[str] = None
        try:
            bag_path = self.start_bag()
            # Bag discovery can occupy the single-threaded executor long
            # enough that the most recently processed joint sample is older
            # than the control watchdog. Require new joint and ATI samples
            # before issuing the very first trajectory command.
            feedback_deadline = time.monotonic() + 1.0
            while time.monotonic() < feedback_deadline:
                rclpy.spin_once(
                    self.joint_feedback_node, timeout_sec=0.01
                )
                rclpy.spin_once(self, timeout_sec=0.01)
                now = time.monotonic()
                joint_fresh = (
                    self.last_joint_state_monotonic is not None
                    and now - self.last_joint_state_monotonic < 0.02
                )
                wrench_fresh = (
                    self.last_measurement_wrench_monotonic is not None
                    and now - self.last_measurement_wrench_monotonic < 0.02
                )
                if joint_fresh and wrench_fresh:
                    break
            else:
                raise RuntimeError(
                    "Fresh ATI and joint feedback unavailable after bag start. "
                    "No contact motion was started."
                )
            self.filtered_force_z = None
            self.switch_velocity()
            # Mark the switched controller for guaranteed zero-command and
            # switch-back cleanup, even if the post-switch feedback check fails.
            self.streaming_active = True
            # The strict controller switch deliberately waits 0.5 s. Refresh
            # both feedback streams again after that wait and before enabling
            # velocity output.
            post_switch_deadline = time.monotonic() + 1.0
            while time.monotonic() < post_switch_deadline:
                rclpy.spin_once(
                    self.joint_feedback_node, timeout_sec=0.01
                )
                rclpy.spin_once(self, timeout_sec=0.01)
                now = time.monotonic()
                if (
                    self.last_joint_state_monotonic is not None
                    and now - self.last_joint_state_monotonic < 0.02
                    and self.last_measurement_wrench_monotonic is not None
                    and now - self.last_measurement_wrench_monotonic < 0.02
                ):
                    break
            else:
                raise RuntimeError(
                    "Fresh feedback unavailable after velocity-controller "
                    "activation. No velocity command was sent."
                )
            contact_position, contact_orientation = self.approach_contact(
                start_position,
                start_orientation,
                target_x_m,
            )
            self.controlled_hold(
                float(self.p("stabilisation_after_contact_s")),
                "stabilisation",
                start_position,
                start_orientation,
                target_x_m,
                contact_position,
                contact_orientation,
            )
            baseline_position, baseline_orientation = self.pose_vector()
            self.run_t0_monotonic = time.monotonic()
            self.samples.clear()
            self.publish_event("baseline_started")
            self.controlled_hold(
                float(self.p("baseline_duration_s")),
                "baseline",
                start_position,
                start_orientation,
                target_x_m,
                baseline_position,
                baseline_orientation,
            )
            self.publish_event("baseline_finished")
            self.controlled_slide(
                start_position,
                start_orientation,
                baseline_position,
                baseline_orientation,
                target_x_m,
            )
            metrics = self.compute_metrics()
            self.publish_event("metrics_computed", **metrics)
            self.release_contact(
                start_position,
                start_orientation,
                target_x_m,
            )
            self.stop_velocity_control()
            status = (
                "COMPLETED_PROVISIONALLY_VALID"
                if metrics["provisional_reference_valid"]
                else "COMPLETED_REVIEW_REQUIRED"
            )
            self.publish_event("phase_b_ati_completed", status=status)
        except Exception as exc:
            status = "INVALID_OR_ABORTED"
            error_text = str(exc)
            try:
                self.publish_event("phase_b_ati_aborted", error=error_text)
            except Exception:
                pass
            raise
        finally:
            try:
                self.stop_velocity_control()
            except Exception as exc:
                self.get_logger().error(
                    f"Velocity-controller cleanup failed: {exc}"
                )
            self.stop_bag()
            self.save_results(status, metrics, error_text)
            if bag_path is not None:
                self.get_logger().info(f"Bag finalized: {bag_path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PhaseBAtiVelocity()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warning("Interrupted by user.")
    except Exception as exc:
        node.get_logger().error(str(exc))
        raise
    finally:
        try:
            node.stop_velocity_control()
        except Exception:
            pass
        try:
            node.stop_bag()
        except Exception:
            pass
        node.joint_feedback_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

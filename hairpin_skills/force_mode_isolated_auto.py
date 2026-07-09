#!/usr/bin/env python3
"""
Fully automated isolated force-mode experiment for the UR3e.

Sequence
--------
1. Move with scaled_joint_trajectory_controller to a calibrated pose
   1–2 mm above the surface.
2. Zero the F/T sensor without contact.
3. Start rosbag2 recording.
4. Switch to passthrough_trajectory_controller + force_mode_controller.
5. Start force mode with a normal force in sliding_frame.
6. Detect first contact from |Fz_sliding|.
7. Hold contact for the configured time.
8. Stop force mode.
9. Switch back to scaled_joint_trajectory_controller.
10. Move back to the calibrated approach/retreat pose.
11. Stop the bag.

The robot stays in Remote Mode for the complete sequence.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import rclpy
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from ur_msgs.srv import SetForceMode

from hairpin_skills.test_sliding_training_day2 import SlidingTrainingNode


class ForceModeIsolatedAuto(SlidingTrainingNode):
    def __init__(self) -> None:
        super().__init__()
        self._node_name = "force_mode_isolated_auto"

        self.declare_parameter("task_frame", "sliding_frame")
        self.declare_parameter("bag_name", "force_mode_isolated_auto")
        self.declare_parameter(
            "bag_root",
            str(Path.home() / "masterarbeit" / "bags"),
        )

        # Existing taught vectors from test_sliding_training_day2.py.
        # Verify that approach_joints is actually 1–2 mm above the surface.
        self.declare_parameter(
            "approach_joints",
            [
                1.3601649443255823,
                -1.069422797565796,
                0.9133870601654053,
                -2.0465508900084437,
                -1.5711777845965784,
                0.6645072102546692,
            ],
        )
        self.declare_parameter(
            "home_joints",
            [0.128, -0.700, 0.699, -1.038, -1.531, -0.356],
        )

        self.declare_parameter("approach_duration_s", 4.0)
        self.declare_parameter("retreat_duration_s", 2.0)
        self.declare_parameter("return_home", False)
        self.declare_parameter("home_duration_s", 4.0)

        self.declare_parameter("force_z_n", -1.0)
        self.declare_parameter("hold_time_s", 2.0)
        self.declare_parameter("contact_threshold_n", 0.5)
        self.declare_parameter("contact_timeout_s", 4.0)
        self.declare_parameter("speed_limit_z_mps", 0.005)
        self.declare_parameter("deviation_z_m", 0.005)
        self.declare_parameter("damping_factor", 0.5)
        self.declare_parameter("gain_scaling", 0.3)

        self.zero_client = self.create_client(
            Trigger,
            "/io_and_status_controller/zero_ftsensor",
        )
        self.zero_client.wait_for_service()

        self.bag_process: Optional[subprocess.Popen] = None
        self.force_mode_running = False

    def p(self, name: str):
        return self.get_parameter(name).value

    def wrench_in_frame(
        self,
        target_frame: str,
    ) -> Optional[
        Tuple[
            Tuple[float, float, float],
            Tuple[float, float, float],
        ]
    ]:
        source_frame = self.current_wrench_frame or "tool0"

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"Cannot transform wrench {source_frame} -> "
                f"{target_frame}: {exc}"
            )
            return None

        q = transform.transform.rotation
        force = self._rotate_vector_by_quaternion(
            self.current_wrench[:3],
            q.x,
            q.y,
            q.z,
            q.w,
        )
        torque = self._rotate_vector_by_quaternion(
            self.current_wrench[3:],
            q.x,
            q.y,
            q.z,
            q.w,
        )
        return force, torque

    def verify_sliding_frame(self) -> None:
        target_frame = str(self.p("task_frame"))
        source_frame = self.current_wrench_frame or "tool0"

        try:
            self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
            )
        except TransformException as exc:
            raise RuntimeError(
                f"No TF chain from {source_frame} to {target_frame}: {exc}"
            ) from exc

        self.get_logger().info(
            f"TF chain available: {source_frame} -> {target_frame}"
        )

    def zero_sensor(self) -> None:
        future = self.zero_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if result is None or not result.success:
            message = result.message if result is not None else "no response"
            raise RuntimeError(f"F/T zeroing failed: {message}")

        self.get_logger().info("F/T sensor zeroed.")
        time.sleep(1.0)

    def start_bag(self) -> Path:
        bag_root = Path(str(self.p("bag_root"))).expanduser()
        bag_root.mkdir(parents=True, exist_ok=True)
        bag_path = bag_root / str(self.p("bag_name"))

        if bag_path.exists():
            raise FileExistsError(f"Bag already exists: {bag_path}")

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
            str(bag_path),
        ]

        self.bag_process = subprocess.Popen(
            command,
            start_new_session=True,
        )
        time.sleep(2.0)

        if self.bag_process.poll() is not None:
            raise RuntimeError("ros2 bag record terminated unexpectedly.")

        self.get_logger().info(f"Bag recording started: {bag_path}")
        return bag_path

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
                try:
                    os.killpg(self.bag_process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self.bag_process.wait(timeout=5.0)

        self.bag_process = None
        self.get_logger().info("Bag recording stopped.")

    def start_force_mode_sliding_frame(self) -> None:
        request = SetForceMode.Request()
        request.task_frame.header.frame_id = str(self.p("task_frame"))
        request.task_frame.pose.orientation.w = 1.0

        request.selection_vector_x = False
        request.selection_vector_y = False
        request.selection_vector_z = True
        request.selection_vector_rx = False
        request.selection_vector_ry = False
        request.selection_vector_rz = False

        request.wrench.force.z = float(self.p("force_z_n"))
        request.type = SetForceMode.Request.NO_TRANSFORM

        request.speed_limits.linear.z = float(
            self.p("speed_limit_z_mps")
        )

        request.deviation_limits = [
            0.002,
            0.002,
            float(self.p("deviation_z_m")),
            0.05,
            0.05,
            0.05,
        ]
        request.damping_factor = float(self.p("damping_factor"))
        request.gain_scaling = float(self.p("gain_scaling"))

        future = self.force_mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if result is None or not result.success:
            raise RuntimeError("Could not start force mode.")

        self.force_mode_running = True
        self.get_logger().info(
            f"Force mode started: Fz={self.p('force_z_n')} N "
            f"in {self.p('task_frame')}"
        )

    def stop_force_mode_safe(self) -> None:
        if not self.force_mode_running:
            return
        self.stop_force_mode()
        self.force_mode_running = False
        self.get_logger().info("Force mode stopped.")

    def wait_for_contact(self) -> float:
        target_frame = str(self.p("task_frame"))
        threshold = abs(float(self.p("contact_threshold_n")))
        timeout_s = float(self.p("contact_timeout_s"))
        started = time.monotonic()

        while time.monotonic() - started < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.01)
            transformed = self.wrench_in_frame(target_frame)
            if transformed is None:
                continue

            force, _ = transformed
            fz = force[2]

            if abs(fz) >= threshold:
                elapsed = time.monotonic() - started
                self.get_logger().info(
                    f"Contact detected after {elapsed:.3f} s: "
                    f"Fz_{target_frame}={fz:.3f} N"
                )
                return elapsed

        raise TimeoutError(
            f"No contact detected within {timeout_s:.3f} s."
        )

    @staticmethod
    def validate_joint_vector(
        values: Sequence[float],
        name: str,
    ) -> list[float]:
        if len(values) != 6:
            raise ValueError(f"{name} must contain six joint values.")
        converted = [float(value) for value in values]
        if not all(math.isfinite(value) for value in converted):
            raise ValueError(f"{name} contains non-finite values.")
        return converted

    def run_experiment(self) -> None:
        approach = self.validate_joint_vector(
            self.p("approach_joints"),
            "approach_joints",
        )
        home = self.validate_joint_vector(
            self.p("home_joints"),
            "home_joints",
        )

        if not self.wait_for_measurements(timeout_s=5.0):
            raise RuntimeError("Required measurements are unavailable.")

        self.verify_sliding_frame()

        # Approach in position control.
        if not self.switch_to_scaled():
            raise RuntimeError("Could not activate scaled controller.")

        self.get_logger().info(
            "Moving to calibrated pose 1–2 mm above the surface."
        )
        if not self.move_scaled(
            approach,
            float(self.p("approach_duration_s")),
        ):
            raise RuntimeError("Approach movement failed.")

        # Refresh signals and verify TF at the reached pose.
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.02)
        self.verify_sliding_frame()

        # Must be contact-free here.
        self.zero_sensor()
        bag_path = self.start_bag()

        try:
            if not self.switch_to_passthrough_with_force():
                raise RuntimeError(
                    "Could not activate passthrough + force controllers."
                )

            self.start_force_mode_sliding_frame()
            self.wait_for_contact()

            hold_time = float(self.p("hold_time_s"))
            self.get_logger().info(
                f"Holding contact for {hold_time:.3f} s."
            )
            hold_started = time.monotonic()

            while time.monotonic() - hold_started < hold_time:
                rclpy.spin_once(self, timeout_sec=0.01)

            self.stop_force_mode_safe()

            if not self.switch_to_scaled():
                raise RuntimeError(
                    "Could not restore scaled controller."
                )

            self.get_logger().info(
                "Retreating to calibrated approach pose."
            )
            if not self.move_scaled(
                approach,
                float(self.p("retreat_duration_s")),
            ):
                raise RuntimeError("Retreat movement failed.")

            if bool(self.p("return_home")):
                self.get_logger().info("Returning home.")
                if not self.move_scaled(
                    home,
                    float(self.p("home_duration_s")),
                ):
                    raise RuntimeError("Home movement failed.")

            self.get_logger().info(
                f"Experiment completed successfully: {bag_path}"
            )

        finally:
            try:
                self.stop_force_mode_safe()
            finally:
                try:
                    self.switch_to_scaled()
                finally:
                    self.stop_bag()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ForceModeIsolatedAuto()

    try:
        node.run_experiment()
    except KeyboardInterrupt:
        node.get_logger().warning("Experiment interrupted by user.")
    except Exception as exc:
        node.get_logger().error(f"Experiment failed: {exc}")
        raise
    finally:
        try:
            node.stop_force_mode_safe()
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

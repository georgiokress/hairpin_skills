#!/usr/bin/env python3
"""Bounded, contact-free diagnostic for the UR Force Mode F/T source.

This node does not move to a start pose and never executes a sliding
trajectory.  After an explicit terminal approval it requests zero normal
force for a short, bounded interval while logging the internal UR wrench,
the independent ATI TCP wrench, and TCP displacement.  It cannot validate
Phase B or Phase C.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Optional

import rclpy

from hairpin_skills.phase_b_flat_reference import PhaseBFlatReference


class PhaseBForceSourceDiagnostic(PhaseBFlatReference):
    """Run a zero-wrench diagnostic without approach, contact, or sliding."""

    def __init__(self) -> None:
        super().__init__("phase_b_force_source_diagnostic")
        self.declare_parameter("minimum_start_z_R_m", 0.004)
        self.declare_parameter("diagnostic_duration_s", 0.5)
        self.declare_parameter("max_diagnostic_displacement_m", 0.00025)
        self.diagnostic_controller_switched = False

    def validate_diagnostic_configuration(self) -> None:
        if not bool(self.p("phase_a_validated")):
            raise RuntimeError("phase_a_validated is false.")
        requested_force = float(self.p("normal_force_n"))
        if abs(requested_force) > 2.0 + 1e-12:
            raise ValueError(
                "Bounded diagnostic |normal_force_n| must not exceed 2 N."
            )

        duration = float(self.p("diagnostic_duration_s"))
        speed = float(self.p("speed_limit_z_mps"))
        displacement_limit = float(self.p("max_diagnostic_displacement_m"))
        if not 0.0 < duration <= 0.5:
            raise ValueError("diagnostic_duration_s must be in (0, 0.5] s.")
        if not 0.0 < speed <= 0.0002:
            raise ValueError("speed_limit_z_mps must be in (0, 0.0002] m/s.")
        if not 0.0 < displacement_limit <= 0.00025:
            raise ValueError(
                "max_diagnostic_displacement_m must be in (0, 0.00025] m."
            )
        if speed * duration > 0.00010 + 1e-12:
            raise ValueError(
                "Nominal diagnostic travel must not exceed 0.10 mm."
            )

    def validate_output_paths(self) -> None:
        run_name = str(self.p("run_name"))
        result_path = Path(str(self.p("result_root"))).expanduser() / run_name
        bag_path = Path(str(self.p("bag_root"))).expanduser() / run_name
        if result_path.exists():
            raise FileExistsError(f"Diagnostic result already exists: {result_path}")
        if bool(self.p("record_bag")) and bag_path.exists():
            raise FileExistsError(f"Diagnostic bag already exists: {bag_path}")

    def validate_contact_free_pose(self) -> tuple[list[float], list[float]]:
        position, orientation = self.pose_vector()
        minimum_z = float(self.p("minimum_start_z_R_m"))
        self.get_logger().info(
            "Diagnostic pose validation: "
            f"p_R={[round(value, 6) for value in position]}, "
            f"required z_R >= {minimum_z * 1000:.3f} mm."
        )
        if position[2] < minimum_z:
            raise RuntimeError(
                "Current pose is not sufficiently far from the surface: "
                f"z_R={position[2] * 1000:.3f} mm, required "
                f">={minimum_z * 1000:.3f} mm. No motion was started."
            )
        return position, orientation

    def diagnostic_metrics(self) -> Optional[dict]:
        if not self.samples:
            return None
        dx = [float(row["dx_R_m"]) for row in self.samples]
        dy = [float(row["dy_R_m"]) for row in self.samples]
        dz = [float(row["dz_R_m"]) for row in self.samples]
        displacement = [
            math.sqrt(x_value * x_value + y_value * y_value + z_value * z_value)
            for x_value, y_value, z_value in zip(dx, dy, dz)
        ]
        internal_reported_z = [
            float(row["force_z_source_N"]) for row in self.samples
        ]
        ati_z_sliding = [
            float(row["measurement_force_z_R_N"]) for row in self.samples
        ]
        return {
            "sample_count": len(self.samples),
            "sampled_duration_s": float(self.samples[-1]["time_s"]),
            "delta_R_m": [dx[-1], dy[-1], dz[-1]],
            "max_abs_dx_R_m": max(abs(value) for value in dx),
            "max_abs_dy_R_m": max(abs(value) for value in dy),
            "max_abs_dz_R_m": max(abs(value) for value in dz),
            "max_translation_m": max(displacement),
            "internal_ur_topic_force_z_reported_mean_n": statistics.fmean(
                internal_reported_z
            ),
            "ati_force_z_R_mean_n": statistics.fmean(ati_z_sliding),
            "ati_force_z_R_abs_max_n": max(abs(value) for value in ati_z_sliding),
        }

    def save_diagnostic_results(
        self,
        status: str,
        error: Optional[str],
        metrics: Optional[dict],
    ) -> Path:
        root = Path(str(self.p("result_root"))).expanduser()
        run_dir = root / str(self.p("run_name"))
        run_dir.mkdir(parents=True, exist_ok=False)

        csv_path = run_dir / "zero_force_diagnostic_time_series.csv"
        if self.samples:
            fields = list(self.samples[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.samples)
        else:
            csv_path.touch()

        summary = {
            "status": status,
            "error": error,
            "diagnostic_only": True,
            "phase_b_validated": False,
            "run_name": str(self.p("run_name")),
            "normal_force_n": float(self.p("normal_force_n")),
            "diagnostic_duration_s": float(self.p("diagnostic_duration_s")),
            "speed_limit_z_mps": float(self.p("speed_limit_z_mps")),
            "max_diagnostic_displacement_m": float(
                self.p("max_diagnostic_displacement_m")
            ),
            "measurement_wrench_topic": str(self.p("measurement_wrench_topic")),
            "measurement_wrench_expected_frame": str(
                self.p("measurement_wrench_expected_frame")
            ),
            "metrics": metrics,
        }
        json_path = run_dir / "summary.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self.get_logger().info(
            f"Saved diagnostic results: {csv_path}, {json_path}"
        )
        return run_dir

    def run(self) -> None:
        self.validate_diagnostic_configuration()
        self.validate_output_paths()
        self.wait_interfaces()
        self.validate_ati_bias()
        reference_position, reference_orientation = (
            self.validate_contact_free_pose()
        )

        input(
            "\nPHASE B - BOUNDED FORCE-DIRECTION DIAGNOSTIC APPROVAL\n"
            "No approach and no sliding trajectory will be executed.\n"
            f"Current z_R={reference_position[2] * 1000:.3f} mm. "
            f"The request is {float(self.p('normal_force_n')):.1f} N in z_R "
            "for at most "
            f"{float(self.p('diagnostic_duration_s')):.3f} s at "
            f"{float(self.p('speed_limit_z_mps')) * 1000:.3f} mm/s.\n"
            "The geometric stop limit is "
            f"{float(self.p('max_diagnostic_displacement_m')) * 1000:.3f} mm. "
            "Keep the speed slider at 100% and emergency stop reachable.\n"
            "Press Enter to start the diagnostic Force Mode, or Ctrl+C to abort..."
        )

        bag_path: Optional[Path] = None
        status = "FAILED"
        error_text: Optional[str] = None
        metrics: Optional[dict] = None
        try:
            bag_path = self.start_bag()
            self.switch_force()
            self.diagnostic_controller_switched = True

            request_started = time.monotonic()
            self.run_t0_monotonic = request_started
            self.samples.clear()
            self.publish_event(
                "zero_force_diagnostic_started",
                duration_s=float(self.p("diagnostic_duration_s")),
                speed_limit_z_mps=float(self.p("speed_limit_z_mps")),
                displacement_limit_m=float(
                    self.p("max_diagnostic_displacement_m")
                ),
            )
            self.start_force_mode()

            deadline = request_started + float(self.p("diagnostic_duration_s"))
            sample_period = float(self.p("sample_period_s"))
            next_sample = time.monotonic()
            displacement_limit = float(
                self.p("max_diagnostic_displacement_m")
            )
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=min(0.005, sample_period))
                self.assert_wrench_safe()
                now = time.monotonic()
                if now < next_sample:
                    continue
                sample = self.append_sample(
                    "zero_force_diagnostic",
                    reference_position,
                    reference_orientation,
                )
                translation = math.sqrt(
                    float(sample["dx_R_m"]) ** 2
                    + float(sample["dy_R_m"]) ** 2
                    + float(sample["dz_R_m"]) ** 2
                )
                if translation > displacement_limit:
                    raise RuntimeError(
                        "Diagnostic displacement limit exceeded: "
                        f"{translation * 1000:.3f} mm > "
                        f"{displacement_limit * 1000:.3f} mm."
                    )
                next_sample = now + sample_period

            self.stop_force_mode()
            self.append_sample(
                "zero_force_diagnostic_final",
                reference_position,
                reference_orientation,
            )
            metrics = self.diagnostic_metrics()
            status = "COMPLETED_DIAGNOSTIC_ONLY"
            self.publish_event("zero_force_diagnostic_completed", **(metrics or {}))
        except Exception as exc:
            status = "INVALID_OR_ABORTED"
            error_text = str(exc)
            try:
                self.publish_event("zero_force_diagnostic_aborted", error=error_text)
            except Exception:
                pass
            raise
        finally:
            try:
                self.stop_force_mode()
            except Exception as exc:
                self.get_logger().error(f"Force-mode cleanup failed: {exc}")
            if self.diagnostic_controller_switched:
                try:
                    self.switch_scaled()
                except Exception as exc:
                    self.get_logger().error(f"Controller cleanup failed: {exc}")
            self.stop_bag()
            metrics = metrics or self.diagnostic_metrics()
            self.save_diagnostic_results(status, error_text, metrics)
            if bag_path is not None:
                self.get_logger().info(f"Diagnostic bag finalized: {bag_path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PhaseBForceSourceDiagnostic()
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
        if node.diagnostic_controller_switched:
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

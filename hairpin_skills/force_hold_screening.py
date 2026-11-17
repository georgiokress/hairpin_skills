#!/usr/bin/env python3
"""Static Phase-B force-hold screening without tangential sliding.

This is a qualification-only acquisition path.  It reuses the validated
internal-UR force-control path and the passive ATI reference checks from the
Phase-B SPI collector, but it never sends the calibrated sliding endpoint.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import rclpy

from hairpin_skills.phase_b_spi_data_collection import (
    INTERNAL_TARE_SERVICE,
    INTERNAL_WRENCH_TOPIC,
    PASSIVE_ATI_TOPIC,
    PhaseBSPIDataCollectionNode,
    sha256_file,
    sha256_files,
)


RUN_ID_PATTERN = re.compile(
    r"^pb_spi_s4_setznest2_intctrl_atipass_hold_"
    r"fn[0-9]{3}_boundscreen_r[0-9]{3}$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_gauge_mm(text: str) -> Optional[float]:
    value = text.strip().replace(",", ".")
    if not value or value.upper() == "NA":
        return None
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("Gauge value must be finite or NA.")
    return converted


class ForceHoldScreeningNode(PhaseBSPIDataCollectionNode):
    """Run one bounded normal-force hold at the qualified start location."""

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("hold_time_s", 5.0)
        self.declare_parameter("max_screening_setpoint_n", 3.0)
        self.declare_parameter("settling_exclusion_s", 0.5)
        self.declare_parameter("tcp_normal_drift_abort_m", 0.001)
        self.declare_parameter("tcp_cross_drift_abort_m", 0.0005)
        self.declare_parameter("gauge_resolution_mm", 0.05)
        self.declare_parameter("gauge_slip_threshold_mm", 0.10)
        self.declare_parameter("screening_config_path", str(Path(__file__)))

        self.hold_samples: list[dict[str, float]] = []
        self.hold_summary: Optional[dict[str, Any]] = None
        self.approach_position_xyz_R_m: Optional[list[float]] = None
        self.contact_position_xyz_R_m: Optional[list[float]] = None
        self.return_position_xyz_R_m: Optional[list[float]] = None
        self.gauge_before_mm: Optional[float] = None
        self.gauge_after_mm: Optional[float] = None
        self.operator_slip_observation: Optional[str] = None

    def validate_screening_configuration(self) -> None:
        errors: list[str] = []
        run_id = str(self.p("bag_name"))
        force_magnitude = abs(float(self.p("normal_force_n")))
        hold_time = float(self.p("hold_time_s"))
        force_abort = float(self.p("max_abs_normal_force_n"))

        if not RUN_ID_PATTERN.fullmatch(run_id):
            errors.append(f"invalid force-screening run_id: {run_id}")
        force_token = f"fn{round(force_magnitude * 10):03d}"
        if force_token not in run_id:
            errors.append(f"run_id must contain force token {force_token}")
        if str(self.p("purpose")) != "force_bounds_screening":
            errors.append("purpose must be force_bounds_screening")
        if bool(self.p("publish_sliding_frame")):
            errors.append("publish_sliding_frame must be false")
        if str(self.p("wrench_topic")) != INTERNAL_WRENCH_TOPIC:
            errors.append(f"wrench_topic must be {INTERNAL_WRENCH_TOPIC}")
        if str(self.p("ati_passive_topic")) != PASSIVE_ATI_TOPIC:
            errors.append(f"ati_passive_topic must be {PASSIVE_ATI_TOPIC}")
        if str(self.p("tare_service")) != INTERNAL_TARE_SERVICE:
            errors.append(f"tare_service must be {INTERNAL_TARE_SERVICE}")
        if float(self.p("normal_force_n")) >= 0.0:
            errors.append("normal_force_n must remain negative")
        if force_magnitude <= 0.0:
            errors.append("normal force magnitude must be positive")
        if force_magnitude > float(self.p("max_screening_setpoint_n")):
            errors.append("force setpoint exceeds the approved screening limit")
        if not math.isclose(hold_time, 5.0, abs_tol=1.0e-12):
            errors.append("hold_time_s must equal the approved 5.0 s")
        if not 0.0 < float(self.p("contact_threshold_n")) < force_magnitude:
            errors.append("contact threshold must lie between zero and setpoint")
        if not force_magnitude < force_abort <= 17.2:
            errors.append(
                "measured-force abort must exceed the setpoint and be <= 17.2 N"
            )
        if float(self.p("speed_limit_z_mps")) > 0.001:
            errors.append("normal speed limit must remain <= 1 mm/s")
        if float(self.p("tcp_normal_drift_abort_m")) > 0.001:
            errors.append("TCP normal-drift abort must remain <= 1 mm")
        if float(self.p("tcp_cross_drift_abort_m")) > 0.0005:
            errors.append("TCP cross-drift abort must remain <= 0.5 mm")
        if not isinstance(self.p("collection_dry_run"), bool):
            errors.append("collection_dry_run must be boolean")
        if not isinstance(self.p("run_authorized"), bool):
            errors.append("run_authorized must be boolean")
        if (
            not bool(self.p("collection_dry_run"))
            and not bool(self.p("run_authorized"))
        ):
            errors.append("physical screening requires run_authorized=true")

        paths = {
            "validated collection config": Path(
                str(self.p("collection_config_path"))
            ),
            "screening config": Path(str(self.p("screening_config_path"))),
            "bag QoS config": Path(str(self.p("bag_qos_overrides_path"))),
        }
        for label, path in paths.items():
            if not path.is_file():
                errors.append(f"{label} missing: {path}")

        bag_path = (
            Path(str(self.p("bag_root"))).expanduser()
            / str(self.p("bag_name"))
        )
        if bag_path.exists():
            errors.append(f"bag path already exists: {bag_path}")

        if errors:
            raise RuntimeError(
                "Force-hold screening configuration rejected:\n- "
                + "\n- ".join(errors)
            )

    def hold_contact_screening(self) -> None:
        hold_time = float(self.p("hold_time_s"))
        settling = float(self.p("settling_exclusion_s"))
        normal_abort = float(self.p("tcp_normal_drift_abort_m"))
        cross_abort = float(self.p("tcp_cross_drift_abort_m"))

        self.contact_position_xyz_R_m, _ = self.pose_vector()
        started = time.monotonic()
        stable_reference: Optional[list[float]] = None
        self.publish_event(
            "force_hold_started",
            target_duration_s=hold_time,
            contact_position_xyz_R_m=self.contact_position_xyz_R_m,
        )

        while True:
            elapsed = time.monotonic() - started
            if elapsed >= hold_time:
                break
            rclpy.spin_once(self, timeout_sec=0.005)
            self.assert_force_safe()
            position, _ = self.pose_vector()
            force = float(self.current_normal_force_n)
            self.hold_samples.append(
                {
                    "elapsed_s": elapsed,
                    "x_R_m": position[0],
                    "y_R_m": position[1],
                    "z_R_m": position[2],
                    "internal_fz_R_n": force,
                }
            )

            if elapsed >= settling:
                if stable_reference is None:
                    stable_reference = list(position)
                dz = position[2] - stable_reference[2]
                dx = position[0] - stable_reference[0]
                dy = position[1] - stable_reference[1]
                if abs(dz) > normal_abort:
                    raise RuntimeError(
                        "TCP normal drift guard exceeded during hold: "
                        f"dz_R={dz * 1000.0:.3f} mm"
                    )
                if max(abs(dx), abs(dy)) > cross_abort:
                    raise RuntimeError(
                        "TCP cross-axis drift guard exceeded during hold: "
                        f"dx_R={dx * 1000.0:.3f} mm, "
                        f"dy_R={dy * 1000.0:.3f} mm"
                    )

        if not self.hold_samples:
            raise RuntimeError("No TCP/force samples collected during hold.")
        self.hold_summary = self.summarise_hold_samples()
        self.publish_event("force_hold_finished", summary=self.hold_summary)

    def summarise_hold_samples(self) -> dict[str, Any]:
        settling = float(self.p("settling_exclusion_s"))
        stable = [
            row for row in self.hold_samples
            if row["elapsed_s"] >= settling
        ]
        if len(stable) < 2:
            raise RuntimeError("Too few post-settling hold samples.")

        window_s = 0.25
        first_t = stable[0]["elapsed_s"]
        last_t = stable[-1]["elapsed_s"]
        first = [row for row in stable if row["elapsed_s"] <= first_t + window_s]
        last = [row for row in stable if row["elapsed_s"] >= last_t - window_s]

        def median(rows: list[dict[str, float]], key: str) -> float:
            return float(statistics.median(row[key] for row in rows))

        forces = [row["internal_fz_R_n"] for row in stable]
        x_first, x_last = median(first, "x_R_m"), median(last, "x_R_m")
        y_first, y_last = median(first, "y_R_m"), median(last, "y_R_m")
        z_first, z_last = median(first, "z_R_m"), median(last, "z_R_m")
        return {
            "sample_count": len(self.hold_samples),
            "post_settling_sample_count": len(stable),
            "settling_exclusion_s": settling,
            "tcp_window_drift_xyz_R_m": [
                x_last - x_first,
                y_last - y_first,
                z_last - z_first,
            ],
            "tcp_post_settling_range_xyz_R_m": [
                max(row["x_R_m"] for row in stable)
                - min(row["x_R_m"] for row in stable),
                max(row["y_R_m"] for row in stable)
                - min(row["y_R_m"] for row in stable),
                max(row["z_R_m"] for row in stable)
                - min(row["z_R_m"] for row in stable),
            ],
            "internal_fz_R_mean_n": float(statistics.fmean(forces)),
            "internal_fz_R_stdev_n": (
                float(statistics.stdev(forces)) if len(forces) > 1 else 0.0
            ),
            "internal_fz_R_min_n": min(forces),
            "internal_fz_R_max_n": max(forces),
            "internal_fz_R_peak_abs_n": max(abs(value) for value in forces),
        }

    def write_screening_manifest(
        self,
        bag_path: Path,
        status: str,
        error_text: Optional[str],
    ) -> Path:
        script_snapshot = bag_path / "force_hold_screening_collector_snapshot.py"
        validated_config_snapshot = bag_path / "validated_phase_b_config_snapshot.yaml"
        screening_config_snapshot = bag_path / "force_hold_screening_config_snapshot.yaml"
        snapshot_sources = (
            (Path(__file__), script_snapshot),
            (Path(str(self.p("collection_config_path"))), validated_config_snapshot),
            (Path(str(self.p("screening_config_path"))), screening_config_snapshot),
        )
        for source, destination in snapshot_sources:
            if destination.exists():
                raise FileExistsError(f"Snapshot already exists: {destination}")
            shutil.copy2(source, destination)

        gauge_delta = None
        fixture_slip = None
        if self.gauge_before_mm is not None and self.gauge_after_mm is not None:
            gauge_delta = self.gauge_after_mm - self.gauge_before_mm
            fixture_slip = abs(gauge_delta) >= float(
                self.p("gauge_slip_threshold_mm")
            )

        mcap_files = list(bag_path.glob("*.mcap"))
        manifest = {
            "schema_version": "phase_b_force_hold_screening_v1",
            "identity": {
                "run_id": str(self.p("bag_name")),
                "purpose": "force_bounds_screening",
                "created_at_utc": self.collection_started_at_utc or utc_now(),
            },
            "architecture": {
                "control_force_source": "internal_UR_FT",
                "ati_role": "passive_reference",
                "ati_control_influence": False,
                "ati_rtde_bridge_used": False,
                "tangential_slide_commanded": False,
            },
            "process": {
                "normal_force_command_n": float(self.p("normal_force_n")),
                "normal_force_magnitude_n": abs(float(self.p("normal_force_n"))),
                "hold_time_s": float(self.p("hold_time_s")),
                "contact_threshold_n": float(self.p("contact_threshold_n")),
                "measured_force_abort_n": float(
                    self.p("max_abs_normal_force_n")
                ),
                "speed_limit_z_mps": float(self.p("speed_limit_z_mps")),
            },
            "frames": {
                "base_frame": str(self.p("sliding_parent_frame")),
                "sliding_frame": str(self.p("sliding_frame")),
                "tcp_frame": str(self.p("tcp_frame")),
                "sliding_frame_transform": self.runtime_sliding_transform,
            },
            "measurements": {
                "approach_position_xyz_R_m": self.approach_position_xyz_R_m,
                "contact_position_xyz_R_m": self.contact_position_xyz_R_m,
                "return_position_xyz_R_m": self.return_position_xyz_R_m,
                "hold_summary": self.hold_summary,
                "gauge": {
                    "resolution_mm": float(self.p("gauge_resolution_mm")),
                    "slip_threshold_mm": float(
                        self.p("gauge_slip_threshold_mm")
                    ),
                    "before_mm": self.gauge_before_mm,
                    "after_mm": self.gauge_after_mm,
                    "delta_mm": gauge_delta,
                    "fixture_slip": fixture_slip,
                },
                "operator_slip_observation": self.operator_slip_observation,
            },
            "acquisition": {
                "bag_path": str(bag_path),
                "bag_sha256": sha256_files(mcap_files) if mcap_files else None,
                "internal_wrench_topic": str(self.p("wrench_topic")),
                "ati_passive_topic": str(self.p("ati_passive_topic")),
                "event_topic": str(self.p("event_topic")),
            },
            "software": {
                "validated_collection_config_path": str(
                    self.p("collection_config_path")
                ),
                "validated_collection_config_sha256": sha256_file(
                    Path(str(self.p("collection_config_path")))
                ),
                "screening_config_path": str(self.p("screening_config_path")),
                "screening_config_sha256": sha256_file(
                    Path(str(self.p("screening_config_path")))
                ),
                "screening_script_path": str(Path(__file__)),
                "screening_script_sha256": sha256_file(Path(__file__)),
                "immutable_snapshots": {
                    "collector": {
                        "path": str(script_snapshot),
                        "sha256": sha256_file(script_snapshot),
                    },
                    "validated_phase_b_config": {
                        "path": str(validated_config_snapshot),
                        "sha256": sha256_file(validated_config_snapshot),
                    },
                    "screening_config": {
                        "path": str(screening_config_snapshot),
                        "sha256": sha256_file(screening_config_snapshot),
                    },
                },
            },
            "outcome": {
                "status": status,
                "stop_reason": error_text,
                "fixture_slip": fixture_slip,
                "valid_for_training": False,
            },
        }
        path = bag_path / "screening_manifest.json"
        if path.exists():
            raise FileExistsError(f"Screening manifest already exists: {path}")
        path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.get_logger().info(f"Screening manifest written: {path}")
        return path

    def run_screening(self) -> None:
        self.collection_started_at_utc = utc_now()
        bag_path: Optional[Path] = None
        status = "ABORTED"
        error_text: Optional[str] = None
        caught: Optional[BaseException] = None
        controller_interaction_started = False

        try:
            answer = input(
                "\nPHYSICAL MOTION GATE: clear workspace and reachable "
                "emergency stop. Type RUN to move only to the qualified "
                "2 mm approach pose: "
            )
            if answer.strip() != "RUN":
                raise RuntimeError("Screening aborted before controller switch.")

            controller_interaction_started = True
            self.switch_scaled()
            self.send_goal(
                self.scaled_client,
                self.start_joints(),
                float(self.p("move_duration_s")),
            )
            self.validate_start()
            self.approach_position_xyz_R_m, _ = self.pose_vector()

            self.gauge_before_mm = parse_gauge_mm(
                input(
                    "Measure Hairpin protrusion in the contact-free approach "
                    "pose. Enter value in mm (comma or decimal point), or NA: "
                )
            )
            answer = input(
                "Type HOLD to authorize bag, internal tare and one static "
                f"{abs(float(self.p('normal_force_n'))):.2f} N / "
                f"{float(self.p('hold_time_s')):.1f} s force hold: "
            )
            if answer.strip() != "HOLD":
                raise RuntimeError("Screening aborted before bag, tare and force mode.")

            bag_path = self.start_bag()
            self.zero_sensor()
            self.wait_post_tare_baseline()
            self.switch_force()
            self.publish_event("force_controller_activated")
            self.start_force_mode()
            self.wait_contact()
            self.hold_contact_screening()

            self.stop_force_mode()
            self.switch_scaled()
            self.publish_event("return_started")
            self.send_goal(
                self.scaled_client,
                self.start_joints(),
                float(self.p("move_duration_s")),
            )
            self.validate_start()
            self.return_position_xyz_R_m, _ = self.pose_vector()
            self.publish_event("return_finished")
            status = "COMPLETED"
            self.publish_event("screening_run_completed", summary=self.hold_summary)
        except BaseException as exc:  # noqa: BLE001
            caught = exc
            error_text = f"{type(exc).__name__}: {exc}"
            if self.bag_process is not None:
                self.publish_event("screening_run_aborted", error=error_text)
        finally:
            try:
                self.stop_force_mode()
            except Exception as cleanup_exc:  # noqa: BLE001
                if error_text is None:
                    error_text = f"cleanup force mode: {cleanup_exc}"
                    status = "ABORTED"
            if controller_interaction_started:
                try:
                    self.switch_scaled()
                except Exception as cleanup_exc:  # noqa: BLE001
                    if error_text is None:
                        error_text = f"cleanup controller switch: {cleanup_exc}"
                        status = "ABORTED"
            self.stop_bag()

        if status == "COMPLETED":
            self.gauge_after_mm = parse_gauge_mm(
                input(
                    "Force mode is stopped and TCP returned. Measure Hairpin "
                    "protrusion again; enter mm or NA: "
                )
            )
            observation = input(
                "Enter observation NOSLIP, SLIP, UNKNOWN or "
                "NOT_REASSESSED_LOWER_FORCE: "
            ).strip().upper()
            allowed_observations = {
                "NOSLIP",
                "SLIP",
                "UNKNOWN",
                "NOT_REASSESSED_LOWER_FORCE",
            }
            if observation not in allowed_observations:
                raise ValueError(
                    "Invalid qualitative slip observation."
                )
            self.operator_slip_observation = observation
        if bag_path is not None and bag_path.exists():
            self.write_screening_manifest(bag_path, status, error_text)
        if caught is not None:
            raise caught

    def run(self) -> None:
        self.validate_screening_configuration()
        self.validate_ros_graph()
        self.validate_internal_wrench_signal()
        self.wait_interfaces()
        if bool(self.p("collection_dry_run")):
            self.get_logger().info(
                "DRY RUN PASSED: no controller switch, trajectory, tare, "
                "force mode or bag recording was requested."
            )
            return
        self.run_screening()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ForceHoldScreeningNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warning("Interrupted by operator.")
    except Exception as exc:  # noqa: BLE001
        node.get_logger().error(f"Force-hold screening failed: {exc}")
        raise
    finally:
        try:
            node.stop_force_mode()
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

#!/usr/bin/env python3
"""
Phase B data collection for SPI with internal UR force control.

This executable extends the validated 15 mm radial sliding implementation with
immutable metadata, structured events and passive ATI recording. It contains no
chamfer-triggered stop path. The physical motion always targets the validated
15 mm endpoint unless an existing safety check aborts it.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Optional

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import WrenchStamped
from hairpin_skills.radial_sliding_test import RadialSlidingNode
import rclpy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException


RUN_ID_PATTERN = re.compile(
    r"^pb_spi_s4_setznest2_intctrl_atipass_d015_"
    r"v[0-9]{4}_fn[0-9]{3}_[a-z0-9_]+_r[0-9]{3}$"
)
INTERNAL_WRENCH_TOPIC = "/force_torque_sensor_broadcaster/wrench"
PASSIVE_ATI_TOPIC = "/ati/wrench_passive"
INTERNAL_TARE_SERVICE = "/io_and_status_controller/zero_ftsensor"
EVENT_TOPIC = "/phase_b_spi_collection/events"
FORBIDDEN_NODE_TOKENS = (
    "ati_rtde_enable",
    "ati_rtde_disable",
    "ati_ft_rtde_bridge",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class PhaseBSPIDataCollectionNode(RadialSlidingNode):
    """Validated radial slide plus metadata and passive reference recording."""

    def __init__(self) -> None:
        # The inherited node name remains radial_sliding_test. The dedicated
        # executable and event topic distinguish this acquisition route without
        # modifying the already validated base class.
        super().__init__()
        package_share = Path(get_package_share_directory("hairpin_skills"))

        self.declare_parameter("experiment_id", "pb_spi_s4_qualification_v1")
        self.declare_parameter("purpose", "qualification")
        self.declare_parameter("repetition_index", 1)
        self.declare_parameter("batch_id", "qualification_01")
        self.declare_parameter("planned_split", "development")
        self.declare_parameter("planned_order_index", 1)
        self.declare_parameter("randomization_seed", -1)
        self.declare_parameter("manual_reference_frozen", False)
        self.declare_parameter("parameter_bounds_version", "")
        self.declare_parameter("ati_passive_topic", PASSIVE_ATI_TOPIC)
        self.declare_parameter("event_topic", EVENT_TOPIC)
        self.declare_parameter("require_ati_passive_topic", True)
        self.declare_parameter("collection_dry_run", True)
        self.declare_parameter("run_authorized", False)
        self.declare_parameter("graph_validation_timeout_s", 5.0)
        self.declare_parameter("post_tare_baseline_s", 1.0)
        self.declare_parameter("return_settle_s", 1.0)
        self.declare_parameter("internal_signal_check_duration_s", 2.0)
        self.declare_parameter("internal_signal_min_samples", 20)
        self.declare_parameter("internal_signal_min_peak_abs_n", 1.0e-6)
        self.declare_parameter(
            "ground_truth_file",
            str(package_share / "examples" / "rgt_caliper_v1.json"),
        )
        self.declare_parameter(
            "collection_config_path",
            str(package_share / "config" / "phase_b_spi_data_collection.yaml"),
        )
        self.declare_parameter(
            "manifest_schema_path",
            str(package_share / "schemas" / "run_manifest.schema.json"),
        )
        self.declare_parameter(
            "bag_qos_overrides_path",
            str(package_share / "config" / "rosbag_qos_overrides.yaml"),
        )
        self.declare_parameter(
            "repository_path",
            os.environ.get(
                "HAIRPIN_REPOSITORY_PATH",
                str(Path(__file__).resolve().parents[1]),
            ),
        )

        self.event_publisher = self.create_publisher(
            String,
            str(self.p("event_topic")),
            10,
        )
        self.collection_started_at_utc: Optional[str] = None
        self.slide_start_position: Optional[list[float]] = None
        self.slide_start_orientation_xyzw: Optional[list[float]] = None
        self.slide_end_position: Optional[list[float]] = None
        self.displacement_metrics: Optional[dict[str, Any]] = None
        self.internal_signal_samples: deque[tuple[float, ...]] = deque(
            maxlen=5000
        )
        self.internal_signal_frames: deque[str] = deque(maxlen=5000)
        self.runtime_sliding_transform: Optional[dict[str, list[float]]] = None

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        """Retain raw internal samples, then update transformed Fz_R."""
        if hasattr(self, "internal_signal_samples"):
            self.internal_signal_samples.append(
                (
                    float(msg.wrench.force.x),
                    float(msg.wrench.force.y),
                    float(msg.wrench.force.z),
                    float(msg.wrench.torque.x),
                    float(msg.wrench.torque.y),
                    float(msg.wrench.torque.z),
                )
            )
            self.internal_signal_frames.append(msg.header.frame_id)
        super()._wrench_cb(msg)

    def publish_event(self, event: str, **fields: Any) -> None:
        now = self.get_clock().now()
        payload = {
            "schema_version": "phase_b_spi_event_v1",
            "event": event,
            "run_id": str(self.p("bag_name")),
            "ros_timestamp_ns": int(now.nanoseconds),
            "monotonic_s": time.monotonic(),
            **fields,
        }
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.event_publisher.publish(message)
        self.get_logger().info(f"EVENT {message.data}")

    def validate_collection_configuration(self) -> None:
        errors = []
        run_id = str(self.p("bag_name"))
        if not RUN_ID_PATTERN.fullmatch(run_id):
            errors.append(f"run_id does not match Phase B SPI schema: {run_id}")
        if str(self.p("mode")).strip().lower() != "slide":
            errors.append("mode must be 'slide'")
        if not bool(self.p("free_space_validated")):
            errors.append("free_space_validated must be true")
        if str(self.p("wrench_topic")) != INTERNAL_WRENCH_TOPIC:
            errors.append(f"wrench_topic must be {INTERNAL_WRENCH_TOPIC}")
        if str(self.p("tare_service")) != INTERNAL_TARE_SERVICE:
            errors.append(f"tare_service must be {INTERNAL_TARE_SERVICE}")
        if str(self.p("ati_passive_topic")) != PASSIVE_ATI_TOPIC:
            errors.append(f"ati_passive_topic must be {PASSIVE_ATI_TOPIC}")
        if str(self.p("event_topic")) != EVENT_TOPIC:
            errors.append(f"event_topic must be {EVENT_TOPIC}")
        if not math.isclose(
            float(self.p("slide_distance_m")), 0.015, abs_tol=1e-12
        ):
            errors.append("slide_distance_m must remain 0.015 m")
        if float(self.p("radial_speed_mps")) <= 0.0:
            errors.append("radial_speed_mps must be positive")
        if float(self.p("normal_force_n")) >= 0.0:
            errors.append("normal_force_n must be negative in the ROS command")
        if float(self.p("post_tare_baseline_s")) < 0.5:
            errors.append("post_tare_baseline_s must be at least 0.5 s")
        if float(self.p("return_settle_s")) < 0.5:
            errors.append("return_settle_s must be at least 0.5 s")
        if float(self.p("internal_signal_check_duration_s")) < 0.25:
            errors.append(
                "internal_signal_check_duration_s must be at least 0.25 s"
            )
        if int(self.p("internal_signal_min_samples")) < 10:
            errors.append("internal_signal_min_samples must be at least 10")
        if float(self.p("internal_signal_min_peak_abs_n")) <= 0.0:
            errors.append("internal_signal_min_peak_abs_n must be positive")
        if str(self.p("planned_split")) not in {
            "development", "test", "validation"
        }:
            errors.append("planned_split is invalid")
        if not isinstance(self.p("collection_dry_run"), bool):
            errors.append("collection_dry_run must be boolean")
        if not isinstance(self.p("run_authorized"), bool):
            errors.append("run_authorized must be boolean")
        if (
            not bool(self.p("collection_dry_run"))
            and not bool(self.p("run_authorized"))
        ):
            errors.append(
                "a physical run requires run_authorized=true in addition to "
                "collection_dry_run=false"
            )

        speed_token = f"v{round(float(self.p('radial_speed_mps')) * 1e4):04d}"
        force_token = f"fn{round(abs(float(self.p('normal_force_n'))) * 10):03d}"
        if speed_token not in run_id:
            errors.append(f"run_id must contain speed token {speed_token}")
        if force_token not in run_id:
            errors.append(f"run_id must contain force token {force_token}")

        ground_truth_path = Path(str(self.p("ground_truth_file")))
        if not ground_truth_path.is_file():
            errors.append(f"ground_truth_file missing: {ground_truth_path}")
        config_path = Path(str(self.p("collection_config_path")))
        if not config_path.is_file():
            errors.append(f"collection_config_path missing: {config_path}")
        schema_path = Path(str(self.p("manifest_schema_path")))
        if not schema_path.is_file():
            errors.append(f"manifest_schema_path missing: {schema_path}")
        qos_path = Path(str(self.p("bag_qos_overrides_path")))
        if not qos_path.is_file():
            errors.append(f"bag_qos_overrides_path missing: {qos_path}")

        if errors:
            raise RuntimeError(
                "Phase B SPI configuration rejected:\n- " + "\n- ".join(errors)
            )

    def validate_ros_graph(self) -> None:
        deadline = time.monotonic() + float(
            self.p("graph_validation_timeout_s")
        )
        ati_ready = not bool(self.p("require_ati_passive_topic"))
        internal_wrench_ready = False
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            topic_types = dict(self.get_topic_names_and_types())
            ati_types = topic_types.get(str(self.p("ati_passive_topic")), [])
            ati_ready = "geometry_msgs/msg/WrenchStamped" in ati_types
            internal_types = topic_types.get(str(self.p("wrench_topic")), [])
            internal_wrench_ready = (
                "geometry_msgs/msg/WrenchStamped" in internal_types
            )
            if ati_ready and internal_wrench_ready:
                break
        if not ati_ready:
            raise RuntimeError(
                "Passive ATI topic unavailable or has wrong type: "
                f"{self.p('ati_passive_topic')}"
            )
        if not internal_wrench_ready:
            raise RuntimeError(
                "Internal UR wrench topic unavailable or has wrong type: "
                f"{self.p('wrench_topic')}"
            )

        graph_resources = [
            f"{namespace.rstrip('/')}/{name}".replace("//", "/").lower()
            for name, namespace in self.get_node_names_and_namespaces()
        ]
        graph_resources.extend(
            name.lower() for name, _ in self.get_service_names_and_types()
        )
        graph_resources.extend(
            name.lower() for name, _ in self.get_topic_names_and_types()
        )
        forbidden = [
            resource
            for resource in graph_resources
            if any(token in resource for token in FORBIDDEN_NODE_TOKENS)
        ]
        if forbidden:
            raise RuntimeError(
                "Forbidden ATI-to-RTDE component active: "
                + ", ".join(sorted(set(forbidden)))
            )

    def validate_internal_wrench_signal(self) -> None:
        """Reject an available but numerically stuck internal wrench stream."""
        self.internal_signal_samples.clear()
        self.internal_signal_frames.clear()
        deadline = time.monotonic() + float(
            self.p("internal_signal_check_duration_s")
        )
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)

        samples = list(self.internal_signal_samples)
        minimum_count = int(self.p("internal_signal_min_samples"))
        if len(samples) < minimum_count:
            raise RuntimeError(
                "Internal UR wrench preflight received too few raw "
                f"samples: {len(samples)} < {minimum_count}."
            )
        if not all(
            math.isfinite(value) for sample in samples for value in sample
        ):
            raise RuntimeError("Internal UR wrench contains NaN or Inf.")
        unexpected_frames = sorted(
            set(self.internal_signal_frames) - {"tool0_controller"}
        )
        if unexpected_frames:
            raise RuntimeError(
                "Internal UR wrench has unexpected frame_id values: "
                + ", ".join(repr(value) for value in unexpected_frames)
            )
        peak_abs = max(abs(value) for sample in samples for value in sample)
        minimum_peak = float(self.p("internal_signal_min_peak_abs_n"))
        if peak_abs < minimum_peak:
            raise RuntimeError(
                "Internal UR wrench is numerically stuck at zero: "
                f"peak absolute raw component={peak_abs:.9g} < "
                f"{minimum_peak:.9g}. "
                "No physical run is permitted."
            )

        transform_deadline = time.monotonic() + 2.0
        transform_ready = False
        while time.monotonic() < transform_deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            try:
                self.tf_buffer.lookup_transform(
                    str(self.p("sliding_frame")),
                    "tool0_controller",
                    Time(),
                )
                base_to_sliding = self.tf_buffer.lookup_transform(
                    str(self.p("sliding_parent_frame")),
                    str(self.p("sliding_frame")),
                    Time(),
                )
                t = base_to_sliding.transform.translation
                q = base_to_sliding.transform.rotation
                self.runtime_sliding_transform = {
                    "translation_xyz_m": [float(t.x), float(t.y), float(t.z)],
                    "quaternion_xyzw": [
                        float(q.x), float(q.y), float(q.z), float(q.w)
                    ],
                }
                configured_translation = [
                    float(self.p("sliding_tf_x_m")),
                    float(self.p("sliding_tf_y_m")),
                    float(self.p("sliding_tf_z_m")),
                ]
                translation_error_m = math.sqrt(
                    sum(
                        (actual - configured) ** 2
                        for actual, configured in zip(
                            self.runtime_sliding_transform[
                                "translation_xyz_m"
                            ],
                            configured_translation,
                        )
                    )
                )
                configured_quaternion = [
                    float(self.p("sliding_tf_qx")),
                    float(self.p("sliding_tf_qy")),
                    float(self.p("sliding_tf_qz")),
                    float(self.p("sliding_tf_qw")),
                ]
                runtime_quaternion = self.runtime_sliding_transform[
                    "quaternion_xyzw"
                ]
                configured_norm = math.sqrt(
                    sum(value * value for value in configured_quaternion)
                )
                runtime_norm = math.sqrt(
                    sum(value * value for value in runtime_quaternion)
                )
                quaternion_dot = abs(
                    sum(
                        actual * configured
                        for actual, configured in zip(
                            runtime_quaternion, configured_quaternion
                        )
                    )
                    / (runtime_norm * configured_norm)
                )
                rotation_error_rad = 2.0 * math.acos(
                    min(1.0, max(-1.0, quaternion_dot))
                )
                if translation_error_m > 5.0e-5:
                    raise RuntimeError(
                        "Runtime base -> sliding_frame translation differs "
                        "from the validated configuration by "
                        f"{translation_error_m * 1000.0:.3f} mm."
                    )
                if rotation_error_rad > math.radians(0.1):
                    raise RuntimeError(
                        "Runtime base -> sliding_frame rotation differs "
                        "from the validated configuration by "
                        f"{math.degrees(rotation_error_rad):.3f} deg."
                    )
                transform_ready = True
                break
            except TransformException:
                continue
        if not transform_ready:
            raise RuntimeError(
                "Internal wrench transform tool0_controller -> "
                f"{self.p('sliding_frame')} is unavailable."
            )
        self.get_logger().info(
            "Internal wrench signal preflight passed: "
            f"samples={len(samples)}, peak absolute raw component="
            f"{peak_abs:.6f}."
        )

    def start_bag(self) -> Path:
        root = Path(str(self.p("bag_root"))).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        path = root / str(self.p("bag_name"))
        if path.exists():
            raise FileExistsError(f"Bag already exists: {path}")

        topics = [
            "/tf",
            "/tf_static",
            "/joint_states",
            "/tcp_pose_broadcaster/pose",
            str(self.p("wrench_topic")),
            str(self.p("ati_passive_topic")),
            str(self.p("event_topic")),
        ]
        command = [
            "ros2", "bag", "record",
            "-s", "mcap",
            "--qos-profile-overrides-path",
            str(self.p("bag_qos_overrides_path")),
            "-o", str(path),
            "--topics", *topics,
        ]
        self.bag_process = subprocess.Popen(command, start_new_session=True)
        time.sleep(float(self.p("bag_start_wait_s")))
        if self.bag_process.poll() is not None:
            raise RuntimeError("ros2 bag record terminated.")
        if bool(self.p("publish_sliding_frame")):
            self.publish_sliding_frame()
        self.publish_event("bag_started", bag_path=str(path), topics=topics)
        return path

    def zero_sensor(self) -> None:
        self.publish_event(
            "internal_sensor_zero_requested",
            service=str(self.p("tare_service")),
        )
        super().zero_sensor()
        self.publish_event("internal_sensor_zero_completed")

    def start_force_mode(self) -> None:
        super().start_force_mode()
        self.publish_event(
            "force_mode_started",
            normal_force_command_n=float(self.p("normal_force_n")),
            normal_force_magnitude_n=abs(float(self.p("normal_force_n"))),
        )

    def stop_force_mode(self) -> None:
        was_active = self.force_mode_active
        if was_active:
            self.publish_event("force_mode_stop_requested")
        super().stop_force_mode()
        if was_active:
            self.publish_event("force_mode_stopped")

    def wait_contact(self) -> None:
        self.publish_event("contact_wait_started")
        super().wait_contact()
        self.publish_event(
            "contact_detected",
            measured_internal_fz_R_n=self.current_normal_force_n,
        )

    def stabilise_contact(self) -> None:
        self.publish_event(
            "stabilisation_started",
            target_duration_s=float(self.p("stabilisation_after_contact_s")),
        )
        super().stabilise_contact()
        self.publish_event("stabilisation_finished")

    def wait_post_tare_baseline(self) -> None:
        duration_s = float(self.p("post_tare_baseline_s"))
        self.publish_event("post_tare_baseline_started", duration_s=duration_s)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        self.publish_event("post_tare_baseline_finished")

    def wait_return_settle(self) -> None:
        """Allow pose/TF feedback to settle before validating the return."""
        duration_s = float(self.p("return_settle_s"))
        self.publish_event("return_settle_started", duration_s=duration_s)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        self.publish_event("return_settle_finished")

    def run_sliding(self) -> None:
        self.collection_started_at_utc = utc_now()
        bag_path: Optional[Path] = None
        status = "ABORTED"
        error_text: Optional[str] = None
        controller_interaction_started = False

        try:
            motion_answer = input(
                "\nPHYSICAL MOTION GATE: the next step moves the robot to the "
                "validated start pose. Confirm a clear workspace and reachable "
                "emergency stop. Type RUN to authorize this one motion sequence; "
                "anything else aborts without controller switch, motion, bag, "
                "tare or Force Mode: "
            )
            if motion_answer.strip() != "RUN":
                raise RuntimeError(
                    "Run aborted by operator before controller switch or motion."
                )

            controller_interaction_started = True
            self.switch_scaled()
            self.send_goal(
                self.scaled_client,
                self.start_joints(),
                float(self.p("move_duration_s")),
            )
            self.validate_start()

            acquisition_answer = input(
                "\nPhase B SPI qualification: start pose verified. "
                "Confirm internal UR control, passive ATI, clear 15 mm path "
                "and reachable emergency stop. Type ACQUIRE to start recording; "
                "anything else aborts without bag, tare or Force Mode: "
            )
            if acquisition_answer.strip() != "ACQUIRE":
                raise RuntimeError(
                    "Run aborted by operator before recording, tare or Force Mode."
                )

            bag_path = self.start_bag()
            self.zero_sensor()
            self.wait_post_tare_baseline()

            self.switch_force()
            self.publish_event("force_controller_activated")
            self.start_force_mode()
            self.wait_contact()
            self.stabilise_contact()

            self.slide_start_position, self.slide_start_orientation_xyzw = (
                self.pose_vector()
            )
            slide_duration_s = (
                float(self.p("slide_distance_m"))
                / float(self.p("radial_speed_mps"))
            )
            self.publish_event(
                "slide_started",
                start_position_xyz_R_m=self.slide_start_position,
                start_quaternion_xyzw=self.slide_start_orientation_xyzw,
                target_distance_m=float(self.p("slide_distance_m")),
                target_speed_mps=float(self.p("radial_speed_mps")),
                target_duration_s=slide_duration_s,
            )
            self.send_goal(
                self.passthrough_client,
                self.end_joints(),
                slide_duration_s,
                monitor_force=True,
            )
            self.slide_end_position, _ = self.pose_vector()
            self.publish_event(
                "slide_finished",
                end_position_xyz_R_m=self.slide_end_position,
            )
            self.displacement_metrics = self.validate_radial_displacement(
                self.slide_start_position,
                self.slide_end_position,
            )

            self.stop_force_mode()
            self.switch_scaled()
            self.publish_event("return_started")
            self.send_goal(
                self.scaled_client,
                self.start_joints(),
                float(self.p("move_duration_s")),
            )
            self.wait_return_settle()
            self.validate_start()
            self.publish_event("return_finished")
            status = "COMPLETED"
            self.publish_event(
                "run_completed",
                displacement_metrics=self.displacement_metrics,
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            if self.bag_process is not None:
                self.publish_event("run_aborted", error=error_text)
            raise
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
                        error_text = (
                            f"cleanup controller switch: {cleanup_exc}"
                        )
                        status = "ABORTED"
            self.stop_bag()
            if bag_path is not None and bag_path.exists():
                self.write_run_manifest(bag_path, status, error_text)

    def load_ground_truth(self) -> tuple[dict[str, Any], Path]:
        path = Path(str(self.p("ground_truth_file")))
        return json.loads(path.read_text(encoding="utf-8")), path

    def git_state(self) -> tuple[Optional[str], bool]:
        root = Path(str(self.p("repository_path"))).expanduser().resolve()
        commit_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        status_result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            capture_output=True,
            check=False,
        )
        commit = (
            commit_result.stdout.strip()
            if commit_result.returncode == 0 else None
        )
        dirty = status_result.returncode != 0 or bool(status_result.stdout.strip())
        return commit, dirty

    def write_run_manifest(
        self,
        bag_path: Path,
        status: str,
        error_text: Optional[str],
    ) -> Path:
        ground_truth, ground_truth_path = self.load_ground_truth()
        edge_x_m = float(
            ground_truth["result"]["nominal_absolute_edge_x_R_m"]
        )
        slide_start_x_m = (
            float(self.slide_start_position[0])
            if self.slide_start_position is not None else None
        )
        relative_r_gt_m = (
            edge_x_m - slide_start_x_m
            if slide_start_x_m is not None else None
        )
        config_path = Path(str(self.p("collection_config_path")))
        commit, dirty = self.git_state()
        mcap_files = list(bag_path.glob("*.mcap"))
        bag_digest = sha256_files(mcap_files) if mcap_files else None
        randomization_seed = int(self.p("randomization_seed"))
        if randomization_seed < 0:
            randomization_seed = None

        if self.runtime_sliding_transform is None:
            raise RuntimeError(
                "Runtime base -> sliding_frame transform was not captured."
            )

        manifest = {
            "schema_version": "phase_b_spi_run_manifest_v1",
            "identity": {
                "experiment_id": str(self.p("experiment_id")),
                "run_id": str(self.p("bag_name")),
                "purpose": str(self.p("purpose")),
                "repetition_index": int(self.p("repetition_index")),
                "created_at_utc": self.collection_started_at_utc or utc_now(),
                "operator": "",
            },
            "architecture": {
                "control_force_source": "internal_UR_FT",
                "ati_role": "passive_reference",
                "ati_control_influence": False,
                "ati_rtde_bridge_used": False,
            },
            "process": {
                "slide_direction": "+x_R",
                "slide_distance_m": float(self.p("slide_distance_m")),
                "slide_speed_mps": float(self.p("radial_speed_mps")),
                "normal_force_magnitude_n": abs(float(self.p("normal_force_n"))),
                "normal_force_command_n": float(self.p("normal_force_n")),
                "manual_reference_frozen": bool(
                    self.p("manual_reference_frozen")
                ),
                "parameter_bounds_version": (
                    str(self.p("parameter_bounds_version")) or None
                ),
            },
            "frames": {
                "base_frame": str(self.p("sliding_parent_frame")),
                "sliding_frame": str(self.p("sliding_frame")),
                "tcp_frame": str(self.p("tcp_frame")),
                "sliding_frame_transform": self.runtime_sliding_transform,
            },
            "acquisition": {
                "bag_path": str(bag_path),
                "storage_id": "mcap",
                "topics": {
                    "internal_wrench": str(self.p("wrench_topic")),
                    "ati_passive_wrench": str(self.p("ati_passive_topic")),
                    "tcp_pose": "/tcp_pose_broadcaster/pose",
                    "tf": "/tf",
                    "tf_static": "/tf_static",
                    "joint_states": "/joint_states",
                },
                "event_topic": str(self.p("event_topic")),
                "tare_service": str(self.p("tare_service")),
                "raw_data_immutable": True,
                "bag_sha256": bag_digest,
            },
            "software": {
                "ros_distro": "jazzy",
                "hairpin_skills_commit": commit,
                "hairpin_skills_dirty": dirty,
                "collection_config_path": str(config_path),
                "collection_config_sha256": sha256_file(config_path),
                "collection_script_sha256": sha256_file(Path(__file__)),
            },
            "doe": {
                "batch_id": str(self.p("batch_id")),
                "randomization_seed": randomization_seed,
                "planned_split": str(self.p("planned_split")),
                "planned_order_index": int(self.p("planned_order_index")),
            },
            "ground_truth": {
                "reference_file": str(ground_truth_path),
                "reference_sha256": sha256_file(ground_truth_path),
                "reference_frozen_before_detector_tuning": True,
                "absolute_edge_x_R_m": edge_x_m,
                "run_slide_start_absolute_x_R_m": slide_start_x_m,
                "run_relative_r_gt_m": relative_r_gt_m,
            },
            "outcome": {
                "status": status,
                "stop_reason": error_text,
                "anomaly_flags": [] if error_text is None else ["run_aborted"],
                "usable_for_conversion": None,
            },
        }
        manifest_path = bag_path / "run_manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"Manifest already exists: {manifest_path}")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.get_logger().info(f"Run manifest written: {manifest_path}")
        return manifest_path

    def run(self) -> None:
        # All architecture and graph checks happen before any trajectory action,
        # Force Mode request or sensor-tare call.
        self.validate_collection_configuration()
        self.validate_ros_graph()
        self.validate_internal_wrench_signal()
        # Discovery only: wait_for_server/wait_for_service sends no action
        # goal and invokes no service.
        self.wait_interfaces()
        if bool(self.p("collection_dry_run")):
            self.get_logger().info(
                "DRY RUN PASSED: configuration and ROS graph are valid. "
                "No controller switch, trajectory, tare, Force Mode or bag "
                "recording was requested."
            )
            return
        self.run_sliding()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PhaseBSPIDataCollectionNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warning("Interrupted by user.")
    except Exception as exc:  # noqa: BLE001
        node.get_logger().error(f"Phase B SPI collection failed: {exc}")
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

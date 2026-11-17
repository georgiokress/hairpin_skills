#!/usr/bin/env python3
"""Export immutable Phase-B pilot MCAP bags to the native SPI run format.

Only completed DOE runs are exported. The trajectory is segmented by the
event payload timestamps from ``slide_started`` through ``slide_finished``.
TCP pose and internal UR wrench are expressed in ``sliding_frame``. ATI is
checked for presence but never enters the model tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rosbag2_py
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation, Slerp
from std_msgs.msg import String


EVENT_TOPIC = "/phase_b_spi_collection/events"
POSE_TOPIC = "/tcp_pose_broadcaster/pose"
INTERNAL_TOPIC = "/force_torque_sensor_broadcaster/wrench"
ATI_TOPIC = "/ati/wrench_passive"
REQUIRED_TOPICS = {
    EVENT_TOPIC,
    POSE_TOPIC,
    INTERNAL_TOPIC,
    ATI_TOPIC,
    "/tf",
    "/tf_static",
    "/joint_states",
}


@dataclass
class RawRun:
    run_dir: Path
    manifest: dict[str, Any]
    annotation: dict[str, Any]
    events: dict[str, dict[str, Any]]
    pose_t_ns: np.ndarray
    pose_p_b: np.ndarray
    pose_q_bt_xyzw: np.ndarray
    wrench_t_ns: np.ndarray
    wrench_tool: np.ndarray
    topic_counts: dict[str, int]
    pose_lag_ms: np.ndarray
    wrench_lag_ms: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_root", type=Path)
    parser.add_argument("annotation_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collector_bag_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def message_stamp_ns(message: Any, fallback_ns: int) -> int:
    stamp = message.header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else int(fallback_ns)


def sorted_unique_rows(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    keep = np.concatenate(([True], np.diff(values[:, 0]) > 0))
    return values[keep]


def load_annotation(annotation_dir: Path, run_id: str) -> dict[str, Any]:
    path = annotation_dir / f"{run_id}_operator_annotation.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing operator annotation: {path}")
    annotation = json.loads(path.read_text(encoding="utf-8"))
    if annotation.get("run_id") != run_id:
        raise ValueError(f"Annotation run_id mismatch: {path}")
    annotation["_path"] = str(path)
    return annotation


def load_raw_run(run_dir: Path, annotation_dir: Path) -> RawRun:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest["identity"]["run_id"]
    annotation = load_annotation(annotation_dir, run_id)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(run_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    events: dict[str, dict[str, Any]] = {}
    poses: list[tuple[float, ...]] = []
    wrenches: list[tuple[float, ...]] = []
    pose_lags: list[float] = []
    wrench_lags: list[float] = []
    topic_counts: dict[str, int] = {}
    pose_frames: set[str] = set()
    wrench_frames: set[str] = set()

    while reader.has_next():
        topic, serialized, bag_time_ns = reader.read_next()
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        if topic == EVENT_TOPIC:
            payload = json.loads(deserialize_message(serialized, String).data)
            name = payload["event"]
            if name in events:
                raise ValueError(f"Duplicate event {name} in {run_id}")
            events[name] = payload
        elif topic == POSE_TOPIC:
            message = deserialize_message(serialized, PoseStamped)
            sample_time_ns = message_stamp_ns(message, bag_time_ns)
            p = message.pose.position
            q = message.pose.orientation
            poses.append(
                (sample_time_ns, p.x, p.y, p.z, q.x, q.y, q.z, q.w)
            )
            pose_lags.append((int(bag_time_ns) - sample_time_ns) * 1e-6)
            pose_frames.add(message.header.frame_id)
        elif topic == INTERNAL_TOPIC:
            message = deserialize_message(serialized, WrenchStamped)
            sample_time_ns = message_stamp_ns(message, bag_time_ns)
            f = message.wrench.force
            m = message.wrench.torque
            wrenches.append(
                (sample_time_ns, f.x, f.y, f.z, m.x, m.y, m.z)
            )
            wrench_lags.append((int(bag_time_ns) - sample_time_ns) * 1e-6)
            wrench_frames.add(message.header.frame_id)

    missing_topics = sorted(REQUIRED_TOPICS - set(topic_counts))
    if missing_topics:
        raise ValueError(f"Missing required topics in {run_id}: {missing_topics}")
    if pose_frames != {"base"}:
        raise ValueError(f"Unexpected pose frames in {run_id}: {pose_frames}")
    if wrench_frames != {"tool0_controller"}:
        raise ValueError(f"Unexpected internal wrench frames in {run_id}: {wrench_frames}")
    for event in ("slide_started", "slide_finished"):
        if event not in events:
            raise ValueError(f"Missing event {event} in {run_id}")

    pose_array = sorted_unique_rows(np.asarray(poses, dtype=float))
    wrench_array = sorted_unique_rows(np.asarray(wrenches, dtype=float))
    if not np.all(np.isfinite(pose_array)) or not np.all(np.isfinite(wrench_array)):
        raise ValueError(f"NaN/Inf in source messages for {run_id}")
    return RawRun(
        run_dir=run_dir,
        manifest=manifest,
        annotation=annotation,
        events=events,
        pose_t_ns=pose_array[:, 0].astype(np.int64),
        pose_p_b=pose_array[:, 1:4],
        pose_q_bt_xyzw=pose_array[:, 4:8],
        wrench_t_ns=wrench_array[:, 0].astype(np.int64),
        wrench_tool=wrench_array[:, 1:7],
        topic_counts=topic_counts,
        pose_lag_ms=np.asarray(pose_lags, dtype=float),
        wrench_lag_ms=np.asarray(wrench_lags, dtype=float),
    )


def continuous_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = quat_xyzw[:, [3, 0, 1, 2]]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    for index in range(1, len(quat)):
        if np.dot(quat[index], quat[index - 1]) < 0:
            quat[index] *= -1
    return quat


def interp_columns(t_src_ns: np.ndarray, y_src: np.ndarray, t_query_ns: np.ndarray) -> np.ndarray:
    reference_ns = int(t_src_ns[0])
    t_src_s = (t_src_ns - reference_ns) * 1e-9
    t_query_s = (t_query_ns - reference_ns) * 1e-9
    return np.stack(
        [np.interp(t_query_s, t_src_s, y_src[:, index]) for index in range(y_src.shape[1])],
        axis=1,
    )


def interpolated_rotations(
    t_src_ns: np.ndarray,
    quat_xyzw: np.ndarray,
    t_query_ns: np.ndarray,
) -> Rotation:
    reference_ns = int(t_src_ns[0])
    t_src_s = (t_src_ns - reference_ns) * 1e-9
    t_query_s = (t_query_ns - reference_ns) * 1e-9
    if t_query_s[0] < t_src_s[0] or t_query_s[-1] > t_src_s[-1]:
        raise ValueError("Rotation query outside recorded pose interval")
    return Slerp(t_src_s, Rotation.from_quat(quat_xyzw))(t_query_s)


def median_gap_ms(times_ns: np.ndarray, start_ns: int, end_ns: int) -> tuple[float, float]:
    selected = times_ns[(times_ns >= start_ns) & (times_ns <= end_ns)]
    gaps_ms = np.diff(selected) * 1e-6
    return float(np.median(gaps_ms)), float(np.max(gaps_ms))


def event_time_ns(run: RawRun, name: str) -> int:
    return int(run.events[name]["ros_timestamp_ns"])


def export_one(
    run: RawRun,
    output_root: Path,
) -> dict[str, Any]:
    manifest = run.manifest
    run_id = manifest["identity"]["run_id"]
    start_ns = event_time_ns(run, "slide_started")
    finish_ns = event_time_ns(run, "slide_finished")
    if not start_ns < finish_ns:
        raise ValueError(f"Invalid slide interval in {run_id}")

    transform = manifest["frames"]["sliding_frame_transform"]
    p_bs = np.asarray(transform["translation_xyz_m"], dtype=float)
    r_bs = Rotation.from_quat(transform["quaternion_xyzw"]).as_matrix()
    pose_rotations_bt = Rotation.from_quat(run.pose_q_bt_xyzw)
    pose_p_r = (r_bs.T @ (run.pose_p_b - p_bs).T).T
    pose_r_rt = Rotation.from_matrix(r_bs.T[None, :, :] @ pose_rotations_bt.as_matrix())

    native_inside = run.pose_t_ns[
        (run.pose_t_ns > start_ns) & (run.pose_t_ns < finish_ns)
    ]
    series_t_ns = np.concatenate(
        (np.asarray([start_ns], dtype=np.int64), native_inside, np.asarray([finish_ns], dtype=np.int64))
    )
    pos = interp_columns(run.pose_t_ns, pose_p_r, series_t_ns)
    quat_wxyz = continuous_wxyz(
        interpolated_rotations(run.pose_t_ns, pose_r_rt.as_quat(), series_t_ns).as_quat()
    )

    wrench_pose_overlap = (
        (run.wrench_t_ns >= run.pose_t_ns[0])
        & (run.wrench_t_ns <= run.pose_t_ns[-1])
    )
    wrench_t_ns = run.wrench_t_ns[wrench_pose_overlap]
    wrench_tool = run.wrench_tool[wrench_pose_overlap]
    wrench_pose_rotations = interpolated_rotations(
        run.pose_t_ns, run.pose_q_bt_xyzw, wrench_t_ns
    ).as_matrix()
    r_rt_wrench = r_bs.T[None, :, :] @ wrench_pose_rotations
    force_r = np.einsum("nij,nj->ni", r_rt_wrench, wrench_tool[:, :3])
    torque_r_at_tcp = np.einsum("nij,nj->ni", r_rt_wrench, wrench_tool[:, 3:])
    wrench_r = np.column_stack((force_r, torque_r_at_tcp))

    has_baseline_events = {
        "post_tare_baseline_started",
        "post_tare_baseline_finished",
    }.issubset(run.events)
    if has_baseline_events:
        baseline_start_ns = event_time_ns(run, "post_tare_baseline_started")
        baseline_end_ns = event_time_ns(run, "post_tare_baseline_finished")
        baseline_method = "recorded_event_window"
    else:
        baseline_start_ns = 0
        baseline_end_ns = 0
    baseline_mask = (
        (wrench_t_ns >= baseline_start_ns)
        & (wrench_t_ns <= baseline_end_ns)
    )
    if np.count_nonzero(baseline_mask) < 50:
        # The early event publisher and rosbag topic subscriptions did not
        # always become ready at the same instant. Select an actually recorded
        # contact-free fallback interval using the earliest available physical
        # boundary. Hardware tare has already been requested by the collector.
        if "force_controller_activated" in run.events:
            baseline_end_ns = event_time_ns(run, "force_controller_activated")
            baseline_method = "recorded_pre_force_controller_fallback"
        elif "force_mode_started" in run.events:
            baseline_end_ns = event_time_ns(run, "force_mode_started")
            baseline_method = "recorded_pre_force_mode_fallback"
        elif "contact_wait_started" in run.events:
            baseline_end_ns = event_time_ns(run, "contact_wait_started")
            baseline_method = "recorded_pre_contact_wait_fallback"
        elif "contact_detected" in run.events:
            baseline_end_ns = event_time_ns(run, "contact_detected") - int(0.5e9)
            baseline_method = "recorded_pre_contact_detected_fallback_with_0p5s_guard"
        else:
            raise ValueError(f"No contact-free baseline anchor in {run_id}")
        baseline_start_ns = max(int(wrench_t_ns[0]), baseline_end_ns - int(0.5e9))
        if "internal_sensor_zero_completed" in run.events:
            baseline_start_ns = max(
                baseline_start_ns,
                event_time_ns(run, "internal_sensor_zero_completed"),
            )
        baseline_mask = (
            (wrench_t_ns >= baseline_start_ns)
            & (wrench_t_ns <= baseline_end_ns)
        )
    if np.count_nonzero(baseline_mask) < 50:
        raise ValueError(
            f"Insufficient post-tare baseline samples in {run_id}: "
            f"n={np.count_nonzero(baseline_mask)}, "
            f"window=[{baseline_start_ns}, {baseline_end_ns}], "
            f"wrench_range=[{wrench_t_ns[0]}, {wrench_t_ns[-1]}]"
        )
    baseline = np.median(wrench_r[baseline_mask], axis=0)
    wrench_debiased = wrench_r - baseline
    wrench = interp_columns(wrench_t_ns, wrench_debiased, series_t_ns)

    t = (series_t_ns - start_ns) * 1e-9
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"Non-monotonic exported timestamps in {run_id}")
    x_start_reference_m = float(
        manifest["ground_truth"]["run_slide_start_absolute_x_R_m"]
    )
    d = pos[:, 0] - x_start_reference_m
    if d[-1] < 0.0145 or np.min(np.diff(d)) < -0.0002:
        raise ValueError(f"Implausible +x_R trajectory in {run_id}: d_end={d[-1]}")

    block_match = re.search(r"(\d+)$", manifest["doe"]["batch_id"])
    if block_match is None:
        raise ValueError(f"Cannot parse block in {run_id}")
    operator_slip = bool(run.annotation["operator_observation"]["hairpin_slip"])
    r_gt_run_mm = float(manifest["ground_truth"]["run_relative_r_gt_m"]) * 1000
    point_start = np.concatenate((pos[0], quat_wxyz[0]))
    mcap_paths = list(run.run_dir.glob("*.mcap"))
    if collector_bag_sha256(mcap_paths) != manifest["acquisition"]["bag_sha256"]:
        raise ValueError(f"Raw MCAP digest mismatch in {run_id}")

    pre_edge_mask = (d >= 0.002) & (d <= 0.0075)
    force_pre_edge_median = float(np.median(wrench[pre_edge_mask, 2]))
    force_setpoint = float(manifest["process"]["normal_force_magnitude_n"])
    force_warning_limit = max(1.0, 0.2 * force_setpoint)
    force_tracking_warning = abs(force_pre_edge_median - force_setpoint) > force_warning_limit
    pose_median_gap_ms, pose_max_gap_ms = median_gap_ms(run.pose_t_ns, start_ns, finish_ns)
    wrench_median_gap_ms, wrench_max_gap_ms = median_gap_ms(run.wrench_t_ns, start_ns, finish_ns)

    meta = {
        "schema_version": "phase_b_spi_native_export_v1",
        "run_id": run_id,
        "block": int(block_match.group(1)),
        "planned_order_index": int(manifest["doe"]["planned_order_index"]),
        "v_slide_mm_s": float(manifest["process"]["slide_speed_mps"]) * 1000,
        "f_n_mag_N": force_setpoint,
        "r_gt_nom_mm": float(manifest["ground_truth"]["absolute_edge_x_R_m"]) * 1000,
        "r_gt_run_mm": r_gt_run_mm,
        "purpose": "training",
        "technical_valid": True,
        "operator_slip": operator_slip,
        "point_start": point_start.tolist(),
        "x_r_start_m": x_start_reference_m,
        "architecture": manifest["architecture"],
        "source": {
            "raw_bag_dir": str(run.run_dir),
            "manifest_sha256": sha256_file(run.run_dir / "run_manifest.json"),
            "collector_bag_sha256": manifest["acquisition"]["bag_sha256"],
            "operator_annotation_sha256": sha256_file(
                Path(run.annotation.get("_path", ""))
            ) if run.annotation.get("_path") else None,
            "raw_data_modified": False,
        },
        "segmentation": {
            "start_event": "slide_started",
            "end_event": "slide_finished",
            "event_time_source": "payload.ros_timestamp_ns",
            "measurement_time_source": "message_header_stamp",
        },
        "frames": {
            "pose_source": "base",
            "wrench_source": "tool0_controller",
            "target_axes": "sliding_frame",
            "wrench_reference_point": "TCP/tool0_controller origin",
            "wrench_transform": "rotate force and torque axes only; no moment-arm shift",
            "quaternion_order": "wxyz",
        },
        "bias": {
            "sensor_tare": "/io_and_status_controller/zero_ftsensor before contact",
            "software_baseline_method": baseline_method,
            "baseline_start_ros_ns": baseline_start_ns,
            "baseline_end_ros_ns": baseline_end_ns,
            "baseline_median_Fxyz_N_Mxyz_Nm": baseline.tolist(),
        },
        "qc": {
            "topic_counts": run.topic_counts,
            "slide_duration_s": float(t[-1]),
            "end_displacement_x_R_mm": float(d[-1] * 1000),
            "pose_median_gap_ms": pose_median_gap_ms,
            "pose_max_gap_ms": pose_max_gap_ms,
            "wrench_median_gap_ms": wrench_median_gap_ms,
            "wrench_max_gap_ms": wrench_max_gap_ms,
            "pose_median_bag_minus_header_ms": float(np.median(run.pose_lag_ms)),
            "wrench_median_bag_minus_header_ms": float(np.median(run.wrench_lag_ms)),
            "pre_edge_internal_Fz_R_median_N": force_pre_edge_median,
            "force_tracking_warning": force_tracking_warning,
            "force_tracking_warning_limit_N": force_warning_limit,
            "ati_present_but_excluded_from_tensor": True,
        },
    }

    run_output = output_root / run_id
    run_output.mkdir()
    (run_output / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        run_output / "series.npz",
        t=t.astype(np.float64),
        pos=pos.astype(np.float64),
        quat=quat_wxyz.astype(np.float64),
        wrench=wrench.astype(np.float64),
    )
    return {
        "run_id": run_id,
        "block": meta["block"],
        "planned_order_index": meta["planned_order_index"],
        "v_slide_mm_s": meta["v_slide_mm_s"],
        "f_n_mag_N": force_setpoint,
        "operator_slip": operator_slip,
        "baseline_method": baseline_method,
        "n_samples": int(len(t)),
        "slide_duration_s": float(t[-1]),
        "end_displacement_x_R_mm": float(d[-1] * 1000),
        "pre_edge_internal_Fz_R_median_N": force_pre_edge_median,
        "force_tracking_warning": force_tracking_warning,
        "pose_max_gap_ms": pose_max_gap_ms,
        "wrench_max_gap_ms": wrench_max_gap_ms,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    manifests = sorted(args.bag_root.glob("*/run_manifest.json"))
    completed_dirs = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("outcome", {}).get("status") == "COMPLETED":
            completed_dirs.append(path.parent)
    if len(completed_dirs) != 27:
        raise ValueError(f"Expected 27 completed runs, found {len(completed_dirs)}")

    raw_runs = [load_raw_run(path, args.annotation_dir) for path in completed_dirs]
    for run in raw_runs:
        architecture = run.manifest["architecture"]
        if architecture != {
            "control_force_source": "internal_UR_FT",
            "ati_role": "passive_reference",
            "ati_control_influence": False,
            "ati_rtde_bridge_used": False,
        }:
            raise ValueError(f"Architecture mismatch in {run.run_dir.name}")
        if run.annotation.get("technical_status") != "COMPLETED":
            raise ValueError(f"Annotation is not COMPLETED for {run.run_dir.name}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        summaries = [
            export_one(
                run,
                stage,
            )
            for run in sorted(
                raw_runs,
                key=lambda item: int(item.manifest["doe"]["planned_order_index"]),
            )
        ]
        orders = [row["planned_order_index"] for row in summaries]
        if orders != list(range(1, 28)):
            raise ValueError(f"Completed DOE orders are not exactly 1..27: {orders}")
        factors = {
            (row["v_slide_mm_s"], row["f_n_mag_N"]): 0 for row in summaries
        }
        for row in summaries:
            factors[(row["v_slide_mm_s"], row["f_n_mag_N"])] += 1
        if len(factors) != 9 or set(factors.values()) != {3}:
            raise ValueError(f"Unbalanced 3x3 design: {factors}")
        fz_medians = np.asarray(
            [row["pre_edge_internal_Fz_R_median_N"] for row in summaries]
        )
        export_manifest = {
            "schema_version": "phase_b_spi_export_manifest_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_bag_root": str(args.bag_root),
            "source_annotation_dir": str(args.annotation_dir),
            "raw_bags_modified": False,
            "exporter_path": str(Path(__file__).resolve()),
            "exporter_sha256": sha256_file(Path(__file__)),
            "n_runs": len(summaries),
            "design": {
                "type": "replicated_3x3_full_factorial_in_three_complete_blocks",
                "factor_repetition_counts": {
                    f"v{v:g}_F{force:g}": count
                    for (v, force), count in sorted(factors.items())
                },
            },
            "baseline_methods": {
                method: sum(row["baseline_method"] == method for row in summaries)
                for method in sorted({row["baseline_method"] for row in summaries})
            },
            "fz_sign_for_spi_prior": 1.0 if float(np.median(fz_medians)) > 0 else -1.0,
            "force_tracking_warning_count": sum(row["force_tracking_warning"] for row in summaries),
            "runs": summaries,
        }
        (stage / "export_manifest.json").write_text(
            json.dumps(export_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"EXPORTED {len(summaries)} RUNS: {args.output_dir}")
    print(
        "FZ SIGN:",
        json.loads((args.output_dir / "export_manifest.json").read_text())["fz_sign_for_spi_prior"],
    )


if __name__ == "__main__":
    main()

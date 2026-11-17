#!/usr/bin/env python3
"""Read-only ROS extraction; write a new, isolated main90 derivative."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent / 'spi_bounds_v1'))
import export_rosbags as legacy
import rosbag2_py
from geometry_msgs.msg import WrenchStamped
from rclpy.serialization import deserialize_message

ARCH = dict(control_force_source='internal_UR_FT', ati_role='passive_reference',
            ati_control_influence=False, ati_rtde_bridge_used=False)
CONFIG_ROOT = Path(os.environ.get(
    'HAIRPIN_MAIN_CONFIG_ROOT',
    'config/phase_b_spi/main_doe_v1',
)).expanduser().resolve()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_annotation(annotation, force):
    obs = annotation.get('operator_observation', {})
    slip = obs.get('hairpin_slip')
    if type(slip) is not bool:
        raise ValueError('Missing/unknown slip observation: never coerce null to NOSLIP')
    assumed = annotation.get('source') == 'USER_POLICY_ASSUMPTION'
    if assumed and (force >= 10 or slip or obs.get('observation_confirmed') is not False):
        raise ValueError('Invalid assumed-NOSLIP policy application')
    if not assumed and obs.get('observation_confirmed') is False:
        raise ValueError('Unconfirmed observation without an explicit assumption policy')
    if annotation.get('technical_status') != 'COMPLETED':
        raise ValueError('Operator sidecar is not technically COMPLETED')
    return dict(operator_slip=slip, slip_basis='assumed' if assumed else 'observed',
                slip_observation_confirmed=not assumed, slip_source=annotation.get('source'),
                slip_policy_reference=annotation.get('policy_reference'))


def extract_ati(run, output, meta):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(run.run_dir), storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[legacy.ATI_TOPIC]))
    rows, frames = [], set()
    while reader.has_next():
        _, serialized, bag_ns = reader.read_next()
        msg = deserialize_message(serialized, WrenchStamped)
        f, m = msg.wrench.force, msg.wrench.torque
        rows.append((legacy.message_stamp_ns(msg, bag_ns), f.x, f.y, f.z, m.x, m.y, m.z))
        frames.add(msg.header.frame_id)
    if frames != {'ati_ur_aligned'}:
        raise ValueError(f'Unexpected passive ATI frames: {frames}')
    a = legacy.sorted_unique_rows(np.asarray(rows, dtype=float))
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite passive ATI samples')
    stamps = a[:, 0].astype(np.int64)
    overlap = (stamps >= run.pose_t_ns[0]) & (stamps <= run.pose_t_ns[-1])
    stamps, a = stamps[overlap], a[overlap, 1:]
    rotation = Rotation.from_quat(run.manifest['frames']['sliding_frame_transform']['quaternion_xyzw']).as_matrix()
    rt = rotation.T[None] @ legacy.interpolated_rotations(run.pose_t_ns, run.pose_q_bt_xyzw, stamps).as_matrix()
    wrench = np.column_stack([np.einsum('nij,nj->ni', rt, a[:, :3]),
                             np.einsum('nij,nj->ni', rt, a[:, 3:])])
    start, end = legacy.event_time_ns(run, 'slide_started'), legacy.event_time_ns(run, 'slide_finished')
    b = meta['bias']
    mask = (stamps >= b['baseline_start_ros_ns']) & (stamps <= b['baseline_end_ros_ns'])
    n_baseline = int(mask.sum())
    baseline = np.median(wrench[mask], axis=0) if n_baseline >= 50 else None
    inside = (stamps >= start) & (stamps <= end)
    if not inside.any():
        raise ValueError('No passive ATI samples during slide')
    pos = legacy.interp_columns(run.pose_t_ns, run.pose_p_b, stamps[inside])
    p_bs = np.asarray(run.manifest['frames']['sliding_frame_transform']['translation_xyz_m'])
    pos_r = (rotation.T @ (pos - p_bs).T).T
    np.savez_compressed(output / 'ati_series.npz', t=(stamps[inside] - start) * 1e-9,
                        pos=pos_r, wrench_raw=wrench[inside],
                        wrench=wrench[inside] - baseline if baseline is not None else wrench[inside],
                        baseline_corrected=np.asarray(baseline is not None))
    return dict(status='ok' if baseline is not None else 'baseline_unavailable_raw_only',
                source_topic=legacy.ATI_TOPIC, source_frame='ati_ur_aligned',
                orientation_assumption='ati_ur_aligned shares tool0_controller axes',
                target_axes='sliding_frame', torque_reference_point='ATI measurement origin',
                moment_arm_shift_applied=False, excluded_from_model_inputs=True,
                baseline_method=b['software_baseline_method'], baseline_sample_count=n_baseline,
                baseline_median_Fxyz_N_Mxyz_Nm=baseline.tolist() if baseline is not None else None,
                baseline_start_ros_ns=b['baseline_start_ros_ns'], baseline_end_ros_ns=b['baseline_end_ros_ns'],
                baseline_Fz_std_N=float(np.std(wrench[mask, 2])) if baseline is not None else None,
                slide_sample_count=int(inside.sum()), pose_uncovered_samples_omitted=int((~overlap).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inventory', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    inventory = json.loads(args.inventory.read_text())
    rows = sorted(inventory['runs'], key=lambda r: r['sequence_index'])
    if len(rows) != 90 or [r['sequence_index'] for r in rows] != list(range(1, 91)):
        raise ValueError('Inventory must contain exactly the 90 unique planned runs')
    args.output.mkdir(parents=True, exist_ok=False)
    native = args.output / 'native_runs'
    native.mkdir()
    write_json(args.output / 'source_acquisition_inventory.json', inventory)
    successes, excluded = [], []
    for row in rows:
        rid = row['run_id']
        try:
            raw = legacy.load_raw_run(Path(row['bag_dir']), args.inventory.parent / 'operator_annotations')
            if raw.manifest['architecture'] != ARCH or raw.manifest['outcome']['status'] != 'COMPLETED':
                raise ValueError('Unexpected architecture or incomplete raw run')
            force = float(raw.manifest['process']['normal_force_magnitude_n'])
            provenance = validate_annotation(raw.annotation, force)
            begin, end = legacy.event_time_ns(raw, 'slide_started'), legacy.event_time_ns(raw, 'slide_finished')
            if not (raw.pose_t_ns[0] <= begin < end <= raw.pose_t_ns[-1]):
                raise ValueError('Slide events not bracketed by recorded TCP poses')
            if not (raw.wrench_t_ns[0] <= begin < end <= raw.wrench_t_ns[-1]):
                raise ValueError('Slide events not bracketed by recorded internal wrench')
            summary = legacy.export_one(raw, native)
            folder = native / rid
            meta = json.loads((folder / 'meta.json').read_text())
            context_path = CONFIG_ROOT / f'{rid}_run_context_v1.json'
            context = json.loads(context_path.read_text())
            meta.update(provenance)
            meta.update(schema_version='phase_b_spi_main90_native_export_v1', series='main_doe_v1',
                        split='development', sequence_index=row['sequence_index'],
                        logical_run_id=context.get('logical_run_id', rid), attempt_index=context.get('attempt_index', 1),
                        combination_id=context.get('combination_id'), order_in_block=context.get('order_in_block'),
                        hairpin_id=context.get('hairpin_id'), prior_reuse_count=context.get('prior_reuse_count'),
                        setup_id=context.get('setup_id'), contact_status=raw.annotation.get('contact_assessment', 'unqualified'),
                        force_command_N=raw.manifest['process']['normal_force_command_n'],
                        actual_start_utc=raw.manifest['identity']['created_at_utc'],
                        ground_truth=raw.manifest['ground_truth'], technical_status='COMPLETED',
                        endpoint_policy_version=context.get('endpoint_policy_version'),
                        host_after_reboot=context.get('host_after_reboot'),
                        recorder_transport_lost_messages_aggregate=row.get('recorder_transport_lost_messages_aggregate'))
            meta['source'].update(run_context_path=str(context_path), run_context_sha256=legacy.sha256_file(context_path))
            try:
                meta['ati'] = extract_ati(raw, folder, meta)
            except Exception as error:
                meta['ati'] = dict(status='reference_export_failed', error=f'{type(error).__name__}: {error}',
                                   excluded_from_model_inputs=True)
            # This new derivative is still being assembled; original metadata is never modified.
            (folder / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            summary.update(provenance, sequence_index=row['sequence_index'], ati_status=meta['ati']['status'])
            successes.append(summary)
        except Exception as error:
            excluded.append(dict(run_id=rid, sequence_index=row['sequence_index'],
                                 reason=f'{type(error).__name__}: {error}', success=None))
        if row['sequence_index'] % 5 == 0:
            print(f"Native extraction {row['sequence_index']}/90; eligible={len(successes)}; excluded={len(excluded)}", flush=True)
    write_json(args.output / 'native_export_manifest.json', dict(schema_version='main90_native_manifest_v1',
               inventory_sha256=legacy.sha256_file(args.inventory), raw_data_modified=False,
               exported_runs=successes, excluded_runs=excluded, n_exported=len(successes), n_excluded=len(excluded)))


if __name__ == '__main__':
    main()

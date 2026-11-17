#!/usr/bin/env python3
"""Recover only missing ancillary context; never recover missing measurements."""
import argparse
import json
import os
from pathlib import Path

import extract_native as extractor

CONFIG_ROOT = Path(os.environ.get('HAIRPIN_MAIN_CONFIG_ROOT', 'config/phase_b_spi/main_doe_v1')).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get('HAIRPIN_MAIN_RESULTS_ROOT', 'results/phase_b_spi/main_doe_v1')).expanduser().resolve()
PLAN_PATH = Path(os.environ.get('HAIRPIN_MAIN_RUN_PLAN', 'run_plan_v1.json')).expanduser().resolve()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', type=Path)
    args = p.parse_args()
    root = args.dataset
    original = json.loads((root / 'native_export_manifest.json').read_text())
    inventory = json.loads((root / 'source_acquisition_inventory.json').read_text())
    lookup = {r['run_id']: r for r in inventory['runs']}
    plan_path = PLAN_PATH
    plan = {r['run_id']: r for r in json.loads(plan_path.read_text())['development_runs']}
    reconciled, remaining = [], []
    exported = list(original['exported_runs'])
    for exclusion in original['excluded_runs']:
        rid = exclusion['run_id']
        folder = root / 'native_runs' / rid
        context = CONFIG_ROOT / f'{rid}_run_context_v1.json'
        if context.exists() or 'FileNotFoundError' not in exclusion['reason'] or str(context) not in exclusion['reason']:
            remaining.append(exclusion)
            continue
        if not (folder / 'series.npz').exists() or rid not in plan:
            remaining.append(exclusion)
            continue
        row, planned = lookup[rid], plan[rid]
        raw = extractor.legacy.load_raw_run(Path(row['bag_dir']),
                  RESULTS_ROOT / 'operator_annotations')
        manifest = raw.manifest
        assert manifest['outcome']['status'] == 'COMPLETED' and manifest['architecture'] == extractor.ARCH
        assert manifest['process']['normal_force_magnitude_n'] == planned['F_N'] == row['force_N']
        assert abs(manifest['process']['slide_speed_mps'] * 1000 - planned['v_mm_s']) < 1e-9
        assert manifest['doe']['planned_order_index'] == planned['order_in_block']
        meta_path = folder / 'meta.json'
        meta = json.loads(meta_path.read_text())
        assert extractor.legacy.sha256_file(raw.run_dir / 'run_manifest.json') == meta['source']['manifest_sha256']
        assert extractor.legacy.collector_bag_sha256(list(raw.run_dir.glob('*.mcap'))) == manifest['acquisition']['bag_sha256']
        provenance = extractor.validate_annotation(raw.annotation, planned['F_N'])
        meta.update(provenance)
        meta.update(schema_version='phase_b_spi_main90_native_export_v1', series='main_doe_v1', split='development',
                    sequence_index=row['sequence_index'], logical_run_id=raw.annotation.get('logical_run_id', rid),
                    attempt_index=raw.annotation.get('attempt_index'), combination_id=planned['combination_id'],
                    order_in_block=planned['order_in_block'], hairpin_id=raw.annotation.get('hairpin_id'),
                    setup_id=None, prior_reuse_count=None, host_after_reboot=None,
                    contact_status=raw.annotation.get('contact_assessment', 'unqualified'),
                    force_command_N=manifest['process']['normal_force_command_n'],
                    actual_start_utc=manifest['identity']['created_at_utc'], ground_truth=manifest['ground_truth'],
                    technical_status='COMPLETED', endpoint_policy_version=raw.annotation.get('endpoint_policy_version'),
                    recorder_transport_lost_messages_aggregate=row.get('recorder_transport_lost_messages_aggregate'),
                    original_run_context_missing=True)
        meta['source'].update(run_context_path=None, run_context_sha256=None,
                              ancillary_context_reconstruction=dict(plan_path=str(plan_path),
                              plan_sha256=extractor.legacy.sha256_file(plan_path),
                              method='Exact run-ID match to frozen DOE plan, verified against completed raw manifest and operator sidecar; unknown fields retained null'))
        try:
            meta['ati'] = extractor.extract_ati(raw, folder, meta)
        except Exception as error:
            meta['ati'] = dict(status='reference_export_failed', error=f'{type(error).__name__}: {error}',
                               excluded_from_model_inputs=True)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        exported.append(dict(run_id=rid, sequence_index=row['sequence_index'], block=meta['block'],
                             **provenance, ati_status=meta['ati']['status']))
        reconciled.append(dict(run_id=rid, original_exclusion=exclusion['reason'],
                               resolution='Ancillary context reconstructed from exact frozen-plan match; no measurement values or labels invented',
                               original_run_context_missing=True))
    exported.sort(key=lambda r: r['sequence_index'])
    extractor.write_json(root / 'native_reconciliation_manifest_v1.json', dict(
        schema_version='main90_native_reconciliation_v1', original_manifest='native_export_manifest.json',
        exported_runs=exported, excluded_runs=remaining, reconciled_runs=reconciled,
        n_exported=len(exported), n_excluded=len(remaining), raw_data_modified=False))
    print(f'Native reconciliation: {len(exported)} eligible, {len(remaining)} exclusions, {len(reconciled)} ancillary contexts recovered')


if __name__ == '__main__':
    main()

"""Audit/harmonize historical Pilot into a separate 117-run Development bundle."""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tables
import torch

from data import DETECTOR_SHA, sha

BASE = Path(os.environ.get('HAIRPIN_DATASET_ROOT', 'datasets/phase_b_spi')).expanduser().resolve()
MAIN = BASE / 'main_doe_v1_spi_v1'
PILOT = BASE / 'pilot_doe_v0_spi_v1'
CODE = Path(__file__).resolve().parent.parent / 'data_conversion/spi_bounds_v1'
PILOT_ANNOTATIONS = Path(os.environ.get(
    'HAIRPIN_PILOT_ANNOTATIONS',
    'results/phase_b_spi/pilot_doe_v0/operator_annotations',
)).expanduser().resolve()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def raw_hash(folder):
    digest = hashlib.sha256()
    paths = sorted(folder.glob('*.mcap'), key=lambda p: p.name)
    if not paths:
        raise ValueError('Missing raw MCAP')
    for path in paths:
        digest.update(path.name.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
    return digest.hexdigest()


def add_metadata(path, rows):
    with tables.open_file(path, 'a') as handle:
        for group in ('valid_length', 'physical_sample_mask', 'run_metadata'):
            handle.create_group('/', group)
        lengths = np.array([r['valid_length'] for r in rows], dtype=np.int32)
        handle.create_array('/valid_length', '0', lengths)
        handle.create_array('/physical_sample_mask', '0', np.arange(640)[None] < lengths[:, None])
        for name, value in dict(run_id=np.array([r['run_id'].encode() for r in rows]),
            slip_observation_confirmed=np.array([r['slip_observation_confirmed'] for r in rows]),
            native_e_signed_mm=np.array([r['e_signed_mm'] if r['e_signed_mm'] is not None else np.nan for r in rows]),
            native_t_det_s=np.array([r['t_det_native_s'] if r['t_det_native_s'] is not None else np.nan for r in rows]),
            series=np.array([r['series'].encode() for r in rows]),
            series_block=np.array([f"{r['series']}:b{r['block']}".encode() for r in rows])).items():
            handle.create_array('/run_metadata', name, value)
        handle.root._v_attrs.dataset_split = 'development'
        handle.root._v_attrs.detector_sha256 = DETECTOR_SHA
        handle.root._v_attrs.nominal_parameter_order = '[v_m_s,F_N_positive]'
        handle.root._v_attrs.historical_pilot_included = True
        handle.root._v_attrs.padding_requires_masked_training_loss = True


def run(output):
    if output.exists():
        raise FileExistsError('New bundle required; no overwriting')
    if sha(CODE / 'detector.py') != DETECTOR_SHA:
        raise ValueError('Detector changed')
    sys.path.insert(0, str(CODE))
    import convert_runs as convert
    from detector import PilotDetectorConfig, run_pilot_detector
    from sequence_config import SequenceConfig
    torch.set_num_threads(1)
    main_hdf = MAIN / 'full/train/main_doe_all.h5'
    summary = json.loads((MAIN / 'offline_export_summary_v1.json').read_text())
    expected = next(r['sha256'] for r in summary['outputs'] if Path(r['path']) == main_hdf)
    if sha(main_hdf) != expected:
        raise ValueError('Main export changed')
    main_rows = json.loads((MAIN / 'evaluation/labels.json').read_text())
    main_manifests = [json.loads((Path(json.loads((MAIN/'native_runs'/r['run_id']/'meta.json').read_text())['source']['raw_bag_dir'])/'run_manifest.json').read_text()) for r in main_rows]
    frames = main_manifests[0]['frames']
    arch = main_manifests[0]['architecture']
    if any(m['frames'] != frames or m['architecture'] != arch for m in main_manifests):
        raise ValueError('Main-series architecture/frame mismatch')
    original = json.loads((PILOT/'converted/conversion_manifest.json').read_text())
    ids = original['run_ids_in_global_tensor_order']
    if len(ids) != 27 or len(set(ids)) != 27:
        raise ValueError('Unexpected pilot inventory')
    entries, rows, audit = [], [], []
    for rid in ids:
        folder = PILOT/'exported_runs'/rid
        meta = json.loads((folder/'meta.json').read_text())
        raw = Path(meta['source']['raw_bag_dir'])
        manifest_path = raw/'run_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        annotation_path = PILOT_ANNOTATIONS/(rid+'_operator_annotation.json')
        annotation = json.loads(annotation_path.read_text())
        if sha(manifest_path) != meta['source']['manifest_sha256'] or sha(annotation_path) != meta['source']['operator_annotation_sha256']:
            raise ValueError('Pilot provenance changed: ' + rid)
        if manifest['outcome']['status'] != 'COMPLETED' or manifest['doe']['planned_split'] != 'development':
            raise ValueError('Not a completed Development run')
        if manifest['frames'] != frames or manifest['architecture'] != arch or meta['architecture'] != arch:
            raise ValueError('Pilot architecture/frame mismatch')
        if manifest['process']['slide_direction'] != '+x_R' or manifest['process']['slide_distance_m'] != .015:
            raise ValueError('Process mismatch')
        if manifest['ground_truth']['absolute_edge_x_R_m'] != .00925 or meta['r_gt_nom_mm'] != 9.25:
            raise ValueError('Reference value mismatch')
        reference = Path(manifest['ground_truth']['reference_file'])
        if sha(reference) != manifest['ground_truth']['reference_sha256']:
            raise ValueError('Historical reference changed')
        sources = {'operator_response_in_collection_session', 'operator_response_in_collection_session_typo_normalized'}
        if annotation['operator_observation']['hairpin_slip'] is not False or annotation['source'] not in sources or meta['operator_slip'] is not False:
            raise ValueError('Explicit pilot NOSLIP not established')
        if raw_hash(raw) != manifest['acquisition']['bag_sha256']:
            raise ValueError('Raw bag hash mismatch')
        entry, reason = convert.convert_run(str(folder), PilotDetectorConfig(), SequenceConfig())
        if entry is None:
            raise ValueError(reason)
        with np.load(folder/'series.npz', allow_pickle=False) as series:
            t, pos = series['t'], series['pos']
            d = pos[:, 0] - meta['x_r_start_m']
            det = run_pilot_detector(t, d, pos[:, 2], PilotDetectorConfig())
            pre = (d >= .001) & (d <= .007)
            actual = float(np.polyfit(t[pre], d[pre], 1)[0]*1000)
        error = float((det.r_det_m-entry['r_gt_run'])*1000) if entry['label']['triggered'] else None
        row = dict(run_id=rid, series='pilot_doe_v0', split='development', block=meta['block'],
            F_N=meta['f_n_mag_N'], v_nominal_mm_s=meta['v_slide_mm_s'], v_actual_pre_edge_mm_s=actual,
            triggered=bool(entry['label']['triggered']), success=int(entry['label']['success']),
            e_signed_mm=error, e_abs_mm=abs(error) if error is not None else None,
            t_det_native_s=float(det.t_det) if entry['label']['triggered'] else None,
            valid_length=entry['label']['end_idx']+1, slip_observation_confirmed=True,
            hairpin_id=None, historical_hairpin_identity='not independently established',
            ground_truth_reference_sha256=manifest['ground_truth']['reference_sha256'],
            baseline_method=meta['bias']['software_baseline_method'])
        rows.append(row); entries.append(entry)
        audit.append(dict(run_id=rid, frames_equal=True, architecture_equal=True, reference_nominal_equal=True,
            raw_hash_verified=True, manifest_sha256=sha(manifest_path), annotation_sha256=sha(annotation_path),
            source_native_meta_sha256=sha(folder/'meta.json'), source_native_series_sha256=sha(folder/'series.npz')))
    # Keep main CV allocation. Assign additional historical-only combinations
    # reproducibly; common combinations ALWAYS inherit the same fold.
    old_folds = json.loads((MAIN/'grouped_cv_membership_v1.json').read_text())['folds']
    by_id = {r['run_id']: r for r in main_rows}
    mapping = {(by_id[rid]['F_N'], by_id[rid]['v_nominal_mm_s']): fold['fold'] for fold in old_folds for rid in fold['heldout_run_ids']}
    new_combinations = sorted({(r['F_N'], r['v_nominal_mm_s']) for r in rows} - set(mapping))
    for index, position in enumerate(np.random.default_rng(20260918).permutation(len(new_combinations))):
        mapping[new_combinations[int(position)]] = index % 5 + 1
    all_rows = main_rows + rows
    folds = []
    for number in range(1, 6):
        held = [r['run_id'] for r in all_rows if mapping[(r['F_N'], r['v_nominal_mm_s'])] == number]
        train = [r['run_id'] for r in all_rows if r['run_id'] not in set(held)]
        folds.append(dict(fold=number, train_run_ids=train, heldout_run_ids=held,
            main_only_heldout_run_ids=old_folds[number-1]['heldout_run_ids']))
    output.mkdir(parents=True)
    pilot_hdf = output/'pilot_harmonized_640.h5'
    convert.write_h5(entries, str(pilot_hdf), fz_sign=1.)
    add_metadata(pilot_hdf, rows)
    bundle = output/'full/train/development_all.h5'
    bundle.parent.mkdir(parents=True)
    with tables.open_file(main_hdf) as first, tables.open_file(pilot_hdf) as second, tables.open_file(bundle, 'w') as target:
        for name in ('inputs', 'point_start', 'real', 'sim'):
            target.create_group('/', name)
            target.create_array('/'+name, '0', np.concatenate((first.get_node('/'+name+'/0').read(), second.get_node('/'+name+'/0').read())))
    add_metadata(bundle, all_rows)
    contrasts = []
    for speed in (2, 4, 6):
        groups = [[r for r in source if r['F_N'] == 2 and r['v_nominal_mm_s'] == speed] for source in (main_rows, rows)]
        contrasts.append(dict(F_N=2, v_nominal_mm_s=speed, main_n=len(groups[0]), pilot_n=len(groups[1]),
            main_mean_error_mm=float(np.mean([r['e_abs_mm'] for r in groups[0]])),
            pilot_mean_error_mm=float(np.mean([r['e_abs_mm'] for r in groups[1]])),
            main_mean_actual_speed_mm_s=float(np.mean([r['v_actual_pre_edge_mm_s'] for r in groups[0]])),
            pilot_mean_actual_speed_mm_s=float(np.mean([r['v_actual_pre_edge_mm_s'] for r in groups[1]]))))
    report = dict(schema='development_main90_pilot27_bundle_v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
        data_role='development_only', total_runs=117, main_runs=90, pilot_runs=27,
        unique_parameter_combinations=len(mapping), test_runs_included=0,
        common_anchor_comparison=contrasts, comparability='SUPPORTED_WITH_DOCUMENTED_HISTORICAL_LIMITATIONS',
        unchanged_inversion_bounds=dict(force_N=[1, 5], nominal_speed_mm_s=[1, 10]),
        historical_out_of_search_bounds_runs=sum(r['F_N'] > 5 for r in rows),
        limitations=['Historical Hairpin identity/reclamping equivalence not independently established',
            'Same nominal ground truth 9.25 mm, different reference provenance/caliper vs CAD-user reference',
            'Six pilot runs use documented baseline fallback windows',
            'Historical acquisition software differs; compare main-only vs augmented training on identical MAIN heldout runs',
            'Do not pool historical block 1 with main-series block 1',
            'Adding high-force data must demonstrate Development-only benefit; no extension of physical search bounds'],
        hdf_path=str(bundle), hdf_sha256=sha(bundle), pilot_harmonized_sha256=sha(pilot_hdf),
        source_main_hdf_sha256=sha(main_hdf), detector_sha256=DETECTOR_SHA,
        converter_code_sha256={path.name: sha(path) for path in CODE.glob('*.py')},
        raw_data_modified=False,
        original_native_exports_modified=False, pilot_run_audit=audit)
    write_json(output/'pilot_harmonized_labels.json', rows)
    write_json(output/'development_labels.json', all_rows)
    write_json(output/'grouped_cv_membership_v1.json', dict(data_role='development_only', seed=20260918, folds=folds,
        strategy='Preserve main folds; keep all historical duplicates of a combination together'))
    write_json(output/'development_bundle_manifest_v1.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'pilot_run_audit'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)

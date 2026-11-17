#!/usr/bin/env python3
"""Frozen detector labels, native continuous targets, masked SPI HDF5, grouped CV."""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tables
import torch

sys.path.insert(0, str(Path(__file__).parent / 'spi_bounds_v1'))
import convert_runs as convert
from detector import PilotDetectorConfig, run_pilot_detector
from sequence_config import SequenceConfig
from shadow_program_inversion.utils.io import load_data_file

DETECTOR_SHA = 'c729b577128b4870931e516c05e45ea84c9526e8753e6824e61886a55b3e7917'
SEED = 20260918
RESULTS_ROOT = Path(os.environ.get('HAIRPIN_MAIN_RESULTS_ROOT', 'results/phase_b_spi/main_doe_v1')).expanduser().resolve()
BAG_ROOT = Path(os.environ.get('HAIRPIN_MAIN_BAG_ROOT', 'bags/phase_b_spi/main_doe_v1')).expanduser().resolve()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def csv_intermediate(path, rows):
    """Flat machine-learning interchange, not a formatted workbook."""
    if not rows:
        return
    with path.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def labels_and_stats(entry, folder):
    meta, label = entry['meta'], entry['label']
    with np.load(folder / 'series.npz', allow_pickle=False) as a:
        t, pos, wrench = a['t'], a['pos'], a['wrench']
        d = pos[:, 0] - meta['x_r_start_m']
        det = run_pilot_detector(t, d, pos[:, 2], PilotDetectorConfig())
        pre = (d >= .001) & (d <= .007)
        v_actual = float(np.polyfit(t[pre], d[pre], 1)[0] * 1000) if pre.sum() >= 5 else None
        triggered = label['triggered']
        error = (det.r_det_m - entry['r_gt_run']) * 1000 if triggered else None
        ati_mean = ati_std = ati_median = None
        ati = meta.get('ati', {})
        if ati.get('status') == 'ok':
            with np.load(folder / 'ati_series.npz', allow_pickle=False) as a_ati:
                da = a_ati['pos'][:, 0] - meta['x_r_start_m']
                ap = (da >= .001) & (da <= .007)
                if ap.sum() >= 5:
                    af = a_ati['wrench'][ap, 2]
                    ati_mean, ati_std, ati_median = float(af.mean()), float(af.std()), float(np.median(af))
        n = int(label['end_idx']) + 1
        mask = np.arange(entry['sequence'].horizon_samples) < n
        entry['valid_length'], entry['physical_sample_mask'] = n, mask
        row = dict(run_id=meta['run_id'], logical_run_id=meta['logical_run_id'], attempt_index=meta['attempt_index'],
                   sequence_index=meta['sequence_index'], series='main_doe_v1', split='development',
                   combination_id=meta['combination_id'], block=meta['block'], order_in_block=meta['order_in_block'],
                   actual_start_utc=meta.get('actual_start_utc'),
                   manifest_created_at_utc=meta.get('manifest_created_at_utc', meta.get('actual_start_utc')),
                   hairpin_id=meta['hairpin_id'], setup_id=meta['setup_id'],
                   prior_reuse_count=meta['prior_reuse_count'], F_N=meta['f_n_mag_N'], force_command_N=meta['force_command_N'],
                   v_nominal_mm_s=meta['v_slide_mm_s'], v_actual_pre_edge_mm_s=v_actual,
                   technical_status='COMPLETED', stop_reason=None, operator_slip=meta['operator_slip'],
                   slip_basis=meta['slip_basis'], slip_observation_confirmed=meta['slip_observation_confirmed'],
                   slip_source=meta['slip_source'], contact_status=meta['contact_status'], triggered=bool(triggered),
                   t_det_native_s=float(det.t_det) if triggered else None,
                   t_cross_native_s=float(det.t_cross) if triggered else None,
                   r_det_relative_mm=float(det.r_det_m * 1000) if triggered else None,
                   x_det_absolute_mm=float((det.r_det_m + meta['x_r_start_m']) * 1000) if triggered else None,
                   r_GT_run_mm=float(entry['r_gt_run'] * 1000), e_signed_mm=error,
                   e_abs_mm=abs(error) if error is not None else None, success=int(label['success']),
                   success_conditional_on_assumed_noslip=meta['slip_basis'] == 'assumed',
                   eos_reason='native_detector_trigger' if triggered else '15mm_cap_without_valid_trigger',
                   eos_grid_index=int(label['end_idx']), eos_grid_s=float(label['end_idx'] * .032), valid_length=n,
                   observed_slide_duration_s=float(t[-1]),
                   internal_Fz_pre_edge_mean_N=float(wrench[pre, 2].mean()) if pre.any() else None,
                   internal_Fz_pre_edge_std_N=float(wrench[pre, 2].std()) if pre.any() else None,
                   internal_Fz_pre_edge_median_N=float(np.median(wrench[pre, 2])) if pre.any() else None,
                   ATI_Fz_pre_edge_mean_N=ati_mean, ATI_Fz_pre_edge_std_N=ati_std, ATI_Fz_pre_edge_median_N=ati_median,
                   ATI_baseline_Fz_std_N=ati.get('baseline_Fz_std_N'), ATI_reference_status=ati.get('status'),
                   force_tracking_warning=meta['qc']['force_tracking_warning'],
                   recorder_transport_lost_messages_aggregate=meta.get('recorder_transport_lost_messages_aggregate'),
                   endpoint_policy_version=meta.get('endpoint_policy_version'), model_eligible=True,
                   exclusion_reason=None, manifest_sha256=meta['source']['manifest_sha256'],
                   raw_bag_sha256=meta['source']['collector_bag_sha256'],
                   annotation_sha256=meta['source']['operator_annotation_sha256'])
        entry['native_label'] = row
        return row


def grouped_folds(rows):
    combinations = sorted({(r['F_N'], r['v_nominal_mm_s']) for r in rows})
    order = np.random.default_rng(SEED).permutation(len(combinations))
    mapping = {combinations[int(i)]: k % 5 + 1 for k, i in enumerate(order)}
    folds = []
    for fold in range(1, 6):
        held = [i for i, r in enumerate(rows) if mapping[(r['F_N'], r['v_nominal_mm_s'])] == fold]
        train = [i for i in range(len(rows)) if i not in set(held)]
        folds.append(dict(fold=fold, train_indices=train, heldout_indices=held,
                          train_run_ids=[rows[i]['run_id'] for i in train], heldout_run_ids=[rows[i]['run_id'] for i in held]))
    return folds


def save_h5(entries, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    convert.write_h5(entries, str(path), fz_sign=1.0)
    with tables.open_file(path, 'a') as f:
        f.create_group('/', 'valid_length')
        f.create_array('/valid_length', '0', np.asarray([e['valid_length'] for e in entries], dtype=np.int32))
        f.create_group('/', 'physical_sample_mask')
        f.create_array('/physical_sample_mask', '0', np.stack([e['physical_sample_mask'] for e in entries]))
        f.create_group('/', 'run_metadata')
        f.create_array('/run_metadata', 'run_id', np.asarray([e['meta']['run_id'].encode() for e in entries]))
        f.create_array('/run_metadata', 'slip_observation_confirmed', np.asarray([e['meta']['slip_observation_confirmed'] for e in entries]))
        f.create_array('/run_metadata', 'native_e_signed_mm', np.asarray([e['native_label']['e_signed_mm'] if e['native_label']['e_signed_mm'] is not None else np.nan for e in entries]))
        f.create_array('/run_metadata', 'native_t_det_s', np.asarray([e['native_label']['t_det_native_s'] if e['native_label']['t_det_native_s'] is not None else np.nan for e in entries]))
        f.root._v_attrs.dataset_split = 'development'
        f.root._v_attrs.nominal_parameter_order = '[v_m_s,F_N_positive]'
        f.root._v_attrs.detector_sha256 = DETECTOR_SHA
        f.root._v_attrs.padding_requires_masked_training_loss = True
    x, starts, sim, real = load_data_file(str(path))
    expected = (len(entries), 640, 15)
    assert tuple(real.shape) == expected and tuple(sim.shape) == expected
    assert tuple(x.shape) == (len(entries), 2) and tuple(starts.shape) == (len(entries), 7)
    assert all(torch.isfinite(a).all() for a in (x, starts, sim, real))
    for e in entries:
        n, tr = e['valid_length'], e['traj']
        assert e['physical_sample_mask'].sum() == n
        assert np.all(tr[:n-1, 0] == 0) and np.all(tr[n-1:, 0] == 1)
        if n < 640:
            assert np.all(tr[n:] == tr[n-1])
    return dict(path=str(path), n_runs=len(entries), shape=list(expected), sha256=convert.sha256_file(path),
                loader_verified=True, mask_and_EOS_verified=True)


def factor_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['F_N'], row['v_nominal_mm_s'])].append(row)
    result = []
    for (force, speed), g in sorted(groups.items()):
        errors = [r['e_abs_mm'] for r in g if r['e_abs_mm'] is not None]
        result.append(dict(F_N=force, v_nominal_mm_s=speed, n_runs=len(g),
            n_triggered=sum(r['triggered'] for r in g), n_missing_trigger=sum(not r['triggered'] for r in g),
            n_success=sum(r['success'] for r in g), success_fraction=sum(r['success'] for r in g)/len(g),
            n_observed_noslip=sum(r['slip_observation_confirmed'] for r in g),
            n_assumed_noslip=sum(not r['slip_observation_confirmed'] for r in g),
            e_abs_mean_mm=float(np.mean(errors)) if errors else None,
            e_abs_std_sample_mm=float(np.std(errors, ddof=1)) if len(errors)>1 else None,
            v_actual_mean_mm_s=float(np.mean([r['v_actual_pre_edge_mm_s'] for r in g if r['v_actual_pre_edge_mm_s'] is not None]))))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', type=Path)
    args = p.parse_args()
    root = args.dataset
    detector = Path(__file__).parent / 'spi_bounds_v1' / 'detector.py'
    if convert.sha256_file(detector) != DETECTOR_SHA:
        raise ValueError('Frozen detector changed; export refused')
    native_manifest = root / 'native_reconciliation_manifest_v1.json'
    if not native_manifest.exists():
        native_manifest = root / 'native_export_manifest.json'
    native = json.loads(native_manifest.read_text())
    progress_path = RESULTS_ROOT / 'progress_after_r090_resume_v7.json'
    progress = json.loads(progress_path.read_text())
    attempt_register = []
    for rid in progress['aborted_run_ids']:
        base = progress_path.parent
        execution_path = base / f'{rid}_execution_v1.json'
        bag_path = BAG_ROOT / rid
        execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
        attempt_register.append(dict(run_id=rid, acquisition_status='TECHNICALLY_ABORTED_OR_INTERRUPTED',
            model_eligible=False, success=None, execution_path=str(execution_path) if execution is not None else None,
            raw_bag_directory=str(bag_path) if bag_path.exists() else None, original_execution=execution))
    assert len(attempt_register) == progress['technical_aborted_attempts'] == 10
    write_json(root / 'technical_attempt_register_v1.json', attempt_register)
    write_json(root / 'source_final_progress_snapshot.json', progress)
    entries, rows, excluded, regression = [], [], list(native['excluded_runs']), []
    evaluation = root / 'evaluation'
    evaluation.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    for r in native['exported_runs']:
        folder = root / 'native_runs' / r['run_id']
        try:
            meta_path = folder / 'meta.json'
            meta = json.loads(meta_path.read_text())
            if 'manifest_created_at_utc' not in meta:
                meta['manifest_created_at_utc'] = meta.pop('actual_start_utc', None)
                meta['actual_start_utc'] = None
                meta['actual_start_time_note'] = 'Metadata creation is not an independently exported motion-start timestamp'
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            entry, reason = convert.convert_run(str(folder), PilotDetectorConfig(), SequenceConfig())
            if entry is None:
                raise ValueError(reason)
            row = labels_and_stats(entry, folder)
            if row['sequence_index'] <= 3:
                oldpath = RESULTS_ROOT / f"{row['run_id']}_quick_analysis_v1.json"
                old = json.loads(oldpath.read_text())['detector']
                delta = abs(row['e_signed_mm'] - old['signed_localization_error_mm'])
                assert delta < 1e-7, f'Native detector regression mismatch: {delta} mm'
                regression.append(dict(run_id=row['run_id'], signed_error_difference_mm=delta, passed=True))
            entries.append(entry)
            rows.append(row)
            write_json(evaluation / f"{row['run_id']}_offline_analysis_v1.json", row)
        except Exception as error:
            excluded.append(dict(run_id=r['run_id'], sequence_index=r['sequence_index'], reason=f'{type(error).__name__}: {error}', success=None))
    if not entries:
        write_json(evaluation / 'excluded_runs.json', excluded)
        raise ValueError('No eligible runs; no tensors fabricated')
    folds = grouped_folds(rows)
    lobo = []
    for block in (1, 2, 3):
        held = [i for i, r in enumerate(rows) if r['block'] == block]
        train = [i for i, r in enumerate(rows) if r['block'] != block]
        lobo.append(dict(heldout_block=block, train_indices=train, heldout_indices=held,
                         train_run_ids=[rows[i]['run_id'] for i in train], heldout_run_ids=[rows[i]['run_id'] for i in held]))
    for fold in folds:
        a = {(rows[i]['F_N'], rows[i]['v_nominal_mm_s']) for i in fold['train_indices']}
        b = {(rows[i]['F_N'], rows[i]['v_nominal_mm_s']) for i in fold['heldout_indices']}
        assert not a & b
        assert not set(fold['train_run_ids']) & set(fold['heldout_run_ids'])
    csv_intermediate(evaluation / 'labels.csv', rows)
    factors = factor_summary(rows)
    csv_intermediate(evaluation / 'empirical_factor_summary.csv', factors)
    write_json(evaluation / 'labels.json', rows)
    write_json(evaluation / 'empirical_factor_summary.json', factors)
    write_json(evaluation / 'excluded_runs.json', excluded)
    write_json(root / 'grouped_cv_membership_v1.json', dict(seed=SEED, grouping='exact nominal (F_N,v_mm_s)', data_role='development_only', folds=folds))
    write_json(root / 'lobo_membership_v1.json', dict(data_role='development_only', folds=lobo))
    outputs = [save_h5(entries, root / 'full' / 'train' / 'main_doe_all.h5')]
    for fold in folds:
        for split, key in [('train', 'train_indices'), ('heldout', 'heldout_indices')]:
            outputs.append(save_h5([entries[i] for i in fold[key]], root / 'grouped_cv' / f"fold{fold['fold']}" / split / f'{split}.h5'))
    for fold in lobo:
        for split, key in [('train', 'train_indices'), ('heldout', 'heldout_indices')]:
            outputs.append(save_h5([entries[i] for i in fold[key]], root / 'lobo' / f"fold{fold['heldout_block']}" / split / f'{split}.h5'))
    confirmed = [r for r in rows if r['slip_observation_confirmed']]
    assumed = [r for r in rows if not r['slip_observation_confirmed']]
    summary = dict(schema_version='main90_offline_export_summary_v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
                   n_acquired=90, n_model_eligible=len(rows),
                   n_excluded=len(excluded), n_triggered=sum(r['triggered'] for r in rows),
                   n_missing_trigger=sum(not r['triggered'] for r in rows), n_success=sum(r['success'] for r in rows),
                   n_failure=sum(not r['success'] for r in rows),
                   observed_noslip=dict(n=len(confirmed), success=sum(r['success'] for r in confirmed)),
                   assumed_noslip=dict(n=len(assumed), conditional_success=sum(r['success'] for r in assumed)),
                   detector_sha256=DETECTOR_SHA, detector_config=asdict(PilotDetectorConfig()),
                   success_localization_tolerance_mm=1.5, raw_data_modified=False, robot_motion_executed=False,
                   native_regression_tests=regression, group_leakage_tests_passed=True, outputs=outputs,
                   sequence=SequenceConfig().as_dict(), nominal_force_range_N=[1,5], nominal_speed_range_mm_s=[1,10],
                   original_pilot_included=False, independent_model_test_runs=0, physical_end_validation_runs=0,
                   tensor_export_complete=True, production_training_ready=False,
                   ancillary_context_reconciliations=native.get('reconciled_runs', []),
                   training_prerequisites=['Use physical_sample_mask and per-run-normalized loss',
                       'Implement/test continuous native-error quality output and frozen inversion objective if desired',
                       'Do not describe assumed NOSLIP as an observed process outcome',
                       'Low-force contact qualification remains unresolved'],
                   missing_native_error_representation='JSON null / CSV blank / auxiliary HDF NaN with trigger flag; never invented zero')
    write_json(root / 'offline_export_summary_v1.json', summary)
    write_json(root / 'source_code_hashes.json', {str(f.relative_to(Path(__file__).parent)): convert.sha256_file(f)
                 for f in sorted(Path(__file__).parent.rglob('*.py'))})
    text = f'''# Phase B – Hauptserie main_doe_v1: Offline-Export v1

{len(rows)} echte Hauptserienläufe exportiert; {len(excluded)} Auswertungsausschlüsse. Alle Development, kein unabhängiger Modelltest.

## Dateien

- `full/train/main_doe_all.h5`: SPI-kompatible Inputs [N,2], Start [N,7], real/sim [N,640,15].
- `evaluation/labels.csv` und `labels.json`: native Detektionszeit, relativer/absoluter Ort, vorzeichenbehafteter/absoluter Fehler, Success, tatsächliche Geschwindigkeit, interne und passive ATI-Kraftstatistik, Provenienz.
- `evaluation/empirical_factor_summary.csv`: aggregierte Ergebnisse je Parameterkombination, einschließlich fehlender Trigger und Annahmenanzahl.
- `native_runs/<run_id>/`: debiaste interne Messserie und getrennte passive ATI-Referenz, Original-Hashes/Metadaten.
- `grouped_cv/fold1..5`: Training/Heldout nach exakter nominaler (F,v)-Kombination, alle Wiederholungen derselben Kombination im selben Fold. Development-interne Modellwahl, KEIN unberührter Modelltest.
- `lobo/fold1..3`: zusätzliche Leave-one-block-out-Prüfung der Hauptserie.
- `source_acquisition_inventory.json`: alle 90 gültigen Aufnahmen und zehn historische technische Abbruchversuche als inventarisierte Anzahl; Abbrüche werden nicht als physische Failure-Trainingsläufe behandelt.

## Eingefrorene Definition

Internes UR-F/T regelt, ATI ausschließlich passive Referenz. Detektor unverändert ({DETECTOR_SHA}), native TCP-z_R vor Resampling: Baseline bis 7,5 mm, EMA 32 ms, Abfall 0,075 mm, Persistenz 32 ms.
Success = vollständiger Lauf, gültiger Trigger bis zum 15-mm-Cap, absoluter nativer Ortsfehler <= 1,5 mm und NOSLIP-Befund. {len(confirmed)} Befunde beobachtet; {len(assumed)} ausdrücklich gemäß Nutzerregel angenommen. Success dieser zweiten Gruppe ist bedingt auf diese Annahme, keine beobachtete Slip-Freiheit. Kein Zertifikat für stabilen Kontakt oder vollständig erfolgreiches Setzen.
Referenz: absolute Fase x_R=9,25 mm, je Lauf um tatsächlichen Start versetzt. Maßgeblich ist die je Run gehashte Referenz rgt_cad_user_reference_v2: nutzerangegebener CAD-Nennwert mit manueller Plausibilitätsprüfung; CAD-Revision und Messunsicherheit nicht unabhängig verifiziert. Historische Messschieberauflösung 0,05 mm ist keine bestimmte Messunsicherheit.
EOS ist der virtuelle Detektionsendpunkt; alle physischen Läufe fuhren die volle 15-mm-Bahn. Trigger-EOS erstes Raster bei/nach nativer Detektion, sonst letztes Raster bei/vor Cap. dt=0,032 s, T=640, letzter Zeitstempel 20,448 s. Padding wiederholt das Endsample.

## Vor Training beachten

HDF `/valid_length/0` und `/physical_sample_mask/0` unterscheiden reale Rasterwerte einschließlich EOS von Padding. Der bisherige Standard-SPI-Loader liest diese Zusatzgruppen NICHT automatisch: Trainingsadapter und Verlust müssen die Maske explizit lesen, pro Lauf normieren und getestet werden. Export ist abgeschlossen; produktive Trainingsfreigabe bleibt bis dahin offen.
Kanalfolge: EOS, Success, px, py, pz, qw, qx, qy, qz, interner Fx, Fy, Fz, Mx, My, Mz. Inputs sind nominal [v in m/s, positiver F_N-Betrag], tatsächliche Geschwindigkeit separat; sie wird nicht rückwirkend zum Sollparameter. Wrench-Achsen ins Sliding Frame rotiert, Drehmomente ohne Hebelarmverschiebung am jeweiligen Sensor-/TCP-Ursprung. ATI ist kein Modellinput. Interne Wrench für Raster: Butterworth 4. Ordnung, 12 Hz, zero-phase; Detektor selbst hiervon unbeeinflusst.
Native Fehler fehlen ohne Trigger (null/leeres Feld/auxiliäres NaN), nicht 0. kontinuierliche Güte-Zielfunktion und Gewichte sind hier NICHT neu festgelegt oder trainiert. Analytischer `sim`-Prior sind keine zusätzlichen realen Versuche.
Historische Pilotdaten 27 separat belassen; keine unbestätigte Vergleichbarkeit vorausgesetzt. Unabhängiger Modelltest 30 und Endvalidierung 60 noch nicht aufgenommen. Kernel-Neustart während Block 2 und Endpoint-Policy-Änderungen sind in originalen Run-Kontexten nachvollziehbar und potentielle Störeinflüsse. Gleiches Hairpin-ID 0; unbekannte Wiederverwendungszahl bleibt unbekannt. `manifest_created_at_utc` bezeichnet Metadatenerstellung, nicht den Bewegungsbeginn; `actual_start_utc` bleibt hier unbestimmt. Die native Zeitbasis ist exakt relativ zum Ereignis slide_started.

## Integrität

Rohdaten und historische Auswertungen unverändert. MCAP-Hashes geprüft, Loader-Tensoren, EOS/Padding/Masken und kombinierungsfreie CV-Folds getestet. Die ersten drei nativen Detektorresultate gegen vorhandene Auswertungen geprüft. Ausschlüsse und passive ATI-Exportprobleme bleiben offen sichtbar. Fehlende ergänzende Run-Kontextdateien dürfen ausschließlich durch exakten Abgleich mit eingefrorenem Plan, vollständigem Manifest und Operator-Sidecar ergänzt werden; fehlende Messwerte werden niemals ergänzt. Jede solche Auflösung ist in `native_reconciliation_manifest_v1.json` und den jeweiligen Metadaten ausgewiesen. Kein Ausschluss allein wegen Verlustzähler, keiner wegen einer frei erfundenen 10-ms-Grenze; keine interne/ATI-Korrelation als Voraussetzung.
Siehe `offline_export_summary_v1.json`, `evaluation/excluded_runs.json`, Quellcode-Hashes und den archivierten Offline-Exporter. Kein Training, keine Inversion und keine Roboterbewegung durch diesen Export.
'''
    with (root / 'README.md').open('x', encoding='utf-8') as f:
        f.write(text)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('outputs','detector_config')}, indent=2), flush=True)


if __name__ == '__main__':
    main()

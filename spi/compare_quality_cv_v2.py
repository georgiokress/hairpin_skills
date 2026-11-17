"""Paired Development-only pilot-benefit study of the independent quality head."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data import frozen_fold, load_development, select, sha
from model import MaskedQualityGRU, Scaling

BASE = Path(os.environ.get('HAIRPIN_DATASET_ROOT', 'datasets/phase_b_spi')).expanduser().resolve()
MAIN = BASE/'main_doe_v1_spi_v1'
AUGMENTED = BASE/'development_main90_pilot27_v1'
SEEDS = [20260918, 20260919, 20260920]
STEPS = 4000


def store(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def quality_features(model, data):
    return torch.cat((model.scaling.inputs(data['inputs']),
        Scaling.scale(data['point_start'], model.scaling.output_limits[:, :7])), -1)


def polynomial(x):
    return np.column_stack((np.ones(len(x)), x[:, 0], x[:, 1], x[:, 0]**2,
                            x[:, 0]*x[:, 1], x[:, 1]**2))


def ridge_fit_predict(train_x, train_y, held_x):
    mean, scale = train_x.mean(0), train_x.std(0)
    scale = np.where(scale == 0, 1, scale)
    design = polynomial((train_x-mean)/scale)
    penalty = np.diag([0., .001, .001, .001, .001, .001])
    coefficients = np.linalg.solve(design.T@design + penalty, design.T@train_y)
    prediction = np.maximum(0, polynomial((held_x-mean)/scale)@coefficients)
    return prediction, dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=coefficients.tolist())


def metrics(y, prediction):
    diff = np.asarray(prediction)-np.asarray(y)
    return dict(n=len(diff), mae_mm=float(np.abs(diff).mean()),
                rmse_mm=float(np.sqrt(np.square(diff).mean())), bias_mm=float(diff.mean()))


def validate_sources():
    main_source = MAIN/'full/train/main_doe_all.h5'
    augmented_source = AUGMENTED/'full/train/development_all.h5'
    original = json.loads((MAIN/'offline_export_summary_v1.json').read_text())
    expected = next(r['sha256'] for r in original['outputs'] if Path(r['path']) == main_source)
    combined = json.loads((AUGMENTED/'development_bundle_manifest_v1.json').read_text())
    if sha(main_source) != expected or sha(augmented_source) != combined['hdf_sha256']:
        raise ValueError('Frozen data changed')
    main = load_development(main_source)
    augmented = load_development(augmented_source)
    if len(main['run_ids']) != 90 or len(augmented['run_ids']) != 117:
        raise ValueError('Unexpected inventory')
    lookup = {rid:i for i,rid in enumerate(augmented['run_ids'])}
    duplicate = select(augmented, [lookup[rid] for rid in main['run_ids']])
    for key in main:
        if key != 'run_ids' and not torch.equal(main[key], duplicate[key]):
            raise ValueError('Main observations altered inside combined export: '+key)
    if not main['error_valid'].all() or not augmented['error_valid'].all():
        raise ValueError('This frozen comparison requires available native errors; revise protocol for missing detections')
    sources = dict(main_hdf_sha256=sha(main_source), augmented_hdf_sha256=sha(augmented_source),
        main_membership_sha256=sha(MAIN/'grouped_cv_membership_v1.json'),
        augmented_membership_sha256=sha(AUGMENTED/'grouped_cv_membership_v1.json'))
    return main, augmented, sources


def run(output):
    if output.exists():
        raise FileExistsError('Preserve existing study; use a new output directory')
    main, augmented, sources = validate_sources()
    folds = []
    for number in range(1,6):
        train, held = frozen_fold(main, MAIN/'grouped_cv_membership_v1.json', number)
        extra, extra_held = frozen_fold(augmented, AUGMENTED/'grouped_cv_membership_v1.json', number)
        if not set(held['run_ids']) <= set(extra_held['run_ids']):
            raise ValueError('Paired main heldout membership differs')
        if not set(train['run_ids']) <= set(extra['run_ids']):
            raise ValueError('Augmented variant loses main training runs')
        folds.append((number, train, extra, held))
    output.mkdir(parents=True)
    protocol = dict(schema='paired_pilot_benefit_quality_cv_v2', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        development_adaptation='After inspecting v1 training curves and Development CV results: same variants/seeds/heldout, fixed 4000 rather than 1000 updates; no independent-test access',
        scope='Native absolute localization error surrogate ONLY; not full trajectory/SPI validation',
        dataset_variants=['main_only', 'main_plus_pilot'], folds=5, seeds=SEEDS,
        evaluation='Exactly the same 18 MAIN runs per fold, all three replicates of each combination held together',
        primary_metric='Mean of seed-specific out-of-fold MAE predicting native |e_r| in mm',
        secondary_metrics=['RMSE', 'prediction bias', 'per-fold/per-force MAE', 'observed/assumed NOSLIP subsets'],
        architecture='Existing separate 9-input,32-hidden,SELU,1-output,Softplus quality MLP',
        inputs='Commanded nominal v,F and known start pose; no target EOS/length/trajectory inputs',
        optimizer=dict(name='Adam', learning_rate=.001, weight_decay=1e-5,
            full_batch=True, steps=STEPS, same_number_of_updates_per_variant=True),
        loss='Mean squared error in mm^2 on native absolute-error targets',
        normalization='Fit on each training fold only, matching frozen Scaling implementation',
        epoch_selection='None: fixed preliminary budget, no early stopping/checkpoint selection from heldout',
        training_budget_note='4000 quality-only full-batch steps for BOTH variants; Development-only budget sensitivity experiment, not an independent confirmation',
        baselines=['Training-target mean', 'Quadratic ridge on nominal v,F; alpha=.001, nonnegative output'],
        uncertainty='Descriptive paired combination-cluster bootstrap; no formal independent-CV confidence claim',
        final_checkpoint_selected=False, independent_test_runs_read=0, robot_motion=False,
        inversion_bounds_changed=False, source_hashes=sources,
        code_hashes={name:sha(Path(__file__).parent/name) for name in ('data.py','model.py',Path(__file__).name)})
    store(output/'protocol_frozen_before_training.json', protocol)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    predictions, histories = [], []
    for number, main_train, extra_train, held in folds:
        for variant, train in [('main_only',main_train), ('main_plus_pilot',extra_train)]:
            y = train['error_mm'].numpy()
            baseline = np.full(len(held['run_ids']), y.mean())
            ridge, ridge_state = ridge_fit_predict(train['inputs'].numpy(), y, held['inputs'].numpy())
            store(output/f'fold{number}_{variant}_baselines.json', dict(training_mean_mm=float(y.mean()), ridge=ridge_state))
            for method, values in [('mean', baseline), ('quadratic_ridge',ridge)]:
                for i,rid in enumerate(held['run_ids']):
                    predictions.append(dict(run_id=rid, fold=number, variant=variant, model=method, seed=None,
                        target_mm=float(held['error_mm'][i]), predicted_mm=float(values[i])))
            for seed in SEEDS:
                torch.manual_seed(seed)
                model = MaskedQualityGRU(Scaling.fit(train), hidden=32, layers=2, dropout=.1)
                x = quality_features(model, train)
                optimizer = torch.optim.Adam(model.quality.parameters(), lr=.001, weight_decay=1e-5)
                history=[]
                for step in range(1,STEPS+1):
                    optimizer.zero_grad()
                    value = model.quality(x).squeeze(-1)
                    loss = torch.nn.functional.mse_loss(value, train['error_mm'])
                    loss.backward()
                    if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.quality.parameters()):
                        raise RuntimeError('Nonfinite quality training')
                    optimizer.step()
                    if step in (1,10,100,250,500,1000,2000,3000,4000):
                        history.append(dict(step=step, training_mse_mm2=float(loss.detach())))
                with torch.no_grad():
                    estimate = model.quality(quality_features(model, held)).squeeze(-1).numpy()
                if not np.isfinite(estimate).all():
                    raise RuntimeError('Nonfinite heldout prediction')
                checkpoint = output/f'fold{number}_{variant}_seed{seed}_QUALITY_ONLY.pt'
                torch.save(dict(schema='quality_head_cv_only', final_model=False,
                    quality_state=model.quality.state_dict(), scaling_state=model.scaling.state_dict(),
                    protocol_sha256=sha(output/'protocol_frozen_before_training.json'),
                    train_run_ids=train['run_ids'], heldout_run_ids=held['run_ids'], seed=seed,
                    hidden=32, nominal_inversion_bounds=dict(v_mm_s=[1,10], force_N=[1,5])), checkpoint)
                histories.append(dict(fold=number,variant=variant,seed=seed,history=history,
                    train_count=len(train['run_ids']),heldout_main_count=18,checkpoint_sha256=sha(checkpoint)))
                for i,rid in enumerate(held['run_ids']):
                    predictions.append(dict(run_id=rid,fold=number,variant=variant,model='quality_mlp',seed=seed,
                        target_mm=float(held['error_mm'][i]),predicted_mm=float(estimate[i])))
                print(f'QUALITY_CV completed fold {number}/5 {variant} seed {seed}',flush=True)
    store(output/'training_histories.json',histories)
    store(output/'out_of_fold_predictions.json',predictions)
    summarize(output, main, predictions)


def summarize(output, main, predictions):
    results=[]
    for method in ('mean','quadratic_ridge','quality_mlp'):
        for variant in ('main_only','main_plus_pilot'):
            seed_metrics=[]
            for seed in (SEEDS if method=='quality_mlp' else [None]):
                rows=[p for p in predictions if p['model']==method and p['variant']==variant and p['seed']==seed]
                if len(rows)!=90 or len({r['run_id'] for r in rows})!=90:
                    raise ValueError('OOF inventory invalid')
                seed_metrics.append(dict(seed=seed,**metrics([r['target_mm'] for r in rows],[r['predicted_mm'] for r in rows])))
            results.append(dict(model=method,variant=variant,seed_metrics=seed_metrics,
                mean_mae_mm=float(np.mean([r['mae_mm'] for r in seed_metrics])),
                mean_rmse_mm=float(np.mean([r['rmse_mm'] for r in seed_metrics]))))
    errors={}
    for variant in ('main_only','main_plus_pilot'):
        errors[variant]={rid:float(np.mean([abs(r['predicted_mm']-r['target_mm']) for r in predictions
            if r['model']=='quality_mlp' and r['variant']==variant and r['run_id']==rid])) for rid in main['run_ids']}
    deltas=np.array([errors['main_plus_pilot'][rid]-errors['main_only'][rid] for rid in main['run_ids']])
    combos=np.asarray(main['inputs'])
    unique=np.unique(combos,axis=0)
    grouped=np.array([deltas[np.all(combos==c,axis=1)].mean() for c in unique])
    rng=np.random.default_rng(20260918)
    bootstrap=grouped[rng.integers(0,len(grouped),size=(10000,len(grouped)))].mean(1)
    per_force=[dict(force_N=float(force),mean_paired_MAE_delta_mm=float(deltas[combos[:,1]==force].mean()))
               for force in np.unique(combos[:,1])]
    fold_lookup={r['run_id']:r['fold'] for r in predictions}
    per_fold=[dict(fold=fold,mean_paired_MAE_delta_mm=float(np.mean([deltas[i] for i,rid in enumerate(main['run_ids']) if fold_lookup[rid]==fold]))) for fold in range(1,6)]
    slip_subsets=[]
    for observed in (True,False):
        indices=np.flatnonzero(main['slip_observed'].numpy()==observed)
        slip_subsets.append(dict(slip_observation_confirmed=observed,n=len(indices),
            mean_paired_MAE_delta_mm=float(deltas[indices].mean()),
            main_only_mae_mm=float(np.mean([errors['main_only'][main['run_ids'][i]] for i in indices])),
            main_plus_pilot_mae_mm=float(np.mean([errors['main_plus_pilot'][main['run_ids'][i]] for i in indices]))))
    report=dict(schema='paired_pilot_benefit_quality_result_v2',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Quality-only Development CV, not complete SPI or independent model test',
        independent_main_oof_runs=90, parameter_groups=30, training_fits=30,
        independent_test_runs_read=0, results=results,
        paired_quality_MAE_delta_mm=float(deltas.mean()), delta_direction='negative favors adding pilot',
        descriptive_cluster_bootstrap_interval_mm=np.quantile(bootstrap,[.025,.975]).tolist(),
        uncertainty_caveat='Overlapping CV training folds and one setup: descriptive interval, not a confirmatory confidence interval',
        groups_improved=int((grouped<0).sum()), groups_total=len(grouped),per_force=per_force,
        per_fold=per_fold,slip_provenance_subsets=slip_subsets,
        final_model_frozen=False,inversion_performed=False,full_trajectory_trained=False)
    store(output/'comparison_summary.json',report)
    lines=['# Pilot-Ergänzung: Development-Vergleich des Ortsfehler-Ausgangs','',
        'Fünf gruppierte Folds, drei feste Seeds. Je Fold identische 18 zurückgehaltene Hauptserienläufe. '
        'Vorhersageziel: absoluter nativer Detektionsortsfehler in mm. MAE hier ist der Fehler dieser Vorhersage, nicht der Detektionsfehler selbst.','',
        '| Modell | Trainingsbestand | mittlere Vorhersage-MAE [mm] | mittlere RMSE [mm] |',
        '| --- | --- | ---: | ---: |']
    for r in results:
        lines.append(f"| {r['model']} | {r['variant']} | {r['mean_mae_mm']:.4f} | {r['mean_rmse_mm']:.4f} |")
    lines += ['',f"Gepaarte MAE-Differenz (mit Pilot minus ohne): {deltas.mean():+.4f} mm. "
        f"{(grouped<0).sum()}/30 Kombinationen profitieren beim seedgemittelten absoluten Vorhersagefehler.",
        '', 'Das Verfahren und das feste 4000-Schritte-Budget wurden vor diesem Training gespeichert. '
        'Development-basierte Folgeprüfung nach v1 (1000 Schritte), keine unabhängige Bestätigung. '
        'Keine Epochenwahl anhand der zurückgehaltenen Läufe. Normierung pro Trainingsfold. '
        'Historische Duplikate einer Kombination liegen vollständig außerhalb des entsprechenden Trainingsfolds.',
        '', 'Die drei Seeds sind keine zusätzlichen physischen Messungen. Bootstrap auf 30 Parametergruppen ist '
        'nur deskriptiv: CV-Trainingsmengen überlappen. Noch kein Generalisierungsnachweis am unabhängigen Testsatz.',
        '', 'Nur der separate Qualitätszweig wurde trainiert. EOS, Success, Trajektoriennetz und Inversionsgüte '
        'sind hier nicht validiert. Das Ergebnis allein entscheidet nicht über das gesamte SPI-Modell. '
        'Kein finales Refit, keine Inversion, keine Roboterbewegung.']
    (output/'Vergleich_Pilot_Ortsfehler.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    run(args.output)

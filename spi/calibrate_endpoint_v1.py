"""Train-fold-only EOS threshold selection, evaluated on existing MAIN OOF runs.

This is a Development adaptation after inspecting previous EOS bias. It does
not probability-calibrate scores or establish untouched-test performance.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import torch

from compare_quality_cv_v2 import validate_sources, MAIN, AUGMENTED, SEEDS
from data import frozen_fold, sha
from model import MaskedQualityGRU, Scaling
from endpoint_contract_v1 import read_endpoint
from train_trajectory_trial_v1 import store
from train_trajectory_cv_v1 import row_metrics, validate_oof

RESULTS_ROOT = Path(os.environ.get('HAIRPIN_RESULTS_ROOT', 'results/phase_b_spi')).expanduser().resolve()
CV = RESULTS_ROOT/'trajectory_development_cv_v1'
THRESHOLDS = [.5, .6, .7, .8, .9, .95, .98]


def selection_score(endpoint, target_time):
    # A missing prediction is a missing event, not a synthetic valid endpoint.
    found=endpoint['endpoint_found']
    error=torch.where(found,(endpoint['t_eos_s']-target_time).abs(),torch.full_like(target_time,20.448))
    return float(error.mean())


def evaluate_readout(prediction, data, threshold, fold, seed):
    out=read_endpoint(prediction['trajectory'],eos_threshold=threshold)
    rows=[]
    for i,rid in enumerate(data['run_ids']):
        found=bool(out['endpoint_found'][i]); target=float((data['length'][i]-1)*.032)
        rows.append(dict(run_id=rid,fold=fold,seed=seed,eos_threshold=threshold,
            endpoint_found=found,target_eos_grid_s=target,
            predicted_eos_s=float(out['t_eos_s'][i]) if found else None,
            eos_time_error_s=float(out['t_eos_s'][i])-target if found else None,
            target_success=bool(data['real'][i,0,1]),predicted_success=bool(out['success_decision'][i]),
            predicted_success_probability=float(out['success_probability'][i]) if found else None,
            predicted_abs_error_mm=float(prediction['error_mm'][i]),target_abs_error_mm=float(data['error_mm'][i]),
            slip_observation_confirmed=bool(data['slip_observed'][i]),
            v_nominal_mm_s=float(data['inputs'][i,0])*1000,force_N=float(data['inputs'][i,1])))
    return rows


def run(output):
    if output.exists(): raise FileExistsError('Preserve prior calibration studies')
    torch.set_num_threads(1)
    main,augmented,sources=validate_sources()
    cv_protocol=json.loads((CV/'protocol_frozen_before_training.json').read_text())
    if sources!=cv_protocol['source_hashes']: raise ValueError('CV data mismatch')
    frozen=json.loads((CV/'fold_summaries.json').read_text())
    output.mkdir(parents=True)
    store(output/'protocol_before_evaluation.json',dict(schema='train_fold_eos_threshold_v1',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        development_adaptation='After inspecting early-EOS bias in cv_v1; no new model fit',
        candidates=THRESHOLDS,selection='Minimum training-fold mean absolute grid-EOS error; missing endpoint costs 20.448 s; first/smallest threshold wins exact ties',
        selection_data='Augmented training fold only; no heldout labels used in per-fit threshold choice',
        heldout='Same MAIN heldout runs as cv_v1; assess endpoint coverage, time bias/MAE and Success',
        caution='Resubstitution calibration within training fold; heldout assessment is Development-only; scores not probability-calibrated',
        thresholds_are_not_detector_changes=True,success_threshold=.5,
        source_hashes=sources,cv_protocol_sha256=sha(CV/'protocol_frozen_before_training.json'),
        code_sha256=sha(Path(__file__)),independent_test_access=False,robot_motion=False))
    chosen_rows=[];baseline_rows=[];choices=[]
    for fold in range(1,6):
        train,_=frozen_fold(augmented,AUGMENTED/'grouped_cv_membership_v1.json',fold)
        _,held=frozen_fold(main,MAIN/'grouped_cv_membership_v1.json',fold)
        for seed in SEEDS:
            checkpoint=CV/f'fold{fold}_seed{seed}'/'TRAJECTORY_CV_NOT_FINAL.pt'
            reference=next(r for r in frozen if r['fold']==fold and r['seed']==seed)
            if sha(checkpoint)!=reference['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            model=MaskedQualityGRU(Scaling.fit(train),**saved['model_config'])
            model.load_state_dict(saved['state_dict']);model.eval()
            with torch.no_grad():
                train_pred=model(train['inputs'],train['point_start'],train['sim'])
                held_pred=model(held['inputs'],held['point_start'],held['sim'])
            target=(train['length']-1)*.032
            scores=[]
            for threshold in THRESHOLDS:
                endpoint=read_endpoint(train_pred['trajectory'],eos_threshold=threshold)
                scores.append(dict(threshold=threshold,training_selection_score_s=selection_score(endpoint,target),
                    training_missing=int((~endpoint['endpoint_found']).sum())))
            selected=min(scores,key=lambda s:s['training_selection_score_s'])['threshold']
            choices.append(dict(fold=fold,seed=seed,selected_threshold=selected,training_scores=scores))
            chosen_rows.extend(evaluate_readout(held_pred,held,selected,fold,seed))
            baseline_rows.extend(evaluate_readout(held_pred,held,.5,fold,seed))
            print(f'ENDPOINT fold={fold} seed={seed}: threshold={selected}',flush=True)
    for rows in (chosen_rows,baseline_rows):validate_oof(rows,main['run_ids'],SEEDS)
    report={}
    for label,rows in [('fixed_0_5',baseline_rows),('train_fold_selected',chosen_rows)]:
        per_seed=[dict(seed=seed,**row_metrics([r for r in rows if r['seed']==seed])) for seed in SEEDS]
        report[label]=dict(per_seed=per_seed,mean_seed_metrics={key:float(np.mean([m[key] for m in per_seed]))
            for key in ('success_accuracy','balanced_success_accuracy','eos_MAE_to_grid_s','eos_bias_to_grid_s')},
            total_missing_prediction_rows=sum(not r['endpoint_found'] for r in rows))
    report.update(unique_evaluation_runs=90,prediction_rows_per_variant=270,
        independent_test_access=False,model_weights_changed=False,detector_changed=False,
        final_readout_frozen=False,choices=choices)
    store(output/'selected_thresholds.json',choices)
    store(output/'out_of_fold_selected.json',chosen_rows)
    store(output/'out_of_fold_baseline.json',baseline_rows)
    store(output/'calibration_summary.json',report)
    print(json.dumps({k:report[k] for k in ('fixed_0_5','train_fold_selected')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    run(parser.parse_args().output)

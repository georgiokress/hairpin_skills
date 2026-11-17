"""Fixed-budget, five-fold / three-seed Development trajectory evaluation.

Expands the inspected single-fold trial without changing architecture, loss,
thresholds or epoch budget. Does not select/freeze a final model or access test.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import torch

from compare_quality_cv_v2 import validate_sources, MAIN, AUGMENTED, SEEDS
from data import frozen_fold, select, sha
from model import MaskedQualityGRU, Scaling
from endpoint_contract_v1 import endpoint_training_loss
from train_trajectory_trial_v1 import evaluate, trim_batch, store

RESULTS_ROOT = Path(os.environ.get('HAIRPIN_RESULTS_ROOT', 'results/phase_b_spi')).expanduser().resolve()
QUALITY_ROOT = RESULTS_ROOT/'pilot_benefit_quality_cv_v2'
TRIAL_ROOT = RESULTS_ROOT/'trajectory_development_trial_v1'
EPOCHS = 50


def quality_path(fold, seed):
    return QUALITY_ROOT/f'fold{fold}_main_plus_pilot_seed{seed}_QUALITY_ONLY.pt'


def initialize(train, held, fold, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    saved = torch.load(quality_path(fold,seed), map_location='cpu', weights_only=True)
    if saved['train_run_ids'] != train['run_ids'] or saved['heldout_run_ids'] != held['run_ids']:
        raise ValueError('Frozen quality split mismatch')
    model=MaskedQualityGRU(Scaling.fit(train), hidden=32, layers=2, dropout=.1)
    if not all(torch.equal(value, saved['scaling_state'][key]) for key,value in model.scaling.state_dict().items()):
        raise ValueError('Quality/trajectory normalization mismatch')
    model.quality.load_state_dict(saved['quality_state'])
    for parameter in model.quality.parameters():
        parameter.requires_grad_(False)
    return model, saved


def fit_one(output, fold, seed):
    started=time.monotonic()
    torch.set_num_threads(1)
    main, augmented, sources=validate_sources()
    protocol_path=output/'protocol_frozen_before_training.json'
    protocol=json.loads(protocol_path.read_text())
    if sources != protocol['source_hashes']:
        raise ValueError('Sources changed after preflight')
    if sha(quality_path(fold,seed)) != protocol['quality_checkpoint_hashes'][quality_path(fold,seed).name]:
        raise ValueError('Quality checkpoint changed after preflight')
    for name, expected in protocol['code_hashes'].items():
        if sha(Path(__file__).parent/name) != expected:
            raise ValueError('Code changed after protocol freeze: '+name)
    train,_=frozen_fold(augmented,AUGMENTED/'grouped_cv_membership_v1.json',fold)
    _,held=frozen_fold(main,MAIN/'grouped_cv_membership_v1.json',fold)
    model,quality=initialize(train,held,fold,seed)
    directory=output/f'fold{fold}_seed{seed}'
    directory.mkdir()
    store(directory/'membership.json', dict(train_run_ids=train['run_ids'],heldout_main_run_ids=held['run_ids']))
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-5)
    history=[]
    for epoch in range(1,EPOCHS+1):
        model.train(); order=torch.randperm(len(train['run_ids'])).tolist(); weighted=0.
        for offset in range(0,len(order),16):
            batch=trim_batch(select(train,order[offset:offset+16]))
            optimizer.zero_grad()
            prediction=model(batch['inputs'],batch['point_start'],batch['sim'])
            loss,_=endpoint_training_loss(model,prediction,batch)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite loss/gradient')
            optimizer.step();weighted+=float(loss.detach())*len(batch['run_ids'])
        history.append(dict(epoch=epoch,mean_training_loss=weighted/len(train['run_ids']),elapsed_s=time.monotonic()-started))
        if epoch in (1,25,50):
            print(f'CV fold={fold} seed={seed} epoch={epoch}/50 loss={history[-1]["mean_training_loss"]:.5f}',flush=True)
    if not all(torch.equal(v,quality['quality_state'][k]) for k,v in model.quality.state_dict().items()):
        raise RuntimeError('Frozen quality branch changed')
    training,_=evaluate(model,train)
    validation,rows=evaluate(model,held)
    for i,row in enumerate(rows):
        row.update(fold=fold,seed=seed,v_nominal_mm_s=float(held['inputs'][i,0])*1000,force_N=float(held['inputs'][i,1]))
    checkpoint=directory/'TRAJECTORY_CV_NOT_FINAL.pt'
    torch.save(dict(schema='spi_trajectory_cv_checkpoint_v1',final_model=False,state_dict=model.state_dict(),
        model_config=dict(hidden=32,layers=2,dropout=.1),fold=fold,seed=seed,
        protocol_sha256=sha(protocol_path)), checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    clone=MaskedQualityGRU(Scaling.fit(train),**saved['model_config'])
    clone.load_state_dict(saved['state_dict']);clone.eval();model.eval()
    with torch.no_grad():
        before=model(held['inputs'],held['point_start'],held['sim'])
        after=clone(held['inputs'],held['point_start'],held['sim'])
    if not all(torch.equal(before[k],after[k]) for k in before):
        raise RuntimeError('Checkpoint reload mismatch')
    reproduced=None
    if fold==1 and seed==20260918:
        prior=torch.load(TRIAL_ROOT/'TRAJECTORY_TRIAL_NOT_FINAL.pt',map_location='cpu',weights_only=True)
        reproduced=all(torch.equal(v,prior['state_dict'][k]) for k,v in model.state_dict().items())
        if not reproduced:
            raise RuntimeError('Unchanged fixed trial did not reproduce exactly')
    store(directory/'training_history.json',history)
    store(directory/'heldout_predictions.json',rows)
    report=dict(fold=fold,seed=seed,train=training,development_main_heldout=validation,
        elapsed_s=time.monotonic()-started,checkpoint_sha256=sha(checkpoint),
        checkpoint_reload_exact=True,quality_head_unchanged=True,initial_trial_reproduced_exact=reproduced)
    store(directory/'summary.json',report)
    print(f'CV COMPLETED fold={fold} seed={seed}: accuracy={validation["success_accuracy"]:.3f}, EOS missing={validation["endpoint_missing"]}',flush=True)
    return report, rows


def row_metrics(rows):
    if not rows:
        raise ValueError('No observations')
    y=np.array([r['target_success'] for r in rows],dtype=bool)
    p=np.array([r['predicted_success'] for r in rows],dtype=bool)
    found=np.array([r['endpoint_found'] for r in rows],dtype=bool)
    time_error=np.array([r['eos_time_error_s'] for r in rows if r['endpoint_found']],dtype=float)
    positives=int(y.sum());negatives=len(y)-positives
    tp=int((p&y).sum());tn=int((~p&~y).sum())
    return dict(n=len(rows),endpoint_found=int(found.sum()),endpoint_missing=int((~found).sum()),
        success_correct=int((p==y).sum()),success_accuracy=float((p==y).mean()),
        true_positive=tp,false_negative=positives-tp,true_negative=tn,false_positive=negatives-tn,
        positive_labels=positives,negative_labels=negatives,
        balanced_success_accuracy=.5*(tp/positives+tn/negatives) if positives and negatives else None,
        eos_MAE_to_grid_s=float(np.abs(time_error).mean()) if len(time_error) else None,
        eos_bias_to_grid_s=float(time_error.mean()) if len(time_error) else None,
        eos_max_abs_error_to_grid_s=float(np.abs(time_error).max()) if len(time_error) else None,
        eos_time_error_conditional_on_found=True,
        conditional_success_brier=float(np.mean([(r['predicted_success_probability']-r['target_success'])**2 for r in rows if r['endpoint_found']])) if found.any() else None,
        native_error_head_MAE_mm=float(np.mean([abs(r['predicted_abs_error_mm']-r['target_abs_error_mm']) for r in rows])))


def validate_oof(rows, main_ids, seeds):
    if len(rows) != len(main_ids)*len(seeds):
        raise ValueError('Wrong OOF row count')
    for seed in seeds:
        ids=[r['run_id'] for r in rows if r['seed']==seed]
        if len(ids)!=len(main_ids) or set(ids)!=set(main_ids):
            raise ValueError('Every main run must occur exactly once per seed')


def summarize(output, main, completed):
    reports=sorted([r for r,_ in completed],key=lambda r:(r['fold'],r['seed']))
    rows=sorted([row for _,rr in completed for row in rr],key=lambda r:(r['seed'],r['fold'],r['run_id']))
    validate_oof(rows,main['run_ids'],SEEDS)
    seed_metrics=[dict(seed=seed,**row_metrics([r for r in rows if r['seed']==seed])) for seed in SEEDS]
    subset_metrics={name:[dict(seed=seed,**row_metrics([r for r in rows if r['seed']==seed and r['slip_observation_confirmed']==observed])) for seed in SEEDS]
        for name,observed in [('observed_NOSLIP',True),('assumed_NOSLIP',False)]}
    combination_metrics=[]
    for v,f in sorted({(r['v_nominal_mm_s'],r['force_N']) for r in rows}):
        subset=[r for r in rows if r['v_nominal_mm_s']==v and r['force_N']==f]
        combination_metrics.append(dict(v_nominal_mm_s=v,force_N=f,unique_runs=3,
            prediction_rows_including_seeds=len(subset),descriptive_pooled_metrics=row_metrics(subset)))
    mean={key:float(np.mean([r[key] for r in seed_metrics])) for key in
        ('success_accuracy','balanced_success_accuracy','eos_MAE_to_grid_s','eos_bias_to_grid_s','native_error_head_MAE_mm')
        if all(r[key] is not None for r in seed_metrics)}
    waveform={key:float(np.sqrt(np.mean([r['development_main_heldout'][key]**2 for r in reports])))
        for key in ('position_xyz_rmse_mm','force_xyz_rmse_N','torque_xyz_rmse_Nm')}
    result=dict(schema='spi_trajectory_development_cv_v1_result',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_variant='main_plus_pilot',folds=5,seeds=SEEDS,epochs=EPOCHS,
        unique_main_evaluation_runs=90,unique_parameter_combinations=30,prediction_rows=270,
        seed_metrics=seed_metrics,mean_seed_metrics=mean,waveform_metrics_across_folds_seeds=waveform,
        provenance_subset_metrics=subset_metrics,combination_metrics=combination_metrics,
        checkpoint_reload_all_exact=all(r['checkpoint_reload_exact'] for r in reports),
        initial_trial_reproduced_exact=next(r['initial_trial_reproduced_exact'] for r in reports if r['fold']==1 and r['seed']==20260918),
        independent_test_runs_read=0,robot_motion=False,inversion_performed=False,final_model_frozen=False,
        limitations=['Development evaluation expanded after inspecting fold 1; not an untouched test',
            'Only main-plus-pilot trajectory variant at one fixed budget; no final architecture/data selection',
            '270 predictions describe 90 runs, not 270 independent physical observations',
            'Repeated runs share parameter combinations, setup and hairpin; seeds do not establish physical independence',
            'Success partly conditional on assumed NOSLIP; EOS probabilities uncalibrated',
            'Hard endpoint readout does not yet define differentiable inversion objective'])
    store(output/'out_of_fold_predictions.json',rows)
    store(output/'fold_summaries.json',reports)
    store(output/'cv_summary.json',result)
    print(json.dumps(dict(mean_seed_metrics=mean,waveform_metrics=waveform,seed_metrics=seed_metrics),indent=2),flush=True)


def run(output, workers):
    if output.exists():
        raise FileExistsError('Preserve existing study; choose a new directory')
    if not 1<=workers<=3:
        raise ValueError('Use one to three CPU workers')
    torch.set_num_threads(1)
    main,augmented,sources=validate_sources()
    memberships=[];quality_hashes={}
    for fold in range(1,6):
        train,extra_held=frozen_fold(augmented,AUGMENTED/'grouped_cv_membership_v1.json',fold)
        _,held=frozen_fold(main,MAIN/'grouped_cv_membership_v1.json',fold)
        if not set(held['run_ids']) <= set(extra_held['run_ids']):
            raise ValueError('Augmented/main heldout mismatch')
        memberships.append(dict(fold=fold,train_count=len(train['run_ids']),heldout_main_count=len(held['run_ids']),
            train_run_ids=train['run_ids'],heldout_main_run_ids=held['run_ids']))
        for seed in SEEDS:
            initialize(train,held,fold,seed)
            quality_hashes[quality_path(fold,seed).name]=sha(quality_path(fold,seed))
    output.mkdir(parents=True)
    store(output/'protocol_frozen_before_training.json',dict(schema='spi_trajectory_development_cv_v1_protocol',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        development_adaptation='Expand inspected single-fold trial to all existing grouped folds and three fixed seeds; no hyperparameter/threshold change',
        dataset='main_plus_pilot',folds=5,seeds=SEEDS,epochs=EPOCHS,batch_size=16,
        optimizer=dict(name='Adam',lr=.001,weight_decay=1e-5),architecture=dict(hidden=32,layers=2,dropout=.1),
        loss='EOS: half terminal BCE + half mean preterminal BCE; success: prefix mean BCE; trajectory: 10 * normalized per-run MSE; quality frozen',
        inference='Full 640-sample analytical prior; no target mask or length used to predict endpoint',
        hard_readout='First p_EOS >=0.5; mean p_success through predicted EOS >=0.5; missing endpoint explicit',
        checkpoint_selection='Final fixed epoch only; no early stopping or heldout checkpoint selection',
        primary_metrics=['Endpoint coverage','EOS time MAE/bias conditional on endpoint','Balanced Success accuracy'],
        secondary_metrics=['Per-run weighted waveform RMSE','Unchanged quality-head MAE','Provenance subsets','Per-combination diagnostics'],
        calibration='Scores not independently probability-calibrated',
        not_final_model_selection=True,independent_test_access=False,robot_motion=False,inversion=False,
        memberships=memberships,source_hashes=sources,quality_checkpoint_hashes=quality_hashes,
        code_hashes={name:sha(Path(__file__).parent/name) for name in
            ('data.py','model.py','endpoint_contract_v1.py','train_trajectory_trial_v1.py','compare_quality_cv_v2.py',Path(__file__).name)}))
    context=multiprocessing.get_context('spawn')
    completed=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        jobs=[pool.submit(fit_one,output,fold,seed) for fold in range(1,6) for seed in SEEDS]
        for job in as_completed(jobs):
            completed.append(job.result())
            print(f'Completed {len(completed)}/15 fits',flush=True)
    summarize(output,main,completed)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();run(args.output,args.workers)

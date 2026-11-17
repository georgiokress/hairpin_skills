"""Fixed single-fold Development trial of complete trajectory prediction.

Not final model selection, not refit, not independent model-test evaluation.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from compare_quality_cv_v2 import validate_sources, MAIN, AUGMENTED
from data import frozen_fold, select, sha
from model import MaskedQualityGRU, Scaling
from endpoint_contract_v1 import endpoint_training_loss, read_endpoint

RESULTS_ROOT = Path(os.environ.get('HAIRPIN_RESULTS_ROOT', 'results/phase_b_spi')).expanduser().resolve()
QUALITY = RESULTS_ROOT/'pilot_benefit_quality_cv_v2/fold1_main_plus_pilot_seed20260918_QUALITY_ONLY.pt'
EPOCHS = 50
SEED = 20260918


def store(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def trim_batch(batch):
    n=int(batch['length'].max())
    return {k:(v[:,:n] if k in ('real','sim','mask') else v) for k,v in batch.items()}


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out=model(data['inputs'],data['point_start'],data['sim'])
        total,parts=endpoint_training_loss(model,out,data)
        endpoint=read_endpoint(out['trajectory'])
    found=endpoint['endpoint_found']
    target_success=data['real'][:,0,1].bool()
    target_time=(data['length']-1)*.032
    # Diagnostic waveform errors only use target physical mask; the model
    # readout ABOVE never uses target mask/length to predict its endpoint.
    mask=data['mask']
    waveform={}
    for name,slc,unit_scale in [('position_xyz_rmse_mm',slice(2,5),1000.),
                                ('force_xyz_rmse_N',slice(9,12),1.),
                                ('torque_xyz_rmse_Nm',slice(12,15),1.)]:
        square=((out['trajectory'][...,slc]-data['real'][...,slc])*unit_scale).square().mean(-1)
        per_run=torch.where(mask,square,torch.zeros_like(square)).sum(1)/mask.sum(1)
        waveform[name]=float(per_run.mean().sqrt())
    rows=[]
    for i,rid in enumerate(data['run_ids']):
        valid=bool(found[i])
        rows.append(dict(run_id=rid,endpoint_found=valid,
            target_eos_grid_s=float(target_time[i]),predicted_eos_s=float(endpoint['t_eos_s'][i]) if valid else None,
            eos_time_error_s=float(endpoint['t_eos_s'][i]-target_time[i]) if valid else None,
            target_success=bool(target_success[i]),predicted_success=bool(endpoint['success_decision'][i]),
            predicted_success_probability=float(endpoint['success_probability'][i]) if valid else None,
            predicted_abs_error_mm=float(out['error_mm'][i]),target_abs_error_mm=float(data['error_mm'][i]),
            slip_observation_confirmed=bool(data['slip_observed'][i])))
    true_positive=int((endpoint['success_decision'] & target_success).sum())
    true_negative=int((~endpoint['success_decision'] & ~target_success).sum())
    positives=int(target_success.sum());negatives=len(target_success)-positives
    result=dict(n=len(rows),endpoint_found=int(found.sum()),endpoint_missing=int((~found).sum()),
        eos_time_MAE_to_grid_s=float((endpoint['t_eos_s'][found]-target_time[found]).abs().mean()) if found.any() else None,
        endpoint_time_error_conditional_on_found=True,
        success_accuracy=float((endpoint['success_decision']==target_success).float().mean()),
        true_positive=true_positive,true_negative=true_negative,positive_labels=positives,negative_labels=negatives,
        balanced_success_accuracy=.5*(true_positive/positives+true_negative/negatives) if positives and negatives else None,
        conditional_success_brier=float((endpoint['success_probability'][found]-target_success[found].float()).square().mean()) if found.any() else None,
        native_error_head_MAE_mm=float((out['error_mm']-data['error_mm']).abs().mean()),
        loss=float(total),loss_terms={k:float(v) for k,v in parts.items()},**waveform)
    return result,rows


def run(output):
    if output.exists(): raise FileExistsError('Preserve existing trial')
    main,augmented,sources=validate_sources()
    _,held=frozen_fold(main,MAIN/'grouped_cv_membership_v1.json',1)
    train,_=frozen_fold(augmented,AUGMENTED/'grouped_cv_membership_v1.json',1)
    saved=torch.load(QUALITY,map_location='cpu',weights_only=True)
    if saved['train_run_ids']!=train['run_ids'] or saved['heldout_run_ids']!=held['run_ids']:
        raise ValueError('Quality checkpoint split mismatch')
    torch.set_num_threads(1); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
    model=MaskedQualityGRU(Scaling.fit(train),hidden=32,layers=2,dropout=.1)
    if not all(torch.equal(v,saved['scaling_state'][k]) for k,v in model.scaling.state_dict().items()):
        raise ValueError('Quality/trajectory normalization mismatch')
    model.quality.load_state_dict(saved['quality_state'])
    for parameter in model.quality.parameters(): parameter.requires_grad_(False)
    output.mkdir(parents=True)
    protocol=dict(schema='spi_trajectory_development_trial_v1',recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Fixed-budget single-fold technical/scientific Development trial, NOT final CV/refit',
        fold=1,seed=SEED,train_run_ids=train['run_ids'],heldout_main_run_ids=held['run_ids'],
        train_count=len(train['run_ids']),heldout_count=len(held['run_ids']),epochs=EPOCHS,batch_size=16,
        optimizer=dict(name='Adam',lr=.001,weight_decay=1e-5),architecture=dict(hidden=32,layers=2,dropout=.1),
        loss=dict(eos='0.5 terminal BCE + 0.5 mean preterminal BCE per run',success='physical-prefix per-run mean BCE',
                  trajectory='10 * per-run normalized trajectory MSE',quality='frozen existing head, weight 0'),
        quality_checkpoint=str(QUALITY),quality_checkpoint_sha256=sha(QUALITY),
        training_padding='Mask enforced; trim minibatch to longest physical prefix, exploiting tested causality',
        inference='Full 640 prior samples; no measured mask/length supplied to model or endpoint readout',
        hard_readout='First p_EOS >= 0.5; mean p_success on predicted prefix, threshold 0.5; missing EOS explicit',
        thresholds_status='Predeclared initial Development settings, not calibrated probabilities',
        checkpoint_selection='Final epoch only; no heldout early stopping or best-epoch selection',
        model_test_access=False,inversion=False,robot_motion=False,source_hashes=sources,
        code_sha256={name:sha(Path(__file__).parent/name) for name in ('model.py','data.py','endpoint_contract_v1.py',Path(__file__).name)})
    store(output/'protocol_frozen_before_training.json',protocol)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-5)
    histories=[];start=time.monotonic()
    for epoch in range(1,EPOCHS+1):
        model.train();order=torch.randperm(len(train['run_ids'])).tolist();weighted=0.
        for offset in range(0,len(order),16):
            batch=trim_batch(select(train,order[offset:offset+16]))
            optimizer.zero_grad()
            pred=model(batch['inputs'],batch['point_start'],batch['sim'])
            loss,_=endpoint_training_loss(model,pred,batch)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite trajectory training')
            optimizer.step();weighted+=float(loss.detach())*len(batch['run_ids'])
        histories.append(dict(epoch=epoch,mean_training_loss=weighted/len(train['run_ids']),elapsed_s=time.monotonic()-start))
        if epoch==1 or epoch%5==0:
            print(f'TRAJECTORY_TRIAL epoch {epoch}/{EPOCHS}, train_loss={histories[-1]["mean_training_loss"]:.5f}, elapsed={time.monotonic()-start:.1f}s',flush=True)
    if not all(torch.equal(value,saved['quality_state'][key]) for key,value in model.quality.state_dict().items()):
        raise RuntimeError('Frozen quality head changed')
    training,_=evaluate(model,train)
    validation,rows=evaluate(model,held)
    checkpoint=output/'TRAJECTORY_TRIAL_NOT_FINAL.pt'
    torch.save(dict(schema='spi_trajectory_trial_checkpoint_v1',final_model=False,state_dict=model.state_dict(),
        model_config=dict(hidden=32,layers=2,dropout=.1),protocol_sha256=sha(output/'protocol_frozen_before_training.json')),
        checkpoint)
    restored=torch.load(checkpoint,map_location='cpu',weights_only=True)
    clone=MaskedQualityGRU(Scaling.fit(train),**restored['model_config'])
    clone.load_state_dict(restored['state_dict']);clone.eval();model.eval()
    with torch.no_grad():
        a=model(held['inputs'],held['point_start'],held['sim'])
        b=clone(held['inputs'],held['point_start'],held['sim'])
    if not all(torch.equal(a[k],b[k]) for k in a): raise RuntimeError('Checkpoint reload mismatch')
    store(output/'training_history.json',histories)
    store(output/'main_heldout_predictions.json',rows)
    report=dict(schema='spi_trajectory_trial_result_v1',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        train=training,development_main_heldout=validation,epochs=EPOCHS,fold=1,seed=SEED,
        elapsed_s=time.monotonic()-start,checkpoint_sha256=sha(checkpoint),checkpoint_reload_exact=True,
        quality_head_unchanged=True,independent_model_test_runs_read=0,final_model_frozen=False,
        inversion_performed=False,robot_motion=False,
        limitations=['Only one fold and seed, preliminary training budget',
            'EOS is virtual endpoint; no proof of actual physical robot stopping performance',
            'Success labels partly conditional on assumed NOSLIP',
            'Hard endpoint readout is not differentiable: inversion objective still unresolved',
            'EOS/Success outputs not independently probability-calibrated'])
    store(output/'trial_summary.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    run(parser.parse_args().output)

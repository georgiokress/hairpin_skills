"""Fixed-budget Development refit candidate; NOT an independent test release."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import torch

from compare_quality_cv_v2 import validate_sources,MAIN,quality_features
from data import select,sha
from model import MaskedQualityGRU,Scaling
from endpoint_contract_v1 import endpoint_training_loss,read_endpoint
from train_trajectory_trial_v1 import store,trim_batch
from calibrate_endpoint_v1 import THRESHOLDS,selection_score

SEED=20260918


def run(output,variant):
    if output.exists():raise FileExistsError('Never replace a previous refit')
    torch.set_num_threads(1);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    main,augmented,sources=validate_sources()
    data=augmented if variant=='main_plus_pilot' else main
    # Reference context chosen by parameter and chronological order, not quality.
    index=next(i for i,x in enumerate(main['inputs']) if abs(float(x[0])-.004)<1e-8 and float(x[1])==2.)
    reference_id=main['run_ids'][index]
    meta_path=MAIN/'native_runs'/reference_id/'meta.json';meta=json.loads(meta_path.read_text())
    context=dict(run_id=reference_id,point_start=main['point_start'][index].tolist(),
        reference_distance_m=meta['r_gt_run_mm']*.001,reference_start_x_m=meta['x_r_start_m'],
        meta_sha256=sha(meta_path),selection='First MAIN run at historical 2 N / 4 mm/s reference; no quality-based pose selection',
        limitation='Hypothetical optimization context, not a statement about the current live robot pose')
    output.mkdir(parents=True)
    protocol=dict(schema='spi_refit_candidate_v1',recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_variant=variant,run_ids=data['run_ids'],n=len(data['run_ids']),source_hashes=sources,
        seed=SEED,seed_choice='Previously fixed first seed, not a best-seed selection',
        architecture=dict(hidden=32,layers=2,dropout=.1),quality_full_batch_steps=4000,
        trajectory_updates=300,batch_size=16,optimizer=dict(name='Adam',lr=.001,weight_decay=1e-5),
        budget='300 trajectory updates matched to Development comparison; final partial epoch allowed; no training-metric checkpoint selection',
        eos_thresholds=THRESHOLDS,threshold_selection='Entire Development training set only, same minimum EOS-grid MAE / missing penalty algorithm',
        search_bounds=dict(v_mm_s=[1,10],force_N=[1,5]),inversion_reference_context=context,
        independent_test_access=False,final_test_release=False,robot_motion=False,
        code_hashes={n:sha(Path(__file__).parent/n) for n in ('model.py','data.py','endpoint_contract_v1.py','analytical_prior_v1.py',Path(__file__).name)})
    store(output/'protocol_before_refit.json',protocol)
    model=MaskedQualityGRU(Scaling.fit(data),hidden=32,layers=2,dropout=.1)
    features=quality_features(model,data)
    optimizer=torch.optim.Adam(model.quality.parameters(),lr=.001,weight_decay=1e-5)
    quality_history=[];started=time.monotonic()
    for step in range(1,4001):
        optimizer.zero_grad();estimate=model.quality(features).squeeze(-1)
        loss=(estimate-data['error_mm']).square().mean();loss.backward();optimizer.step()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite quality loss')
        if step in (1,100,1000,2000,4000):quality_history.append(dict(step=step,mse_mm2=float(loss.detach())))
    for p in model.quality.parameters():p.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-5)
    history=[];step=0;epoch=0
    while step<300:
        epoch+=1;model.train();order=torch.randperm(len(data['run_ids'])).tolist()
        for offset in range(0,len(order),16):
            if step>=300:break
            batch=trim_batch(select(data,order[offset:offset+16]));optimizer.zero_grad()
            pred=model(batch['inputs'],batch['point_start'],batch['sim'])
            loss,_=endpoint_training_loss(model,pred,batch);loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite trajectory fit')
            optimizer.step();step+=1
            if step==1 or step%25==0:
                history.append(dict(step=step,epoch=epoch,batch_loss=float(loss.detach()),elapsed_s=time.monotonic()-started))
                print(f'REFIT {variant} {step}/300',flush=True)
    model.eval()
    with torch.no_grad():pred=model(data['inputs'],data['point_start'],data['sim'])
    scores=[dict(threshold=t,score_s=selection_score(read_endpoint(pred['trajectory'],eos_threshold=t),(data['length']-1)*.032)) for t in THRESHOLDS]
    threshold=min(scores,key=lambda r:r['score_s'])['threshold']
    checkpoint=output/'REFIT_CANDIDATE_NOT_TEST_RELEASED.pt'
    torch.save(dict(schema='spi_refit_candidate_checkpoint_v1',state_dict=model.state_dict(),
        model_config=dict(hidden=32,layers=2,dropout=.1),eos_threshold=threshold,seed=SEED,
        variant=variant,protocol_sha256=sha(output/'protocol_before_refit.json'),
        inversion_reference_context=context,independent_test_released=False),checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    clone=MaskedQualityGRU(Scaling.fit(data),**saved['model_config']);clone.load_state_dict(saved['state_dict']);clone.eval()
    with torch.no_grad():reloaded=clone(data['inputs'],data['point_start'],data['sim'])
    if not all(torch.equal(pred[k],reloaded[k]) for k in pred):raise RuntimeError('Refit reload mismatch')
    report=dict(schema='spi_refit_candidate_report_v1',n=len(data['run_ids']),variant=variant,seed=SEED,
        trajectory_updates=step,quality_steps=4000,selected_threshold=threshold,training_threshold_scores=scores,
        quality_training_mae_mm=float((pred['error_mm']-data['error_mm']).abs().mean()),
        metrics_are_resubstitution_not_generalization=True,checkpoint_sha256=sha(checkpoint),reload_exact=True,
        elapsed_s=time.monotonic()-started,independent_test_access=False,final_test_release=False,robot_motion=False)
    store(output/'training_history.json',dict(quality=quality_history,trajectory=history))
    store(output/'refit_summary.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--variant',choices=['main_only','main_plus_pilot'],required=True)
    args=parser.parse_args();run(args.output,args.variant)

"""Bounded multi-start SPI inversion; candidate comparison only, NO robot I/O.

Uses the full differentiable prior/GRU and soft endpoint, plus optional native
absolute-error head. Hard readout rechecks every candidate after optimization.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import MaskedQualityGRU,Scaling
from analytical_prior_v1 import sliding_prior
from endpoint_contract_v1 import read_endpoint
from soft_endpoint_v1 import objective_terms
from train_trajectory_trial_v1 import store
from data import sha

TIME_WEIGHTS=[.1,.3,1.,3.,10.]
ERROR_WEIGHTS=[0.,.3,1.,3.]
STARTS=[(4.,3.),(2.,5.),(10.,1.)]


def nondominated(rows,error='error_mm',time_key='time_s'):
    return [r for r in rows if not any(
        other[error]<=r[error] and other[time_key]<=r[time_key] and
        (other[error]<r[error] or other[time_key]<r[time_key]) for other in rows)]


def run(refit,output):
    if output.exists():raise FileExistsError('Preserve previous candidate search')
    torch.set_num_threads(1)
    checkpoint=refit/'REFIT_CANDIDATE_NOT_TEST_RELEASED.pt'
    report=json.loads((refit/'refit_summary.json').read_text())
    if sha(checkpoint)!=report['checkpoint_sha256']:raise ValueError('Changed candidate checkpoint')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['protocol_sha256']!=sha(refit/'protocol_before_refit.json'):raise ValueError('Refit protocol changed')
    state=saved['state_dict']
    model=MaskedQualityGRU(Scaling(state['scaling.input_limits'],state['scaling.output_limits']),**saved['model_config'])
    model.load_state_dict(state);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    context=saved['inversion_reference_context'];threshold=saved['eos_threshold']
    configurations=[dict(time_weight=tw,error_weight=ew,start_v_mm_s=v,start_force_N=f,
        family='success_time' if ew==0 else 'success_time_error') for ew in ERROR_WEIGHTS for tw in TIME_WEIGHTS for v,f in STARTS]
    output.mkdir(parents=True)
    store(output/'protocol_before_inversion.json',dict(schema='spi_candidate_pareto_inversion_v1',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=sha(checkpoint),
        refit_protocol_sha256=saved['protocol_sha256'],eos_threshold=threshold,soft_gate_sharpness=100.,
        objective='1 - soft_prefix_success + lambda_t * soft_duration_s / 8.192 + lambda_e * predicted_native_abs_error_mm / 1.5',
        original_comparison='lambda_e=0 preserves historical Success/time preference family; endpoint handling is adapted, not bit-identical original GCycle',
        references='8.192 s historical fixed time scale; 1.5 mm unchanged Success localization tolerance',
        sweep_is_not_final_user_preference=True,time_weights=TIME_WEIGHTS,error_weights=ERROR_WEIGHTS,
        starts=STARTS,optimizer=dict(name='Adam on normalized bounded inputs',lr=.03,iterations=120),
        bounds=dict(nominal_v_mm_s=[1,10],force_magnitude_N=[1,5]),
        reference_context=context,hard_recheck='Predicted EOS exists, prefix Success >=0.5, predicted absolute error <=1.5 mm',
        feasibility_is_model_based_not_physical_qualification=True,
        independent_test_access=False,robot_motion=False,final_candidate_selected=False,
        code_hashes={n:sha(Path(__file__).parent/n) for n in ('analytical_prior_v1.py','soft_endpoint_v1.py','endpoint_contract_v1.py',Path(__file__).name)}))
    initial=torch.tensor([[r['start_v_mm_s'],r['start_force_N']] for r in configurations])
    lo=torch.tensor([1.,1.]);hi=torch.tensor([10.,5.])
    parameters=(2*(initial-lo)/(hi-lo)-1).requires_grad_(True)
    start_pose=torch.tensor(context['point_start'])[None].expand(len(parameters),-1)
    tw=torch.tensor([r['time_weight'] for r in configurations]);ew=torch.tensor([r['error_weight'] for r in configurations])
    optimizer=torch.optim.Adam([parameters],lr=.03)
    best_value=torch.full((len(parameters),),float('inf'));best_parameters=parameters.detach().clone()
    history=[];started=time.monotonic()

    def predict(normalized):
        engineering=lo+(normalized+1)*(hi-lo)/2
        inputs=torch.stack((engineering[:,0]*.001,engineering[:,1]),dim=1)
        prior=sliding_prior(inputs,start_pose,context['reference_distance_m'])
        return model(inputs,start_pose,prior)

    for iteration in range(121):
        optimizer.zero_grad();pred=predict(parameters)
        terms=objective_terms(pred,threshold=threshold)
        values=terms['failure']+tw*terms['duration_normalized']+ew*terms['absolute_error_normalized']
        if not torch.isfinite(values).all():raise RuntimeError('Nonfinite inversion objective')
        improved=values.detach()<best_value
        best_value=torch.where(improved,values.detach(),best_value)
        best_parameters=torch.where(improved[:,None],parameters.detach(),best_parameters)
        if iteration%20==0:
            history.append(dict(iteration=iteration,mean_best_objective=float(best_value.mean()),elapsed_s=time.monotonic()-started))
            print(f'INVERSION {iteration}/120, mean best={float(best_value.mean()):.4f}',flush=True)
        if iteration==120:break
        values.sum().backward()
        if parameters.grad is None or not torch.isfinite(parameters.grad).all():raise RuntimeError('Invalid inversion parameter gradient')
        optimizer.step()
        with torch.no_grad():parameters.clamp_(-1,1)
    with torch.no_grad():
        pred=predict(best_parameters);hard=read_endpoint(pred['trajectory'],eos_threshold=threshold)
        soft=objective_terms(pred,threshold=threshold);engineering=lo+(best_parameters+1)*(hi-lo)/2
    rows=[]
    for i,configuration in enumerate(configurations):
        found=bool(hard['endpoint_found'][i]);error=float(pred['error_mm'][i])
        success=bool(hard['success_decision'][i]);duration=float(hard['t_eos_s'][i]) if found else None
        rows.append(dict(candidate_id=i+1,**configuration,v_mm_s=float(engineering[i,0]),force_N=float(engineering[i,1]),
            time_s=duration,error_mm=error,endpoint_found=found,predicted_success=success,
            success_score=float(hard['success_probability'][i]) if found else None,
            soft_time_s=float(soft['duration_s'][i]),soft_success_score=float(soft['success_score'][i]),
            soft_missing_mass=float(soft['missing_endpoint_mass'][i]),best_objective=float(best_value[i]),
            trajectory_endpoint_absolute_error_mm=abs(float(hard['position_at_eos_m'][i,0])-.00925)*1000 if found else None,
            model_feasible=found and success and error<=1.5))
    feasible=[r for r in rows if r['model_feasible']]
    front=sorted(nondominated(feasible),key=lambda r:r['time_s'])
    # Exact duplicate outputs from different sweep settings do not add evidence.
    distinct=[];seen=set()
    for row in front:
        key=(round(row['v_mm_s'],5),round(row['force_N'],5))
        if key not in seen:distinct.append(row);seen.add(key)
    dataset_root=Path(os.environ.get('HAIRPIN_DATASET_ROOT', 'datasets/phase_b_spi')).expanduser().resolve()
    labels=json.loads((dataset_root/'main_doe_v1_spi_v1/evaluation/labels.json').read_text())
    groups={}
    for row in labels:groups.setdefault((row['F_N'],row['v_nominal_mm_s']),[]).append(row)
    empirical=[]
    for (f,v),rr in groups.items():
        empirical.append(dict(force_N=f,v_mm_s=v,n=len(rr),all_success=all(r['success'] for r in rr),
            success_fraction=float(np.mean([r['success'] for r in rr])),
            error_mm=float(np.mean([r['e_abs_mm'] for r in rr])),time_s=float(np.mean([r['t_det_native_s'] for r in rr]))))
    measured_front=nondominated([r for r in empirical if r['all_success']])
    store(output/'candidates.json',rows);store(output/'predicted_pareto.json',distinct)
    store(output/'measured_development_combinations.json',empirical)
    store(output/'measured_development_pareto.json',measured_front)
    store(output/'history.json',history)
    summary=dict(schema='spi_candidate_pareto_result_v1',searches=len(rows),iterations=120,
        model_feasible_searches=len(feasible),distinct_predicted_pareto_candidates=len(distinct),
        elapsed_s=time.monotonic()-started,checkpoint_sha256=sha(checkpoint),
        nominal_bounds_respected=all(1-1e-6<=r['v_mm_s']<=10+1e-6 and 1-1e-6<=r['force_N']<=5+1e-6 for r in rows),
        physical_candidates_executed=0,independent_test_access=False,final_candidate_selected=False,
        limitations=['Finite multi-start search, no global optimality proof','Surrogate/model prediction errors and start-context sensitivity remain',
            'Measured Development means and model-predicted points are different kinds of evidence',
            'Virtual EOS duration is not the measured full robot cycle time','User chooses compromise before final comparison runs'])
    store(output/'summary.json',summary)
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    ax.scatter([r['time_s'] for r in empirical if r['all_success']],[r['error_mm'] for r in empirical if r['all_success']],
        color='#606d78',marker='x',label='Gemessene Development-Mittel (3/3 Success)')
    for family,color,label in [('success_time','#e09b32','Modellkandidaten: Success + Zeit'),
                               ('success_time_error','#3389b8','Modellkandidaten: zusätzlich Ortsfehler')]:
        rr=[r for r in feasible if r['family']==family]
        ax.scatter([r['time_s'] for r in rr],[r['error_mm'] for r in rr],s=24,alpha=.65,color=color,label=label)
    if distinct:
        ax.plot([r['time_s'] for r in distinct],[r['error_mm'] for r in distinct],color='#16794b',lw=1.5,marker='o',label='Nicht dominierte Modellkandidaten')
    reference=next(r for r in empirical if r['force_N']==2 and r['v_mm_s']==4)
    ax.scatter([reference['time_s']],[reference['error_mm']],marker='*',s=180,color='#882e72',label='Historische Referenz 2 N / 4 mm/s (gemessen)')
    ax.set_xlabel('Virtuelle Detektions-/EOS-Zeit [s]');ax.set_ylabel('Absoluter Ortsfehler [mm]')
    ax.set_title('Genauigkeit–Dauer: Kandidatenvergleich, noch keine reale Bestätigung')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(output/'pareto_genauigkeit_dauer.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(summary=summary,predicted_pareto=distinct,measured_pareto=measured_front),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('refit',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();run(args.refit,args.output)

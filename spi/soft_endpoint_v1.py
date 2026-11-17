"""Differentiable predicted-endpoint surrogate for SPI, NOT calibrated probability.

The increasing envelope of a smooth EOS gate approximates first crossing.
Unlike independent per-step hazards, a constant subthreshold score does not
accumulate into a false near-certain event. Horizon is fixed, not target length.
Weights describe an optimization surrogate; they are not statistical event
probabilities. Hard read_endpoint remains the reporting rule.
"""
import math

import torch


def soft_endpoint(trajectory, threshold=.5, sharpness=100., dt_s=.032):
    if trajectory.ndim!=3 or trajectory.shape[-1]!=15 or trajectory.shape[1]<2:
        raise ValueError('Expected B,T,15, T>=2')
    if not (0<threshold<1 and math.isfinite(sharpness) and sharpness>0 and math.isfinite(dt_s) and dt_s>0):
        raise ValueError('Invalid soft readout settings')
    scores=trajectory[...,:2]
    if not torch.isfinite(scores).all() or torch.any((scores<0)|(scores>1)):
        raise ValueError('EOS and Success scores must be finite in [0,1]')
    eos,success=scores[...,0],scores[...,1]
    scalar=eos.new_tensor
    low=torch.sigmoid(scalar(-sharpness*threshold))
    high=torch.sigmoid(scalar(sharpness*(1-threshold)))
    if float(high-low)<1e-6:raise ValueError('Degenerate soft gate')
    gate=((torch.sigmoid(sharpness*(eos-threshold))-low)/(high-low)).clamp(0,1)
    envelope=torch.cummax(gate,dim=1).values
    increments=envelope-torch.cat((torch.zeros_like(envelope[:,:1]),envelope[:,:-1]),dim=1)
    index=torch.arange(trajectory.shape[1],device=trajectory.device,dtype=trajectory.dtype)
    time=index*dt_s
    prefix_success=success.cumsum(1)/(index+1)
    missing=1-envelope[:,-1]
    # Missing mass receives the fixed horizon, not an attractive zero time.
    duration=(increments*time).sum(1)+missing*time[-1]
    success_mass=(increments*prefix_success).sum(1)
    return dict(duration_s=duration,success_score=success_mass,
        missing_endpoint_mass=missing,endpoint_mass=envelope[:,-1],weights=increments)


def objective_terms(predictions,threshold=.5,sharpness=100.,error_reference_mm=1.5,time_reference_s=8.192):
    """Return separate terms, deliberately no unapproved preference weights."""
    if error_reference_mm<=0 or time_reference_s<=0:raise ValueError('References must be positive')
    readout=soft_endpoint(predictions['trajectory'],threshold,sharpness)
    return dict(failure=1-readout['success_score'],
        duration_normalized=readout['duration_s']/time_reference_s,
        absolute_error_normalized=predictions['error_mm']/error_reference_mm,
        **readout)

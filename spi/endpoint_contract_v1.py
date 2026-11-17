"""Hard predicted-endpoint evaluation and balanced terminal-EOS training.

This is an offline model readout, not a robot stop or inversion objective.
Never pass measured valid lengths to predicted endpoint evaluation.
"""
import math

import torch
from torch.nn import functional as F

from model import training_loss


def balanced_eos_loss(probability, target, mask):
    """Each run: 1/2 terminal BCE + 1/2 mean preterminal BCE; no padding."""
    if probability.shape != target.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError('Expected equal B,T tensors and boolean mask')
    positive = mask & (target == 1)
    negative = mask & (target == 0)
    if not torch.all(positive.sum(1) == 1) or not torch.all(negative.sum(1) >= 1):
        raise ValueError('Exactly one physical terminal EOS and at least one negative required')
    if not torch.equal(positive | negative, mask):
        raise ValueError('EOS labels must be binary')
    probability = torch.where(mask, probability, torch.full_like(probability, .5))
    target = torch.where(mask, target, torch.zeros_like(target))
    losses = F.binary_cross_entropy(probability, target, reduction='none')
    pos = torch.where(positive, losses, torch.zeros_like(losses)).sum(1)
    neg = torch.where(negative, losses, torch.zeros_like(losses)).sum(1) / negative.sum(1)
    return (.5 * pos + .5 * neg).mean()


def endpoint_training_loss(model, predictions, batch, trajectory_weight=10., quality_weight=0.):
    """New versioned loss; preserve the original smoke-test loss unchanged."""
    total, terms = training_loss(model, predictions, batch, trajectory_weight, quality_weight)
    balanced = balanced_eos_loss(predictions['residual'][...,0], batch['real'][...,0], batch['mask'])
    result = total - terms['eos'] + balanced
    return result, dict(terms, eos=balanced)


def read_endpoint(trajectory, dt_s=.032, eos_threshold=.5, success_threshold=.5):
    """First predicted EOS crossing, then mean Success on predicted prefix.

    EOS means a virtual endpoint, not necessarily a valid detector trigger.
    With no crossing: index=-1, time/position/probability=NaN, decision=False.
    These NaNs represent missing endpoints; they are never 0-second successes.
    Uses no target mask, native detector output or ground-truth endpoint.
    """
    if trajectory.ndim != 3 or trajectory.shape[-1] != 15 or trajectory.shape[1] < 1:
        raise ValueError('Expected B,T,15 trajectory')
    if not math.isfinite(dt_s) or dt_s <= 0 or not 0 < eos_threshold < 1 or not 0 < success_threshold < 1:
        raise ValueError('Invalid readout settings')
    eos, success = trajectory[...,0], trajectory[...,1]
    crosses = eos >= eos_threshold
    has_endpoint = crosses.any(1)
    first = crosses.to(torch.int64).argmax(1)
    length = trajectory.shape[1]
    checked_prefix = torch.arange(length, device=trajectory.device)[None] <= torch.where(has_endpoint, first, length-1)[:,None]
    if not torch.all(torch.isfinite(eos[checked_prefix])) or not torch.all((eos[checked_prefix] >= 0) & (eos[checked_prefix] <= 1)):
        raise ValueError('Invalid EOS probabilities before predicted end')
    prefix = checked_prefix & has_endpoint[:,None]
    if not torch.all(torch.isfinite(success[prefix])) or not torch.all((success[prefix] >= 0) & (success[prefix] <= 1)):
        raise ValueError('Invalid Success probabilities before predicted end')
    summed = torch.where(prefix, success, torch.zeros_like(success)).sum(1)
    probability = summed / prefix.sum(1).clamp_min(1)
    missing = torch.full_like(probability, float('nan'))
    probability = torch.where(has_endpoint, probability, missing)
    time = torch.where(has_endpoint, first.to(trajectory.dtype)*dt_s, missing)
    indices = torch.where(has_endpoint, first, torch.full_like(first,-1))
    positions = trajectory[torch.arange(len(trajectory),device=trajectory.device),first,2:5]
    if not torch.isfinite(positions[has_endpoint]).all():
        raise ValueError('Nonfinite predicted endpoint position')
    positions = torch.where(has_endpoint[:,None],positions,torch.full_like(positions,float('nan')))
    return dict(endpoint_found=has_endpoint, eos_index=indices, t_eos_s=time,
        success_probability=probability, success_decision=has_endpoint & (probability >= success_threshold),
        position_at_eos_m=positions)

"""Tensor-batched form of the frozen Hairpin Sliding prior, in SI units.

Same equations as spi_bounds_v1/sliding_prior.py, with explicit known run
reference distance. The recorded reference start and interpolated first pose
need not be identical. No measured EOS, final trajectory or masks are used.
"""
import torch


def sliding_prior(inputs,start,reference_distance_m,horizon=640,dt=.032):
    if inputs.ndim!=2 or inputs.shape[1]!=2 or start.shape!=(len(inputs),7):
        raise ValueError('Expected B,2 inputs and B,7 start pose')
    if horizon!=640 or dt!=.032:raise ValueError('Frozen horizon/raster required')
    if not torch.isfinite(inputs).all() or not torch.isfinite(start).all():raise ValueError('Nonfinite prior inputs')
    time=torch.arange(horizon,dtype=inputs.dtype,device=inputs.device)[None]*dt
    speed=inputs[:,0:1].clamp(min=1e-6)
    reference=torch.as_tensor(reference_distance_m,dtype=inputs.dtype,device=inputs.device)
    if reference.ndim==0:reference=reference.expand(len(inputs))
    if reference.shape!=(len(inputs),) or not torch.isfinite(reference).all():
        raise ValueError('Explicit reference distance must be scalar or B-vector')
    nominal_time=reference[:,None]/speed
    if torch.any(nominal_time.detach()<0) or torch.any(nominal_time.detach()>time[0,-1]):
        raise ValueError('Nominal event outside model horizon')
    distance=(speed*time).clamp(max=.015)
    x=start[:,0:1]+distance;y=start[:,1:2].expand_as(x);z=start[:,2:3].expand_as(x)
    quat=start[:,None,3:7].expand(-1,horizon,-1)
    zeros=torch.zeros_like(x);force=inputs[:,1:2].expand_as(x)
    eos=torch.sigmoid((time-nominal_time)/(.032*2))
    eos=torch.cat((eos[:,:-1],torch.ones_like(eos[:,-1:])),dim=1)
    return torch.stack((eos,zeros,x,y,z,quat[:,:,0],quat[:,:,1],quat[:,:,2],quat[:,:,3],
        zeros,zeros,force,zeros,zeros,zeros),dim=-1)

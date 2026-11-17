"""Versioned residual-GRU adapter plus native absolute-error surrogate.

The quality branch uses only commanded parameters and known initial pose.
It never receives measured trajectories, target EOS, true lengths or masks.
"""
import torch
from torch import nn


def bounds(values):
    low, high = values.amin(0), values.amax(0)
    constant = low == high
    return torch.stack((torch.where(constant, low - 1, low),
                        torch.where(constant, high + 1, high)))


class Scaling(nn.Module):
    def __init__(self, input_limits, output_limits):
        super().__init__()
        self.register_buffer('input_limits', input_limits)
        self.register_buffer('output_limits', output_limits)

    @classmethod
    def fit(cls, train):
        # Only training runs and physical labels determine normalization.
        physical = torch.cat((train['real'][train['mask']], train['sim'][train['mask']]))
        return cls(bounds(train['inputs']), bounds(physical[:, 2:]))

    @staticmethod
    def scale(x, limits):
        return 2 * (x - limits[0]) / (limits[1] - limits[0]) - 1

    def inputs(self, x):
        return self.scale(x, self.input_limits)

    def trajectory(self, x):
        return torch.cat((x[..., :2], self.scale(x[..., 2:], self.output_limits)), -1)

    def restore(self, x):
        values = (x[..., 2:] + 1) * (self.output_limits[1] - self.output_limits[0]) / 2 + self.output_limits[0]
        return torch.cat((x[..., :2], values), -1)


class MaskedQualityGRU(nn.Module):
    def __init__(self, scaling, hidden=32, layers=2, dropout=0.1):
        super().__init__()
        self.scaling = scaling
        self.input_layer = nn.Linear(17, hidden)
        self.grus = nn.ModuleList(nn.GRU(2 * hidden, hidden, batch_first=True) for _ in range(layers))
        self.output_layer = nn.Linear(hidden, 15)
        self.activation = nn.SELU()
        self.dropout = nn.Dropout(dropout)
        # Separate run-level regression head: nonnegative native |e_r| in mm.
        self.quality = nn.Sequential(nn.Linear(9, hidden), nn.SELU(), nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, inputs, start, sim):
        normalized_sim = self.scaling.trajectory(sim)
        static = self.scaling.inputs(inputs)
        x = self.activation(self.input_layer(torch.cat((static[:, None, :].expand(-1, sim.shape[1], -1), normalized_sim), -1)))
        previous = x
        for gru in self.grus:
            # Each layer starts from its own zero state. Never seed it with a
            # previous layer's END state (which would leak future/padded inputs).
            value, _ = gru(torch.cat((previous, x), -1))
            previous, x = x, self.dropout(self.activation(value))
        raw = self.output_layer(x)
        residual = torch.cat((raw[..., :2].sigmoid(), raw[..., 2:]), -1)
        trajectory = torch.cat((residual[..., :2], residual[..., 2:] + normalized_sim[..., 2:]), -1)
        normalized_start = Scaling.scale(start, self.scaling.output_limits[:, :7])
        error_mm = self.quality(torch.cat((static, normalized_start), -1)).squeeze(-1)
        return dict(residual=residual, trajectory=self.scaling.restore(trajectory), error_mm=error_mm)


def per_run_mean(values, mask):
    # Select before reduction: even NaN/Inf padding has no loss or gradient.
    while values.ndim > 2:
        values = values.mean(-1)
    safe = torch.where(mask, values, torch.zeros_like(values))
    return (safe.sum(1) / mask.sum(1).clamp_min(1)).mean()


def training_loss(model, predictions, batch, trajectory_weight=10.0, quality_weight=1.0):
    mask = batch['mask']
    if torch.any(mask.sum(1) < 2):
        raise ValueError('At least two physical samples including EOS required')
    pred = predictions['residual']
    real = model.scaling.trajectory(batch['real'])
    sim = model.scaling.trajectory(batch['sim'])
    target = torch.cat((real[..., :2], real[..., 2:] - sim[..., 2:]), -1)
    # Sanitize padding BEFORE pointwise losses, rather than multiplying NaNs by zero.
    pred = torch.where(mask[..., None], pred, torch.full_like(pred, .5))
    target = torch.where(mask[..., None], target, torch.zeros_like(target))
    eos = nn.functional.binary_cross_entropy(pred[..., 0], target[..., 0], reduction='none')
    success = nn.functional.binary_cross_entropy(pred[..., 1], target[..., 1], reduction='none')
    trajectory = (pred[..., 2:] - target[..., 2:]).square()
    valid = batch['error_valid']
    if valid.any():
        # Physical target in mm; missing detections are excluded, never imputed as 0 mm.
        quality = nn.functional.mse_loss(predictions['error_mm'][valid], batch['error_mm'][valid])
    else:
        quality = predictions['error_mm'].sum() * 0
    parts = dict(eos=per_run_mean(eos, mask), success=per_run_mean(success, mask),
                 trajectory=per_run_mean(trajectory, mask), quality_mm2=quality)
    total = parts['eos'] + parts['success'] + trajectory_weight * parts['trajectory'] + quality_weight * quality
    return total, parts

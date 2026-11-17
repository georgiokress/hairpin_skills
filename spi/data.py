"""Strict Development-only loader; never infer membership from directory names."""
import hashlib
import json
from pathlib import Path

import numpy as np
import tables
import torch

DETECTOR_SHA = 'c729b577128b4870931e516c05e45ea84c9526e8753e6824e61886a55b3e7917'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_development(path):
    """Load the frozen main-series format, rejecting other splits before arrays."""
    with tables.open_file(path, 'r') as handle:
        if handle.root._v_attrs.dataset_split != 'development':
            raise ValueError('Training loader rejects non-Development data')
        if handle.root._v_attrs.detector_sha256 != DETECTOR_SHA:
            raise ValueError('Frozen detector mismatch')
        arrays = {k: handle.get_node('/' + k + '/0').read() for k in
                  ('inputs', 'point_start', 'sim', 'real', 'physical_sample_mask', 'valid_length')}
        ids = [v.decode() for v in handle.root.run_metadata.run_id.read()]
        error = handle.root.run_metadata.native_e_signed_mm.read()
        observed = handle.root.run_metadata.slip_observation_confirmed.read()
        historical = bool(getattr(handle.root._v_attrs, 'historical_pilot_included', False))
    n = len(ids)
    if n != len(set(ids)) or not all('_maindoev1_' in rid or (historical and '_pilotdoev0_' in rid) for rid in ids):
        raise ValueError('Duplicate or unauthorized series IDs')
    if arrays['real'].shape != (n, 640, 15) or arrays['sim'].shape != (n, 640, 15):
        raise ValueError('Expected N x 640 x 15 trajectories')
    if arrays['inputs'].shape != (n, 2) or arrays['point_start'].shape != (n, 7):
        raise ValueError('Invalid static input shapes')
    length = arrays['valid_length']
    mask = arrays['physical_sample_mask']
    if length.shape != (n,) or np.any(length < 2) or np.any(length > 640):
        raise ValueError('Invalid physical lengths')
    if mask.dtype != np.bool_ or not np.array_equal(mask, np.arange(640)[None, :] < length[:, None]):
        raise ValueError('Mask must be a contiguous physical prefix, including terminal EOS')
    if error.shape != (n,) or observed.shape != (n,) or np.isinf(error).any():
        raise ValueError('Invalid native labels/provenance')
    for key in ('inputs', 'point_start', 'sim', 'real'):
        if not np.isfinite(arrays[key]).all():
            raise ValueError('Nonfinite ' + key)
    for i, count in enumerate(length):
        real = arrays['real'][i]
        if not (np.all(real[:count-1, 0] == 0) and np.all(real[count-1:, 0] == 1)):
            raise ValueError('EOS does not match valid_length')
        if not np.all(real[count:] == real[count-1]):
            raise ValueError('Unexpected real padding')
        if real[0, 1] not in (0, 1) or not np.all(real[:, 1] == real[0, 1]):
            raise ValueError('Success must be a run-constant binary label')
    data = {k: torch.from_numpy(v).float() for k, v in arrays.items()
            if k not in ('physical_sample_mask', 'valid_length')}
    data.update(mask=torch.from_numpy(mask), length=torch.from_numpy(length).long(),
                error_mm=torch.from_numpy(np.abs(error)).float(),
                error_valid=torch.from_numpy(np.isfinite(error)),
                slip_observed=torch.from_numpy(observed).bool(), run_ids=ids)
    return data


def select(data, indices):
    return {k: [v[i] for i in indices] if k == 'run_ids' else v[indices]
            for k, v in data.items()}


def frozen_fold(data, membership_path, number):
    membership = json.loads(Path(membership_path).read_text())
    if membership['data_role'] != 'development_only':
        raise ValueError('Not a Development membership')
    fold = next(f for f in membership['folds'] if f['fold'] == number)
    lookup = {rid: i for i, rid in enumerate(data['run_ids'])}
    train_ids, held_ids = fold['train_run_ids'], fold['heldout_run_ids']
    if (len(set(train_ids)) != len(train_ids) or len(set(held_ids)) != len(held_ids)
            or set(train_ids) & set(held_ids) or set(train_ids + held_ids) != set(lookup)):
        raise ValueError('Invalid or leaking membership')
    train = select(data, [lookup[rid] for rid in train_ids])
    held = select(data, [lookup[rid] for rid in held_ids])
    if set(map(tuple, train['inputs'].tolist())) & set(map(tuple, held['inputs'].tolist())):
        raise ValueError('Parameter combination leakage')
    return train, held

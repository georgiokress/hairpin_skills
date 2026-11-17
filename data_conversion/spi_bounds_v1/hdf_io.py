"""Minimal HDF5 I/O used by the frozen SPI data conversion."""

from pathlib import Path

import tables
import torch


def save_data(path, inputs, point_start, *, sim=None, real=None):
    """Append one tensor batch using the original SPI HDF5 layout."""
    path = Path(path)
    mode = 'a' if path.exists() else 'w'
    path.parent.mkdir(parents=True, exist_ok=True)
    with tables.open_file(path, mode, title='Simulation data') as handle:
        if mode == 'w':
            for group in ('inputs', 'point_start'):
                handle.create_group('/', group)
            if sim is not None:
                handle.create_group('/', 'sim')
            if real is not None:
                handle.create_group('/', 'real')
        names = [int(node.name) for node in handle.list_nodes('/inputs')]
        batch = str(max(names) + 1 if names else 0)
        handle.create_array('/inputs', batch, inputs.detach().cpu().numpy())
        handle.create_array(
            '/point_start', batch, point_start.detach().cpu().numpy()
        )
        if sim is not None:
            handle.create_array('/sim', batch, sim.detach().cpu().numpy())
        if real is not None:
            handle.create_array('/real', batch, real.detach().cpu().numpy())


def _load_group(handle, group):
    nodes = sorted(handle.list_nodes(group), key=lambda node: int(node.name))
    return [torch.from_numpy(node.read()) for node in nodes]


def load_data(path):
    """Load all batches from the original SPI HDF5 layout."""
    with tables.open_file(path) as handle:
        inputs = _load_group(handle, '/inputs')
        starts = _load_group(handle, '/point_start')
        sim = _load_group(handle, '/sim')
        real = _load_group(handle, '/real')
    return (
        torch.cat(inputs).float(),
        torch.cat(starts).float(),
        torch.cat(sim).float() if sim else None,
        torch.cat(real).float() if real else None,
    )

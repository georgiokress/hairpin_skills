"""One explicit time base for conversion, training, evaluation and inversion."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

DT = 0.032
DEFAULT_HORIZON_SAMPLES = 640
LEGACY_HORIZON_SAMPLES = 256
CYCLE_REFERENCE_S = LEGACY_HORIZON_SAMPLES * DT


@dataclass(frozen=True)
class SequenceConfig:
    horizon_samples: int = DEFAULT_HORIZON_SAMPLES
    dt_s: float = DT
    cycle_reference_s: float = CYCLE_REFERENCE_S

    def __post_init__(self):
        if isinstance(self.horizon_samples, bool) or not isinstance(self.horizon_samples, int) or self.horizon_samples < 2:
            raise ValueError("horizon_samples must be an integer >= 2")
        if self.dt_s != DT:
            raise ValueError("bounds_v1 freezes dt_s at 0.032 s; a new raster needs separate validation")
        if not math.isfinite(self.cycle_reference_s) or self.cycle_reference_s <= 0:
            raise ValueError("cycle_reference_s must be finite and positive")

    @property
    def last_time_s(self):
        return (self.horizon_samples - 1) * self.dt_s

    def as_dict(self):
        return {
            "horizon_samples": self.horizon_samples,
            "dt_s": self.dt_s,
            "last_time_s": self.last_time_s,
            "cycle_reference_s": self.cycle_reference_s,
        }


def dataset_config(data_dir, horizon_samples=None):
    """Read HDF5 shape and sidecars; reject mixed horizons before any fit."""
    import tables
    h5_files = sorted(Path(data_dir).glob("*.h5"))
    if not h5_files:
        raise ValueError(f"No HDF5 files in {data_dir}")
    lengths = set()
    configurations = []
    for path in h5_files:
        with tables.open_file(path) as data:
            for group in ("/sim", "/real"):
                for node in data.list_nodes(group, classname="Array"):
                    if len(node.shape) != 3 or node.shape[2] != 15:
                        raise ValueError(f"Unexpected trajectory shape {node.shape} in {path}:{group}")
                    lengths.add(node.shape[1])
        sidecar = Path(f"{path}.runs.json")
        metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if "sequence" in metadata:
            seq = metadata["sequence"]
            configurations.append(SequenceConfig(int(seq["horizon_samples"]), float(seq["dt_s"]), float(seq["cycle_reference_s"])))
        else:
            # The frozen pilot format is the only supported metadata-free legacy case.
            configurations.append(SequenceConfig(LEGACY_HORIZON_SAMPLES))
    if len(lengths) != 1:
        raise ValueError(f"Mixed or missing trajectory horizons: {sorted(lengths)}")
    actual = next(iter(lengths))
    if any(c.horizon_samples != actual for c in configurations) or len(set(configurations)) != 1:
        raise ValueError("Dataset sequence sidecars are missing or inconsistent with the HDF5 tensors")
    config = configurations[0]
    if horizon_samples is not None and horizon_samples != actual:
        raise ValueError(f"Requested horizon {horizon_samples} does not match dataset horizon {actual}")
    return config


def checkpoint_config(model_dir, data_dir, horizon_samples=None):
    config = dataset_config(data_dir, horizon_samples)
    model = json.loads((Path(model_dir) / "model_config.json").read_text())
    trained = SequenceConfig(
        int(model["seq_len"]), float(model.get("dt_s", DT)),
        float(model.get("cycle_reference_s", CYCLE_REFERENCE_S)),
    )
    if trained != config:
        raise ValueError(f"Checkpoint sequence {trained} does not match dataset sequence {config}; reconvert/refit explicitly")
    return config

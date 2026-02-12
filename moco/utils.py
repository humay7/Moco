import numpy as np
import mlflow
import jax
import jax.numpy as jnp

from jax._src.lib import pytree
import pickle
from pathlib import Path
from typing import Union, Dict, Any, Mapping
import math

def save_pytree(data: pytree, path: Union[str, Path], overwrite: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_pytree(path: Union[str, Path]) -> pytree:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_jnpz(path, **kwargs):
    """Saves a dict of jnp arrays to a npz file"""
    with open(path, 'wb') as f:
        jnp.savez(f, **kwargs)

def load_jnpz(path):
    """Loads a dict of jnp arrays from a npz file"""
    with open(path, 'rb') as f:
        arrays = dict(jnp.load(f))
    return arrays

def parse_slice(s):
    a = [int(e) if e.strip() else None for e in s.split(":")]
    return slice(*a)

def average_dict_of_time_series(series):
    """"
    series: list of dicts of lists with same keys
    returns: dict of lists with same keys, where each list is the average of the corresponding lists in series
    """
    avg_series = {}
    for key in series[0].keys():
        avg_series[key] = np.mean([s[key] for s in series], axis=0)
    return avg_series

def mlflow_log_dict_of_lists(dict_of_lists):
    """Logs a dict of lists to mlflow"""
    for metric_name, metric_history in dict_of_lists.items():
        for i, val in enumerate(metric_history):
            # print(metric_name, i+1, val)
            mlflow.log_metric(metric_name, val, step=i+1)

def _to_python_float(value: Any) -> float:
    """Best-effort conversion of scalars/0-d arrays to Python float."""
    try:
        return float(value)
    except Exception:
        return float(jnp.asarray(value))

def mlflow_log_metrics_safe(metrics: Mapping[str, Any], step: int, flag_suffix: str = "_nonfinite", on_nonfinite: str = "skip") -> None:
    """
    Log metrics to MLflow while avoiding NaN/Inf values which break the UI.
    - Converts values to Python floats when possible.
    - If a value is non-finite:
        - on_nonfinite == "skip": drop the metric for this step
        - on_nonfinite == "zero": log 0.0 instead
        - on_nonfinite == "clip": clip to [-1e12, 1e12]
      In all cases, also logs a flag metric '<name><flag_suffix>' = 1.0 to signal the issue.
      Additionally logs:
        - '<name>_isnan' = 1.0 if value is NaN
        - '<name>_isinf' = 1.0 if value is +/-Inf
        (only when a non-finite value is encountered)
    """
    safe: Dict[str, float] = {}
    flags: Dict[str, float] = {}
    for name, val in metrics.items():
        try:
            fv = _to_python_float(val)
        except Exception:
            # Un-loggable type; mark and skip
            flags[f"{name}{flag_suffix}"] = 1.0
            continue
        if math.isfinite(fv):
            safe[name] = fv
        else:
            flags[f"{name}{flag_suffix}"] = 1.0
            # Type-specific flags for debugging
            if math.isnan(fv):
                flags[f"{name}_isnan"] = 1.0
            if math.isinf(fv):
                flags[f"{name}_isinf"] = 1.0
            if on_nonfinite == "zero":
                safe[name] = 0.0
            elif on_nonfinite == "clip":
                lim = 1e12
                safe[name] = max(-lim, min(lim, fv)) if not math.isnan(fv) else 0.0
            # else "skip": do not log the offending metric, only the flag
    # Combine and log
    if safe or flags:
        mlflow.log_metrics({**safe, **flags}, step=step)

def jax_has_gpu():
    """Returns True if jax can find a gpu, False otherwise"""
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except:
        return False

def pytree_repr(pytree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x).shape, pytree)


def dataclass_to_dict_of_lists(dataclass_list, stack=False):
    """given a list of dataclasses, return a dict of lists where each key is a field of the dataclass and each value is a list of the values of that field in the dataclasses"""
    dict_of_lists = {k: [getattr(dc, k) for dc in dataclass_list] for k in dataclass_list[0].__dataclass_fields__.keys()}
    if stack:
        dict_of_lists = {k: jnp.stack(v) for k, v in dict_of_lists.items()}
    return dict_of_lists

if __name__ == "__main__":
    series = [{"a": [1,2,3], "b": [4,5,6]}, {"a": [1,3,5], "b": [10,11,12]}]
    avg_series = average_dict_of_time_series(series)
    print(avg_series)
    
    assert np.allclose(avg_series["a"], [1,2.5,4])
    assert np.allclose(avg_series["b"], [7,8,9])
    print("Passed!")

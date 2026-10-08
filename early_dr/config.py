"""Configuration: one YAML file per experiment.

An experiment file only lists what it changes and points to its parent with
`base: ../base.yaml`. Extra changes can be given on the command line as
`--set key=value` (use dots for nested keys, e.g. `--set train.epochs=5`).
"""
import os
import yaml


class Config(dict):
    """A dict whose keys can also be read as attributes: cfg.train.epochs."""

    def __getattr__(self, name):
        value = self[name]
        return Config(value) if isinstance(value, dict) else value


def _merge(base, changes):
    out = dict(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("base", None)
    if parent:
        parent_path = os.path.join(os.path.dirname(path), parent)
        cfg = _merge(_read(parent_path), cfg)
    return cfg


def load_config(path, overrides=()):
    cfg = _read(path)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)  # "5" -> 5, "true" -> True
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return Config(cfg)

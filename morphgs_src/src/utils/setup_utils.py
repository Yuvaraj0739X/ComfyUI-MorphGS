# Copyright (c) 2026 MorphGS Authors.
# Licensed under the MIT License.

import os
import yaml
import ast
from pathlib import Path


DEFAULT_DATA_ROOT = os.environ.get("MORPHGS_DATA_ROOT", "demo")
PROCESSED_VIDEO_DIRNAME = "processed_videos"
MULTIVIEW_VIDEO_DIRNAME = "multiview_videos"
DEMO_CONFIG_DIR = Path("configs") / "demo"


class ProjectPath(object):
    def __init__(self, output_base, project_name, model_name, source_path, target_path):
        self._project_dir = os.path.join(output_base, project_name)
        self._src_dir = source_path
        data_root = None
        src_parent = os.path.dirname(source_path)
        if os.path.basename(src_parent) in {PROCESSED_VIDEO_DIRNAME, MULTIVIEW_VIDEO_DIRNAME}:
            data_root = os.path.dirname(src_parent)
        self._demo_cache_dir = (
            os.path.join(data_root, "cache", project_name)
            if data_root is not None
            else None
        )

        self._tgt_dir = target_path
        self._tgt_2d_feat_dir = os.path.join(target_path, "feature")

        self._tgt_render_dir = os.path.join(target_path, "color")

        self._tgt_feat_path = os.path.join(self._project_dir, "feats_3d.pt")
        self._demo_tgt_feat_path = (
            os.path.join(self._demo_cache_dir, "feats_3d.pt")
            if self._demo_cache_dir is not None
            else None
        )
        self._tgt_obj_path = os.path.join(self._tgt_dir, "mesh.obj")
        self._tgt_rig_path = os.path.join(self._tgt_dir, "rigging", "mesh_ori_rig.txt")

        self._encoder_path = os.path.join(self._project_dir, "encoder.pth")

        self._model_dir = os.path.join(self._project_dir, "model", model_name)
        self._gaussians_dir = os.path.join(self._model_dir, "gaussians")
        self._deform_dir = os.path.join(self._model_dir, "deform")
        self._pm_dir = os.path.join(self._model_dir, "parametric")
        self._render_dir = os.path.join(self._model_dir, "render")

        self._mapping_dir = os.path.join(self._project_dir, "mapping")
        self._demo_mapping_dir = (
            os.path.join(self._demo_cache_dir, "mapping")
            if self._demo_cache_dir is not None
            else None
        )

    def _create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def __getattr__(self, name):
        dir_attr = {
            "project_dir": self._project_dir,
            "src_dir": self._src_dir,
            "tgt_dir": self._tgt_dir,
            "tgt_2d_feat_dir": self._tgt_2d_feat_dir,
            "tgt_render_dir": self._tgt_render_dir,
            "model_dir": self._model_dir,
            "gaussians_dir": self._gaussians_dir,
            "deform_dir": self._deform_dir,
            "render_dir": self._render_dir,
            "pm_dir": self._pm_dir,
            "mapping_dir": self._mapping_dir,
        }

        if name in dir_attr:
            self._create_dir(dir_attr[name])
            return dir_attr[name]

        pathname = object.__getattribute__(self, f"_{name}")
        if pathname is None:
            return None
        self._create_dir(os.path.dirname(pathname))
        return pathname


class DotDict:
    """Dot notation access to dictionary attributes."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))  # convert nested dicts recursively
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)

    def to_dict(self):
        """Recursively convert DotDict back to a standard dictionary."""
        result = {}
        for key in self.__dict__:
            value = getattr(self, key)
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

def load_config(config_path):
    """
    Load configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return DotDict(config)


def save_config(config, config_path):
    """
    Save configuration to a YAML file.
    """
    if not isinstance(config, DotDict):
        raise TypeError("Config must be a DotDict instance")
    
    with open(config_path, 'w') as file:
        yaml.safe_dump(config.to_dict(), file, default_flow_style=False, allow_unicode=True)


def parse_override_arg(arg: str, config):
    """
    Parse an override argument of the form key.subkey=value and update config.
    """
    key, raw_val = arg.split('=', 1)
    key_chain = key.split('.')
    try:
        value = ast.literal_eval(raw_val)
    except Exception:
        try:
            value = yaml.safe_load(raw_val)
        except Exception:
            value = raw_val

    obj = config
    for k in key_chain[:-1]:
        obj = getattr(obj, k) if hasattr(obj, k) else obj[k]
    last = key_chain[-1]
    if isinstance(obj, DotDict):
        setattr(obj, last, value)
    else:
        obj[last] = value


def get_demo_config_path(experiment: str) -> str:
    return str(DEMO_CONFIG_DIR / f"{experiment}.yaml")


def resolve_config_path(config_path: str) -> str:
    if os.path.exists(config_path):
        return config_path

    normalized = config_path.replace("\\", "/")
    if normalized.startswith("demo/"):
        candidate = os.path.join("configs", normalized)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Config file does not exist: {config_path}")


def infer_experiment_from_config(config_path: str) -> str | None:
    normalized = config_path.replace("\\", "/")
    experiment = os.path.splitext(os.path.basename(normalized))[0]
    is_demo_config = (
        normalized.startswith("demo/")
        or normalized.startswith("configs/demo/")
        or "/configs/demo/" in normalized
    )
    if is_demo_config and "_to_" in experiment:
        return experiment
    return None


def split_experiment_name(experiment: str) -> tuple[str, str]:
    if "_to_" not in experiment:
        raise ValueError(
            f"Invalid experiment name '{experiment}'. Expected '<source>_to_<target>'."
        )
    source, target = experiment.rsplit("_to_", 1)
    if not source or not target:
        raise ValueError(
            f"Invalid experiment name '{experiment}'. Expected '<source>_to_<target>'."
        )
    return source, target


def get_data_bases(
    video_layout: str = "multiview",
    data_root: str | None = None,
) -> tuple[str, str]:
    if video_layout not in {"processed", "multiview"}:
        raise ValueError(f"Unsupported video_layout: {video_layout}")

    root = data_root if data_root is not None else DEFAULT_DATA_ROOT
    anim_dirname = (
        PROCESSED_VIDEO_DIRNAME
        if video_layout == "processed"
        else MULTIVIEW_VIDEO_DIRNAME
    )
    return os.path.join(root, anim_dirname), os.path.join(root, "characters")
        

import collections.abc

def merge_configs(base, new):
    """Recursively merges 'new' (DotDict/dict) onto 'base' (DotDict/dict)."""
    
    # Safely convert to a standard dict for iteration
    base_data = base.to_dict() if hasattr(base, 'to_dict') else base
    new_data = new.to_dict() if hasattr(new, 'to_dict') else new
    
    # Ensure both are mapping types before proceeding
    if not isinstance(base_data, collections.abc.Mapping) or \
       not isinstance(new_data, collections.abc.Mapping):
        return new  # New overwrites base if types are incompatible

    for k, v_new in new_data.items():
        v_base = base_data.get(k)
        
        # Check if both are dictionaries for recursion
        if isinstance(v_base, collections.abc.Mapping) and \
           isinstance(v_new, collections.abc.Mapping):
            base_data[k] = merge_configs(v_base, v_new)
        else:
            # Overwrite value
            base_data[k] = v_new
            
    # Return the result as a new DotDict
    return DotDict(base_data)

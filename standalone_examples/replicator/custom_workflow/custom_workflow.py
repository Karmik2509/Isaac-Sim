# SPDX-License-Identifier: Apache-2.0

"""Generate RGB, 2D-tight-box, and 3D-box data for one warehouse asset.

This workflow is intended for a local Isaac Sim stage.  It opens the supplied
warehouse USD, finds the prim named ``RackLargeEmpty_A1`` (including nested
prim paths), labels it, and captures it from one randomized camera.  Unlike
the stock scene-based SDG example, it does not create pallet/top cameras or
additional warehouse props.

Example::

    ./python.sh standalone_examples/replicator/custom_workflow/custom_workflow.py \
        --config standalone_examples/replicator/custom_workflow/config.yaml
"""

import argparse
import json
import math
import os
from pathlib import Path

import carb
import numpy as np
import yaml
from isaacsim import SimulationApp


DEFAULT_CONFIG = {
    "launch_config": {
        "renderer": "RealTimePathTracing",
        "headless": False,
    },
    "resolution": [512, 512],
    "rt_subframes": 32,
    "num_frames": 10,
    "env_url": "/home/i2v/Downloads/Industrial_NVD@10012/Assets/ArchVis/Industrial/Stages/IsaacWarehouse.usd",
    "target_prim_name": "RackLargeEmpty_A1",
    "target_class": "RackLargeEmpty_A1",
    "writer": "BasicWriter",
    "backend_type": "DiskBackend",
    "backend_params": {
        "output_dir": "_out_custom_workflow",
    },
    # Deliberately limited to RGB and the two requested bounding-box outputs.
    "writer_config": {
        "rgb": True,
        "bounding_box_2d_tight": True,
        "bounding_box_3d": True,
    },
    "clear_previous_semantics": True,
    "camera": {
        "focal_length": 28.0,
        "clipping_range": [0.1, 100000.0],
        "radius_scale_min": 1.5,
        "radius_scale_max": 2.5,
        "height_min": 0.35,
        "height_max": 0.8,
    },
    "close_app_after_run": True,
}


def load_config(config_path: str | None) -> dict:
    """Load a JSON/YAML config and merge it over the defaults."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if not config_path:
        return config

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as stream:
        overrides = json.load(stream) if path.suffix.lower() == ".json" else yaml.safe_load(stream)
    if not isinstance(overrides, dict):
        raise ValueError("Config must contain a mapping/object at its root")

    # Keep nested defaults when only part of a nested section is overridden.
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def find_prim_by_name(stage, prim_name: str):
    """Find a unique prim by leaf name, or fail with useful diagnostics."""
    matches = [prim for prim in stage.Traverse() if prim.GetName() == prim_name]
    if not matches:
        raise RuntimeError(f"Could not find prim named '{prim_name}' in the loaded stage")
    if len(matches) > 1:
        paths = ", ".join(str(prim.GetPath()) for prim in matches[:10])
        raise RuntimeError(
            f"Found {len(matches)} prims named '{prim_name}'. Set a unique target_prim_name "
            f"or extend this script to accept a full path. Matches: {paths}"
        )
    return matches[0]


def get_world_bounds(prim, bbox_cache):
    """Return world-space min, max, center, and size for a USD prim."""
    bound = bbox_cache.ComputeWorldBound(prim)
    box = bound.ComputeAlignedRange()
    minimum = np.asarray(box.GetMin(), dtype=np.float64)
    maximum = np.asarray(box.GetMax(), dtype=np.float64)
    size = np.maximum(maximum - minimum, 1e-3)
    center = (minimum + maximum) / 2.0
    return minimum, maximum, center, size


def setup_writer(config: dict):
    """Create the configured Replicator writer and optional backend."""
    import omni.replicator.core as rep

    writer_type = config.get("writer", "BasicWriter")
    if writer_type not in rep.WriterRegistry.get_writers():
        raise RuntimeError(f"Writer type '{writer_type}' is not registered")

    def absolute_output(params: dict) -> dict:
        params = dict(params)
        if "output_dir" in params and not os.path.isabs(params["output_dir"]):
            params["output_dir"] = os.path.abspath(params["output_dir"])
        return params

    writer_kwargs = absolute_output(config.get("writer_config", {}))
    backend = None
    backend_type = config.get("backend_type")
    if backend_type:
        backend = rep.backends.get(backend_type)
        backend_params = absolute_output(config.get("backend_params", {}))
        backend.initialize(**backend_params)
        print(f"[Custom SDG] Backend: {backend_type} | Params: {backend_params}")

    writer = rep.WriterRegistry.get(writer_type)
    if backend is None:
        writer.initialize(**writer_kwargs)
    else:
        writer.initialize(backend=backend, **writer_kwargs)
    print(f"[Custom SDG] Writer: {writer_type} | Config: {writer_kwargs}")
    return writer


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", help="JSON or YAML configuration file")
args, _ = parser.parse_known_args()
config = load_config(args.config)

stage_path = Path(config["env_url"]).expanduser()
if not stage_path.is_file():
    raise FileNotFoundError(f"Warehouse USD does not exist: {stage_path}")

simulation_app = SimulationApp(launch_config=config["launch_config"])

# Runtime imports must happen after SimulationApp is initialized.
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.experimental.utils.semantics import add_labels, remove_all_labels
from isaacsim.core.experimental.utils.stage import get_current_stage, open_stage
from isaacsim.core.experimental.utils.bounds import create_bbox_cache


try:
    carb.log_info(f"[Custom SDG] Loading stage: {stage_path}")
    stage_opened, _ = open_stage(str(stage_path))
    if not stage_opened:
        raise RuntimeError(f"Could not open stage: {stage_path}")
    simulation_app.update()

    stage = get_current_stage()
    if config.get("clear_previous_semantics", True):
        for prim in stage.Traverse():
            remove_all_labels(prim, include_descendants=True)

    target = find_prim_by_name(stage, config["target_prim_name"])
    target_path = str(target.GetPath())
    add_labels(target, labels=[config["target_class"]], taxonomy="class")
    print(f"[Custom SDG] Target: {target_path} | class: {config['target_class']}")

    bbox_cache = create_bbox_cache()
    _, _, target_center, target_size = get_world_bounds(target, bbox_cache)
    target_radius = float(max(target_size[0], target_size[1], target_size[2]))

    # One camera only; no pallet or top camera is created.
    rep.functional.create.scope(name="CustomSDG")
    camera_cfg = config.get("camera", {})
    camera = rep.functional.create.camera(
        focal_length=float(camera_cfg.get("focal_length", 28.0)),
        clipping_range=tuple(camera_cfg.get("clipping_range", [0.1, 100000.0])),
        name="TargetCamera",
        parent="/CustomSDG",
    )
    resolution = config.get("resolution", [512, 512])
    render_product = rep.create.render_product(camera, resolution, name="TargetView")
    render_product.hydra_texture.set_updates_enabled(False)

    writer = setup_writer(config)
    writer.attach([render_product])
    render_product.hydra_texture.set_updates_enabled(True)

    rep.set_global_seed(42)
    rep.orchestrator.set_capture_on_play(False)
    rng = np.random.default_rng(42)
    rt_subframes = config.get("rt_subframes", -1)
    num_frames = int(config.get("num_frames", 0))

    radius_min = float(camera_cfg.get("radius_scale_min", 1.5)) * target_radius
    radius_max = float(camera_cfg.get("radius_scale_max", 2.5)) * target_radius
    height_min = float(camera_cfg.get("height_min", 0.35))
    height_max = float(camera_cfg.get("height_max", 0.8))

    print(f"[Custom SDG] Capturing {num_frames} frames at {resolution}")
    for frame_index in range(num_frames):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(radius_min, radius_max)
        height = rng.uniform(height_min, height_max) * target_size[2]
        camera_position = target_center + np.array(
            [radius * math.cos(angle), radius * math.sin(angle), height], dtype=np.float64
        )

        rep.functional.modify.pose(
            camera,
            position_value=tuple(camera_position.tolist()),
            look_at_value=target,
            look_at_up_axis=(0, 0, 1),
        )
        print(f"[Custom SDG] Frame {frame_index + 1}/{num_frames}")
        rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes)

    rep.orchestrator.wait_until_complete()
    writer.detach()
    render_product.destroy()

finally:
    if config.get("close_app_after_run", True) or config["launch_config"].get("headless", False):
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
        simulation_app.close()

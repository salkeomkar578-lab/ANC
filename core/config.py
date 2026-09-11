"""
IGARD-Net Configuration Module
Loads, validates, and provides structured access to system parameters.
Supports hardware profiles: DESKTOP_NVIDIA, JETSON_NANO, CPU_ONLY.
"""

from pathlib import Path
from typing import Any, Dict
import copy

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

DEFAULT_CONFIG: Dict[str, Any] = {
    "audio": {
        "sample_rate": 16000,
        "frame_size": 256,
        "hop_size": 128,
        "channels": 1,
        "bit_depth": 16,
    },
    "nlms": {
        "taps": 64,
        "initial_mu": 0.25,
        "epsilon": 1.0e-6,
        "leakage": 0.9999,
        "adaptation_enabled": True,
    },
    "gqpso": {
        "enabled": True,
        "num_particles": 8,
        "iterations": 5,
        "interval_s": 1.0,
        "history_samples": 1024,
        "search_min": 0.01,
        "search_max": 1.5,
        "mutation_prob": 0.1,
    },
    "presence_gate": {
        "threshold_margin": 2.5,
        "calibration_blocks": 20,
        "initial_floor": 1.0e-5,
        "hysteresis_db": 2.0,
    },
    "classifier": {
        "confidence_threshold": 0.60,
        "onset_threshold": 0.01,
    },
    "wiener": {
        "enabled": True,
        "gate_strength": 1.4,
        "min_gain": 0.08,
        "smoothing_factor": 0.85,
    },
    "voice_protection": {
        "enabled": True,
        "speech_gain_floor_db": -18.0,
        "compressor_threshold_db": -6.0,
        "compressor_ratio": 3.0,
        "attack_ms": 5.0,
        "release_ms": 40.0,
        "makeup_gain_db": 1.5,
    },
    "accelerometer": {
        "shock_threshold": 0.5,
        "agreement_boost": 0.25,
        "hardware_bus": 1,
    },
    "autopilot": {
        "enabled": True,
    },
    "dashboard": {
        "telemetry_fps": 20,
        "host": "0.0.0.0",
        "port": 5000,
    },
    "hardware": {
        "profile": "auto",
        "cuda_device_id": 0,
        "allow_cpu_fallback": True,
    },
}

# Profile Overrides
PROFILE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "JETSON_NANO": {
        "audio": {"frame_size": 256},
        "gqpso": {"interval_s": 2.0, "iterations": 4, "num_particles": 6, "history_samples": 512},
        "dashboard": {"telemetry_fps": 15},
    },
    "DESKTOP_NVIDIA": {
        "audio": {"frame_size": 256},
        "gqpso": {"interval_s": 1.0, "iterations": 6, "num_particles": 8, "history_samples": 1024},
        "dashboard": {"telemetry_fps": 20},
    },
    "CPU_ONLY": {
        "audio": {"frame_size": 256},
        "gqpso": {"interval_s": 1.5, "iterations": 4, "num_particles": 6, "history_samples": 512},
        "dashboard": {"telemetry_fps": 15},
    }
}


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load config from YAML file or return defaults with profile adjustments."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    if config_path.exists() and YAML_AVAILABLE:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f)
            if user_cfg and isinstance(user_cfg, dict):
                _deep_update(cfg, user_cfg)
        except Exception as e:
            print(f"[Config] Warning: Failed to parse {config_path} ({e}), using default config.")

    return cfg


def apply_profile(cfg: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    """Apply hardware-specific profile overrides."""
    if profile_name in PROFILE_CONFIGS:
        _deep_update(cfg, PROFILE_CONFIGS[profile_name])
        cfg["hardware"]["profile"] = profile_name
    return cfg


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for k, v in source.items():
        if isinstance(v, dict) and k in target and isinstance(target[k], dict):
            _deep_update(target[k], v)
        else:
            target[k] = v



from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open(
        "r", encoding="utf-8"
    ) as f:
        cfg = yaml.safe_load(f)

    required = [
        ("model", "path"),
        ("dataset", "calibration_path"),
        ("dataset", "evaluation_path"),
        ("quantization", "group_size"),
        ("calibration", "num_samples"),
        ("calibration", "sequence_length"),
        ("evaluation", "sequence_length"),
    ]

    for section, key in required:
        if section not in cfg or key not in cfg[section]:
            raise KeyError(
                f"Missing config entry: {section}.{key}"
            )

    return cfg

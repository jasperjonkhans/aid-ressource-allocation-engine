from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_config(config: dict[str, Any], path: str | Path = CONFIG_PATH) -> None:
    Path(path).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def pretty_config(config: Any | None = None) -> str:
    if config is None:
        config = load_config()
    return json.dumps(config, indent=2, sort_keys=True)


def validate_config_value(old_value: Any, new_value: Any) -> None:
    if old_value is None or new_value is None:
        return
    if isinstance(old_value, bool):
        valid = isinstance(new_value, bool)
    elif isinstance(old_value, (int, float)) and not isinstance(old_value, bool):
        valid = isinstance(new_value, (int, float)) and not isinstance(new_value, bool)
    else:
        valid = isinstance(new_value, type(old_value))
    if not valid:
        raise TypeError(
            f"Expected {type(old_value).__name__} for this key, "
            f"got {type(new_value).__name__}."
        )


def parse_config_value(raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def get_config_value(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def set_config_value(
    dotted_key: str,
    value: Any,
    *,
    path: str | Path = CONFIG_PATH,
) -> tuple[Any, Any]:
    config = load_config(path)
    updated = deepcopy(config)
    current: Any = updated
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]

    leaf = parts[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise KeyError(dotted_key)

    old_value = current[leaf]
    validate_config_value(old_value, value)
    current[leaf] = value
    save_config(updated, path)
    return old_value, value


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


CONFIG = load_config()

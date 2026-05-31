from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(__file__).with_name("config.json")

EDITABLE_CONFIG_PREFIX = "agent."

CONFIG_DESCRIPTIONS = {
    "agent.cargo_types": "Hilfsgueter, auf die der Agent Budget verteilen kann.",
    "agent.weights.water_supplies": "Multiplikator fuer kurzfristigen Wasserdruck.",
    "agent.weights.water_infrastructure": "Multiplikator fuer strukturellen Wasserdruck.",
    "agent.weights.food_supplies": "Multiplikator fuer Nahrungsmitteldruck.",
    "agent.weights.fuel": "Multiplikator fuer Treibstoffdruck.",
    "agent.slope_weight": "Einfluss von Forecast-Trends auf Scores.",
    "agent.level_weight": "Einfluss hoher Forecast-Level auf Scores.",
    "agent.uncertainty_weight": "Einfluss von Forecast-Unsicherheit auf Scores.",
    "agent.drought_weight": "Einfluss von Duerredruck auf Wasser und Nahrung.",
    "agent.drought_temperature_weight": "Einfluss steigender Temperatur auf den Duerrescore.",
    "agent.drought_rainfall_weight": "Einfluss von Regen auf den Duerrescore.",
    "agent.drought_humidity_weight": "Einfluss von Luftfeuchtigkeit auf den Duerrescore.",
    "agent.missing_weather_drought_score": "Fallback-Duerrescore, wenn keine Wetterprognose vorhanden ist.",
    "agent.food_drought_weight": "Einfluss von Duerredruck auf Nahrungsscores.",
    "agent.fuel_cross_sector_weight": "Einfluss von Wasser- und Nahrungsscores auf Treibstoff.",
    "agent.softmax_temperature": "Steuert, wie stark Budget zu hohen Scores kippt.",
    "agent.total_budget": "Gesamtbudget, das auf Regionen verteilt wird.",
    "agent.good_unit_costs.water_supplies": "Basis-Lieferkosten fuer Wassergueter.",
    "agent.good_unit_costs.water_infrastructure": "Basis-Lieferkosten fuer Wasserinfrastruktur.",
    "agent.good_unit_costs.food_supplies": "Basis-Lieferkosten fuer Nahrungsmittel.",
    "agent.good_unit_costs.fuel": "Basis-Lieferkosten fuer Treibstoff.",
    "agent.region_populations.Gedo": "Bevoelkerung fuer Gedo-Budgetgewichtung.",
    "agent.region_populations.Buur Hakaba": "Bevoelkerung fuer Buur-Hakaba-Budgetgewichtung.",
    "agent.region_populations.Bakool": "Bevoelkerung fuer Bakool-Budgetgewichtung.",
    "agent.region_accessibility.Gedo": "Erreichbarkeitsfaktor fuer Gedo-Lieferkosten.",
    "agent.region_accessibility.Buur Hakaba": "Erreichbarkeitsfaktor fuer Buur-Hakaba-Lieferkosten.",
    "agent.region_accessibility.Bakool": "Erreichbarkeitsfaktor fuer Bakool-Lieferkosten.",
}


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


def format_config_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return repr(value)


def flatten_config_values(value: Any, prefix: str) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows = []
        for key, child in value.items():
            rows.extend(flatten_config_values(child, f"{prefix}.{key}"))
        return rows
    return [(prefix, value)]


def agent_config_rows(config: dict[str, Any] | None = None) -> list[tuple[str, str, Any]]:
    config = config or load_config()
    return [
        (key, CONFIG_DESCRIPTIONS.get(key, "Agent-Konfigurationswert."), value)
        for key, value in flatten_config_values(config["agent"], "agent")
    ]


def human_readable_agent_config(config: dict[str, Any] | None = None) -> str:
    rows = agent_config_rows(config)
    key_width = max(len(key) for key, _, _ in rows)
    lines = [
        "Agent-Konstanten (aenderbar)",
        f"Config-Datei: {CONFIG_PATH}",
        "",
    ]
    for key, description, value in rows:
        lines.append(f"- {key:<{key_width}}  {format_config_value(value)}")
        lines.append(f"  {description}")
    return "\n".join(lines)


def is_editable_config_key(dotted_key: str) -> bool:
    return dotted_key.startswith(EDITABLE_CONFIG_PREFIX)


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
    if not is_editable_config_key(dotted_key):
        raise PermissionError("Nur Agent-Konstanten duerfen geaendert werden. Nutze Keys mit 'agent.'.")

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
    if isinstance(old_value, dict):
        raise TypeError(f"{dotted_key} ist eine Section. Aendere einen konkreten Key darin.")
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

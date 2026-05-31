from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from project.config import CONFIG

_AGENT_CONFIG = CONFIG["agent"]

CARGO_TYPES = tuple(_AGENT_CONFIG["cargo_types"])
WEIGHT_WATER_SUPPLIES = float(_AGENT_CONFIG["weights"]["water_supplies"])
WEIGHT_WATER_INFRASTRUCTURE = float(_AGENT_CONFIG["weights"]["water_infrastructure"])
WEIGHT_FOOD_SUPPLIES = float(_AGENT_CONFIG["weights"]["food_supplies"])
WEIGHT_FUEL = float(_AGENT_CONFIG["weights"]["fuel"])

SLOPE_WEIGHT = float(_AGENT_CONFIG["slope_weight"])
LEVEL_WEIGHT = float(_AGENT_CONFIG["level_weight"])
UNCERTAINTY_WEIGHT = float(_AGENT_CONFIG["uncertainty_weight"])
DROUGHT_WEIGHT = float(_AGENT_CONFIG["drought_weight"])
DROUGHT_TEMPERATURE_WEIGHT = float(_AGENT_CONFIG["drought_temperature_weight"])
DROUGHT_RAINFALL_WEIGHT = float(_AGENT_CONFIG["drought_rainfall_weight"])
DROUGHT_HUMIDITY_WEIGHT = float(_AGENT_CONFIG["drought_humidity_weight"])
MISSING_WEATHER_DROUGHT_SCORE = float(_AGENT_CONFIG["missing_weather_drought_score"])
FOOD_DROUGHT_WEIGHT = float(_AGENT_CONFIG["food_drought_weight"])
FUEL_CROSS_SECTOR_WEIGHT = float(_AGENT_CONFIG["fuel_cross_sector_weight"])
SOFTMAX_TEMPERATURE = float(_AGENT_CONFIG["softmax_temperature"])
TOTAL_BUDGET = float(_AGENT_CONFIG["total_budget"])
TOTAL_UNITS = int(TOTAL_BUDGET)
GOOD_UNIT_COSTS = {key: float(value) for key, value in _AGENT_CONFIG["good_unit_costs"].items()}
REGION_POPULATIONS = {key: int(value) for key, value in _AGENT_CONFIG["region_populations"].items()}
REGION_ACCESSIBILITY = {key: float(value) for key, value in _AGENT_CONFIG["region_accessibility"].items()}


@dataclass(frozen=True)
class AgentPredictionBundle:
    region: str
    water: Any
    food: Any
    fuel: Any | None = None
    weather: Any | None = None


@dataclass(frozen=True)
class AgentDecision:
    region: str
    allocation: dict[str, float]
    budget_allocation: dict[str, float]
    scores: dict[str, float]
    selected_cargo: str
    regional_budget: float
    accessibility: float
    effective_unit_costs: dict[str, float]
    regional_units: float
    population: int | None = None
    reasoning: dict[str, list[dict[str, Any]]] | None = None
    formulas: dict[str, str] | None = None


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def softmax(scores: dict[str, float], temperature: float = SOFTMAX_TEMPERATURE) -> dict[str, float]:
    if not scores:
        return {}

    keys = list(scores)
    values = np.array([scores[key] for key in keys], dtype=float) / temperature
    values = values - values.max()
    exp_values = np.exp(values)
    shares = exp_values / exp_values.sum()
    return {key: float(share) for key, share in zip(keys, shares, strict=True)}


def _forecast_frame(prediction: Any) -> pd.DataFrame:
    if prediction is None:
        return pd.DataFrame()
    if isinstance(prediction, pd.DataFrame):
        return prediction.copy()
    forecast = getattr(prediction, "forecast", None)
    if forecast is None:
        return pd.DataFrame()
    return forecast.copy()


def _history_frame(prediction: Any) -> pd.DataFrame:
    if prediction is None:
        return pd.DataFrame()
    history = getattr(prediction, "history", None)
    if history is None:
        return pd.DataFrame()
    return history.copy()


def _weather_forecasts(weather: Any) -> dict[str, pd.DataFrame]:
    if weather is None:
        return {}
    forecasts = getattr(weather, "forecasts", None)
    if isinstance(forecasts, dict):
        return {key: value.copy() for key, value in forecasts.items()}
    return {}


def _numeric_series(frame: pd.DataFrame, column: str = "forecast") -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna().astype(float)


def _history_level(history: pd.DataFrame, forecast: pd.DataFrame) -> float:
    history_values = _numeric_series(history, "value")
    forecast_values = _numeric_series(forecast, "forecast")
    if history_values.empty or forecast_values.empty:
        return 0.0

    baseline = float(history_values.median())
    spread = float(history_values.quantile(0.75) - history_values.quantile(0.25))
    if spread <= 0:
        spread = max(abs(baseline) * 0.1, 1.0)
    return float((forecast_values.mean() - baseline) / spread)


def _forecast_slope(forecast: pd.DataFrame) -> float:
    values = _numeric_series(forecast, "forecast")
    if len(values) < 2:
        return 0.0

    first = float(values.iloc[0])
    last = float(values.iloc[-1])
    denominator = max(abs(first), 1.0)
    return float((last - first) / denominator)


def _forecast_uncertainty(forecast: pd.DataFrame) -> float:
    if forecast.empty:
        return 0.0

    low_column = "q10" if "q10" in forecast.columns else "q05" if "q05" in forecast.columns else None
    high_column = "q90" if "q90" in forecast.columns else "q95" if "q95" in forecast.columns else None
    if low_column is None or high_column is None:
        return 0.0

    low = _numeric_series(forecast, low_column)
    high = _numeric_series(forecast, high_column)
    mid = _numeric_series(forecast, "forecast").abs()
    if low.empty or high.empty or mid.empty:
        return 0.0

    width = float((high.reset_index(drop=True) - low.reset_index(drop=True)).mean())
    denominator = max(float(mid.mean()), 1.0)
    return max(width / denominator, 0.0)


def _weather_slope(weather: Any, metric_name: str) -> float:
    frame = _weather_forecasts(weather).get(metric_name, pd.DataFrame())
    return _forecast_slope(frame)


def _drought_score(weather: Any) -> float:
    if weather is None:
        return MISSING_WEATHER_DROUGHT_SCORE

    temperature = _weather_slope(weather, "temperature_avg_c")
    rainfall = _weather_slope(weather, "rainfall_mm_per_day")
    humidity = _weather_slope(weather, "relative_humidity_pct")
    return sigmoid(
        DROUGHT_TEMPERATURE_WEIGHT * temperature
        - DROUGHT_RAINFALL_WEIGHT * rainfall
        - DROUGHT_HUMIDITY_WEIGHT * humidity
    )


def compute_features(predictions: AgentPredictionBundle) -> dict[str, float]:
    water_forecast = _forecast_frame(predictions.water)
    water_history = _history_frame(predictions.water)
    food_forecast = _forecast_frame(predictions.food)
    food_history = _history_frame(predictions.food)
    fuel_forecast = _forecast_frame(predictions.fuel)
    fuel_history = _history_frame(predictions.fuel)

    temperature_slope = _weather_slope(predictions.weather, "temperature_avg_c")
    rainfall_slope = _weather_slope(predictions.weather, "rainfall_mm_per_day")
    humidity_slope = _weather_slope(predictions.weather, "relative_humidity_pct")

    return {
        "water_slope": _forecast_slope(water_forecast),
        "water_level": _history_level(water_history, water_forecast),
        "water_uncertainty": _forecast_uncertainty(water_forecast),
        "food_slope": _forecast_slope(food_forecast),
        "food_level": _history_level(food_history, food_forecast),
        "food_uncertainty": _forecast_uncertainty(food_forecast),
        "fuel_slope": _forecast_slope(fuel_forecast),
        "fuel_level": _history_level(fuel_history, fuel_forecast),
        "fuel_uncertainty": _forecast_uncertainty(fuel_forecast),
        "drought_score": _drought_score(predictions.weather),
        "temperature_slope": temperature_slope,
        "rainfall_slope": rainfall_slope,
        "humidity_slope": humidity_slope,
    }


def compute_score_components(features: dict[str, float]) -> dict[str, list[dict[str, float | str]]]:
    return {
        "water_supplies": [
            {
                "feature": "water_slope",
                "label": "water price trend",
                "value": features["water_slope"],
                "weight": SLOPE_WEIGHT,
                "contribution": SLOPE_WEIGHT * features["water_slope"],
            },
            {
                "feature": "water_level",
                "label": "water price level",
                "value": features["water_level"],
                "weight": LEVEL_WEIGHT,
                "contribution": LEVEL_WEIGHT * features["water_level"],
            },
            {
                "feature": "drought_score",
                "label": "drought pressure",
                "value": features["drought_score"],
                "weight": DROUGHT_WEIGHT,
                "contribution": DROUGHT_WEIGHT * features["drought_score"],
            },
            {
                "feature": "water_uncertainty",
                "label": "water forecast uncertainty",
                "value": features["water_uncertainty"],
                "weight": UNCERTAINTY_WEIGHT,
                "contribution": UNCERTAINTY_WEIGHT * features["water_uncertainty"],
            },
        ],
        "water_infrastructure": [
            {
                "feature": "water_level",
                "label": "water price level",
                "value": features["water_level"],
                "weight": LEVEL_WEIGHT,
                "contribution": LEVEL_WEIGHT * features["water_level"],
            },
            {
                "feature": "drought_score",
                "label": "drought pressure",
                "value": features["drought_score"],
                "weight": DROUGHT_WEIGHT,
                "contribution": DROUGHT_WEIGHT * features["drought_score"],
            },
            {
                "feature": "water_uncertainty",
                "label": "water forecast uncertainty",
                "value": features["water_uncertainty"],
                "weight": UNCERTAINTY_WEIGHT,
                "contribution": UNCERTAINTY_WEIGHT * features["water_uncertainty"],
            },
        ],
        "food_supplies": [
            {
                "feature": "food_slope",
                "label": "food price trend",
                "value": features["food_slope"],
                "weight": SLOPE_WEIGHT,
                "contribution": SLOPE_WEIGHT * features["food_slope"],
            },
            {
                "feature": "food_level",
                "label": "food price level",
                "value": features["food_level"],
                "weight": LEVEL_WEIGHT,
                "contribution": LEVEL_WEIGHT * features["food_level"],
            },
            {
                "feature": "drought_score",
                "label": "drought pressure",
                "value": features["drought_score"],
                "weight": FOOD_DROUGHT_WEIGHT,
                "contribution": FOOD_DROUGHT_WEIGHT * features["drought_score"],
            },
            {
                "feature": "food_uncertainty",
                "label": "food forecast uncertainty",
                "value": features["food_uncertainty"],
                "weight": UNCERTAINTY_WEIGHT,
                "contribution": UNCERTAINTY_WEIGHT * features["food_uncertainty"],
            },
        ],
        "fuel": [
            {
                "feature": "fuel_slope",
                "label": "fuel price trend",
                "value": features["fuel_slope"],
                "weight": SLOPE_WEIGHT,
                "contribution": SLOPE_WEIGHT * features["fuel_slope"],
            },
            {
                "feature": "fuel_level",
                "label": "fuel price level",
                "value": features["fuel_level"],
                "weight": LEVEL_WEIGHT,
                "contribution": LEVEL_WEIGHT * features["fuel_level"],
            },
            {
                "feature": "cross_sector_pressure",
                "label": "water and food pressure spillover",
                "value": _cross_sector_pressure(features),
                "weight": FUEL_CROSS_SECTOR_WEIGHT,
                "contribution": FUEL_CROSS_SECTOR_WEIGHT * _cross_sector_pressure(features),
            },
            {
                "feature": "fuel_uncertainty",
                "label": "fuel forecast uncertainty",
                "value": features["fuel_uncertainty"],
                "weight": UNCERTAINTY_WEIGHT,
                "contribution": UNCERTAINTY_WEIGHT * features["fuel_uncertainty"],
            },
        ],
    }


def _component_pressure(components: list[dict[str, float | str]]) -> float:
    return sigmoid(sum(float(component["contribution"]) for component in components))


def _cross_sector_pressure(features: dict[str, float]) -> float:
    components = compute_score_components_without_fuel_cross_sector(features)
    return float(
        np.mean(
            [
                _component_pressure(components["water_supplies"]),
                _component_pressure(components["water_infrastructure"]),
                _component_pressure(components["food_supplies"]),
            ]
        )
    )


def compute_score_components_without_fuel_cross_sector(
    features: dict[str, float],
) -> dict[str, list[dict[str, float | str]]]:
    return {
        "water_supplies": [
            {
                "feature": "water_slope",
                "label": "water price trend",
                "value": features["water_slope"],
                "weight": SLOPE_WEIGHT,
                "contribution": SLOPE_WEIGHT * features["water_slope"],
            },
            {
                "feature": "water_level",
                "label": "water price level",
                "value": features["water_level"],
                "weight": LEVEL_WEIGHT,
                "contribution": LEVEL_WEIGHT * features["water_level"],
            },
            {
                "feature": "drought_score",
                "label": "drought pressure",
                "value": features["drought_score"],
                "weight": DROUGHT_WEIGHT,
                "contribution": DROUGHT_WEIGHT * features["drought_score"],
            },
            {
                "feature": "water_uncertainty",
                "label": "water forecast uncertainty",
                "value": features["water_uncertainty"],
                "weight": UNCERTAINTY_WEIGHT,
                "contribution": UNCERTAINTY_WEIGHT * features["water_uncertainty"],
            },
        ],
        "water_infrastructure": [
            {
                "feature": "water_level",
                "label": "water price level",
                "value": features["water_level"],
                "weight": LEVEL_WEIGHT,
                "contribution": LEVEL_WEIGHT * features["water_level"],
            },
            {
                "feature": "drought_score",
                "label": "drought pressure",
                "value": features["drought_score"],
                "weight": DROUGHT_WEIGHT,
                "contribution": DROUGHT_WEIGHT * features["drought_score"],
            },
            {
                "feature": "water_uncertainty",
                "label": "water forecast uncertainty",
                "value": features["water_uncertainty"],
                "weight": UNCERTAINTY_WEIGHT,
                "contribution": UNCERTAINTY_WEIGHT * features["water_uncertainty"],
            },
        ],
        "food_supplies": [
            {
                "feature": "food_slope",
                "label": "food price trend",
                "value": features["food_slope"],
                "weight": SLOPE_WEIGHT,
                "contribution": SLOPE_WEIGHT * features["food_slope"],
            },
            {
                "feature": "food_level",
                "label": "food price level",
                "value": features["food_level"],
                "weight": LEVEL_WEIGHT,
                "contribution": LEVEL_WEIGHT * features["food_level"],
            },
            {
                "feature": "drought_score",
                "label": "drought pressure",
                "value": features["drought_score"],
                "weight": FOOD_DROUGHT_WEIGHT,
                "contribution": FOOD_DROUGHT_WEIGHT * features["drought_score"],
            },
            {
                "feature": "food_uncertainty",
                "label": "food forecast uncertainty",
                "value": features["food_uncertainty"],
                "weight": UNCERTAINTY_WEIGHT,
                "contribution": UNCERTAINTY_WEIGHT * features["food_uncertainty"],
            },
        ],
    }


def compute_scores(features: dict[str, float]) -> dict[str, float]:
    components = compute_score_components(features)

    return {
        "water_supplies": WEIGHT_WATER_SUPPLIES * _component_pressure(components["water_supplies"]),
        "water_infrastructure": WEIGHT_WATER_INFRASTRUCTURE
        * _component_pressure(components["water_infrastructure"]),
        "food_supplies": WEIGHT_FOOD_SUPPLIES * _component_pressure(components["food_supplies"]),
        "fuel": WEIGHT_FUEL * _component_pressure(components["fuel"]),
    }


def _region_lookup(region_name: str, values: dict[str, Any]) -> Any | None:
    normalized = region_name.lower().replace("_", " ").replace("-", " ")
    for region, value in values.items():
        if region.lower().replace("_", " ").replace("-", " ") == normalized:
            return value
    return None


def accessibility_for_region(
    region_name: str,
    *,
    accessibilities: dict[str, float] | None = None,
) -> float:
    accessibilities = accessibilities or REGION_ACCESSIBILITY
    accessibility = _region_lookup(region_name, accessibilities)
    if accessibility is None:
        return 1.0
    if accessibility <= 0:
        raise ValueError(f"Accessibility must be positive for {region_name!r}.")
    return float(accessibility)


def effective_unit_costs(
    *,
    accessibility: float,
    unit_costs: dict[str, float] | None = None,
) -> dict[str, float]:
    if accessibility <= 0:
        raise ValueError("Accessibility must be positive.")
    unit_costs = unit_costs or GOOD_UNIT_COSTS
    return {
        cargo_type: float(unit_costs[cargo_type]) / accessibility
        for cargo_type in CARGO_TYPES
    }


def allocate_budget(
    scores: dict[str, float],
    *,
    budget: float = TOTAL_BUDGET,
    unit_costs: dict[str, float] | None = None,
    accessibility: float = 1.0,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    costs = effective_unit_costs(accessibility=accessibility, unit_costs=unit_costs)
    budget_pressure = {cargo_type: scores[cargo_type] for cargo_type in CARGO_TYPES}
    shares = softmax(budget_pressure)
    budget_allocation = {
        cargo_type: round(shares.get(cargo_type, 0.0) * budget, 2)
        for cargo_type in CARGO_TYPES
    }
    diff = round(budget - sum(budget_allocation.values()), 2)
    if diff:
        best_cargo = max(budget_pressure, key=budget_pressure.get)
        budget_allocation[best_cargo] = round(budget_allocation[best_cargo] + diff, 2)

    units = {
        cargo_type: round(budget_allocation[cargo_type] / costs[cargo_type], 2)
        for cargo_type in CARGO_TYPES
    }
    return budget_allocation, units, costs


def allocate(scores: dict[str, float], total_units: int = TOTAL_UNITS) -> dict[str, float]:
    _, units, _ = allocate_budget(scores, budget=float(total_units))
    return units


def cargo_score_weights() -> dict[str, float]:
    return {
        "water_supplies": WEIGHT_WATER_SUPPLIES,
        "water_infrastructure": WEIGHT_WATER_INFRASTRUCTURE,
        "food_supplies": WEIGHT_FOOD_SUPPLIES,
        "fuel": WEIGHT_FUEL,
    }


def reasoning_formulas() -> dict[str, str]:
    return {
        "drought_score": (
            "sigmoid(DROUGHT_TEMPERATURE_WEIGHT * temperature_slope "
            "- DROUGHT_RAINFALL_WEIGHT * rainfall_slope "
            "- DROUGHT_HUMIDITY_WEIGHT * humidity_slope)"
        ),
        "water_supplies": (
            "WEIGHT_WATER_SUPPLIES * sigmoid("
            "SLOPE_WEIGHT * water_slope + LEVEL_WEIGHT * water_level "
            "+ DROUGHT_WEIGHT * drought_score + UNCERTAINTY_WEIGHT * water_uncertainty)"
        ),
        "water_infrastructure": (
            "WEIGHT_WATER_INFRASTRUCTURE * sigmoid("
            "LEVEL_WEIGHT * water_level + DROUGHT_WEIGHT * drought_score "
            "+ UNCERTAINTY_WEIGHT * water_uncertainty)"
        ),
        "food_supplies": (
            "WEIGHT_FOOD_SUPPLIES * sigmoid("
            "SLOPE_WEIGHT * food_slope + LEVEL_WEIGHT * food_level "
            "+ FOOD_DROUGHT_WEIGHT * drought_score + UNCERTAINTY_WEIGHT * food_uncertainty)"
        ),
        "fuel": (
            "WEIGHT_FUEL * sigmoid("
            "SLOPE_WEIGHT * fuel_slope + LEVEL_WEIGHT * fuel_level "
            "+ FUEL_CROSS_SECTOR_WEIGHT * mean(water_pressure, structural_water_pressure, food_pressure) "
            "+ UNCERTAINTY_WEIGHT * fuel_uncertainty)"
        ),
        "budget_allocation": (
            "softmax(cargo_score / SOFTMAX_TEMPERATURE) * regional_budget; "
            "effective_unit_cost is used after budget allocation to estimate units"
        ),
        "effective_unit_cost": "base_unit_cost / accessibility",
    }


def _pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def _feature_reason_text(
    *,
    cargo_type: str,
    label: str,
    value: float,
    contribution: float,
    features: dict[str, float],
) -> str:
    cargo_label = cargo_type.replace("_", " ")
    supports = contribution >= 0
    effect = "raises" if supports else "reduces"
    trend = "rises" if value >= 0 else "falls"

    if label.endswith("price trend"):
        subject = label.removesuffix(" trend")
        return (
            f"The forecasted {subject} {trend} over the horizon "
            f"({_pct(value)} from first to last forecast month), which {effect} "
            f"the allocation to {cargo_label}."
        )

    if label.endswith("price level"):
        subject = label.removesuffix(" level")
        relative = "above" if value >= 0 else "below"
        return (
            f"The forecasted {subject} is {relative} its historical median "
            f"(level signal {value:+.2f}), which {effect} the allocation to {cargo_label}."
        )

    if label == "drought pressure":
        rainfall = features.get("rainfall_slope", 0.0)
        temperature = features.get("temperature_slope", 0.0)
        humidity = features.get("humidity_slope", 0.0)
        drought_signal = value - 0.5
        weather_effect = "adds" if drought_signal >= 0 else "eases"
        return (
            "The weather forecast points to "
            f"rainfall {_pct(rainfall)}, temperature {_pct(temperature)}, "
            f"and humidity {_pct(humidity)} across the horizon; the combined "
            f"weather signal {weather_effect} drought pressure "
            f"({drought_signal:+.2f} vs neutral), shaping the allocation to {cargo_label}."
        )

    if label.endswith("forecast uncertainty"):
        subject = label.removesuffix(" forecast uncertainty")
        return (
            f"The {subject} forecast band is {'wide' if value > 0 else 'narrow'} "
            f"(relative width {value:.2f}), so uncertainty {effect} the allocation "
            f"to {cargo_label}."
        )

    if label == "water and food pressure spillover":
        return (
            f"Water and food forecasts create combined logistics pressure "
            f"(signal {value:.2f}), which {effect} the fuel allocation."
        )

    return f"{label} {effect} the allocation to {cargo_label}."


def _component_reason(
    *,
    cargo_type: str,
    component: dict[str, float | str],
    rank_basis: float,
    features: dict[str, float],
) -> dict[str, Any]:
    contribution = float(component["contribution"])
    feature = str(component["feature"])
    label = str(component["label"])
    value = float(component["value"])
    weight = float(component["weight"])
    return {
        "type": "score_component",
        "feature": feature,
        "rank_basis": round(rank_basis, 6),
        "contribution": round(contribution, 6),
        "value": round(value, 6),
        "weight": round(weight, 6),
        "text": _feature_reason_text(
            cargo_type=cargo_type,
            label=label,
            value=value,
            contribution=contribution,
            features=features,
        ),
    }


def explain_allocations(
    *,
    features: dict[str, float],
    scores: dict[str, float],
    budget_allocation: dict[str, float],
    effective_costs: dict[str, float],
    accessibility: float,
    top_n: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    components = compute_score_components(features)
    reasoning: dict[str, list[dict[str, Any]]] = {}

    for cargo_type in CARGO_TYPES:
        cargo_components = components[cargo_type]
        reasons = [
            _component_reason(
                cargo_type=cargo_type,
                component=component,
                rank_basis=(
                    max(float(component["value"]) - 0.5, 0.0)
                    if component["feature"] == "drought_score"
                    else max(float(component["contribution"]), 0.0)
                ),
                features=features,
            )
            for component in cargo_components
        ]
        reasoning[cargo_type] = sorted(
            reasons,
            key=lambda reason: (-float(reason["rank_basis"]), str(reason["type"])),
        )[:top_n]
    return reasoning


def _population_for_region(region_name: str, populations: dict[str, int]) -> int | None:
    return _region_lookup(region_name, populations)


def population_weighted_budget(
    regions: list[str],
    *,
    total_budget: float = TOTAL_BUDGET,
    populations: dict[str, int] | None = None,
) -> dict[str, float]:
    populations = populations or REGION_POPULATIONS
    region_populations = {
        region: _population_for_region(region, populations)
        for region in regions
    }
    missing = [region for region, population in region_populations.items() if population is None]
    if missing:
        raise ValueError(f"Missing population for regions: {missing}")

    total_population = sum(population for population in region_populations.values() if population is not None)
    if total_population <= 0:
        raise ValueError("Total population must be positive.")

    allocation = {
        region: round((population or 0) / total_population * total_budget, 2)
        for region, population in region_populations.items()
    }
    diff = round(total_budget - sum(allocation.values()), 2)
    if diff:
        largest_region = max(region_populations, key=lambda region: region_populations[region] or 0)
        allocation[largest_region] = round(allocation[largest_region] + diff, 2)
    return allocation


def population_weighted_units(
    regions: list[str],
    *,
    total_units: int = TOTAL_UNITS,
    populations: dict[str, int] | None = None,
) -> dict[str, int]:
    budgets = population_weighted_budget(
        regions,
        total_budget=float(total_units),
        populations=populations,
    )
    return {region: int(round(budget)) for region, budget in budgets.items()}


def decide(
    predictions: AgentPredictionBundle,
    *,
    budget: float = TOTAL_BUDGET,
    total_units: int | None = None,
    population: int | None = None,
    accessibility: float | None = None,
    unit_costs: dict[str, float] | None = None,
    reasoning: str = "reasons",
) -> AgentDecision:
    if reasoning not in {"reasons", "formula", "off"}:
        raise ValueError("reasoning must be 'reasons', 'formula', or 'off'.")
    if total_units is not None:
        budget = float(total_units)
    features = compute_features(predictions)
    scores = compute_scores(features)
    accessibility = accessibility if accessibility is not None else accessibility_for_region(predictions.region)
    budget_allocation, allocation, costs = allocate_budget(
        scores,
        budget=budget,
        unit_costs=unit_costs,
        accessibility=accessibility,
    )
    selected_cargo = max(scores, key=scores.get)
    explanations = None
    formulas = reasoning_formulas() if reasoning == "formula" else None
    if reasoning == "reasons":
        explanations = explain_allocations(
            features=features,
            scores=scores,
            budget_allocation=budget_allocation,
            effective_costs=costs,
            accessibility=accessibility,
        )

    return AgentDecision(
        region=predictions.region,
        allocation=allocation,
        budget_allocation=budget_allocation,
        scores={key: float(value) for key, value in scores.items()},
        selected_cargo=selected_cargo,
        regional_budget=budget,
        accessibility=accessibility,
        effective_unit_costs=costs,
        regional_units=round(sum(allocation.values()), 2),
        population=population,
        reasoning=explanations,
        formulas=formulas,
    )


def make_aid_decision(
    *,
    region: str,
    water_prediction: Any,
    food_prediction: Any,
    fuel_prediction: Any | None = None,
    weather_prediction: Any | None = None,
    budget: float | None = None,
    total_units: int | None = None,
    population: int | None = None,
    accessibility: float | None = None,
    unit_costs: dict[str, float] | None = None,
    reasoning: str = "reasons",
) -> AgentDecision:
    if budget is None:
        budget = float(total_units if total_units is not None else TOTAL_BUDGET)
    predictions = AgentPredictionBundle(
        region=region,
        water=water_prediction,
        food=food_prediction,
        fuel=fuel_prediction,
        weather=weather_prediction,
    )
    return decide(
        predictions,
        budget=budget,
        population=population,
        accessibility=accessibility,
        unit_costs=unit_costs,
        reasoning=reasoning,
    )

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
    reasoning: None = None


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
        return 0.5

    temperature = _weather_slope(weather, "temperature_avg_c")
    rainfall = _weather_slope(weather, "rainfall_mm_per_day")
    humidity = _weather_slope(weather, "relative_humidity_pct")
    return sigmoid(temperature - 1.2 * rainfall - 0.8 * humidity)


def compute_features(predictions: AgentPredictionBundle) -> dict[str, float]:
    water_forecast = _forecast_frame(predictions.water)
    water_history = _history_frame(predictions.water)
    food_forecast = _forecast_frame(predictions.food)
    food_history = _history_frame(predictions.food)
    fuel_forecast = _forecast_frame(predictions.fuel)
    fuel_history = _history_frame(predictions.fuel)

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
    }


def compute_scores(features: dict[str, float]) -> dict[str, float]:
    water_pressure = sigmoid(
        SLOPE_WEIGHT * features["water_slope"]
        + LEVEL_WEIGHT * features["water_level"]
        + DROUGHT_WEIGHT * features["drought_score"]
        + UNCERTAINTY_WEIGHT * features["water_uncertainty"]
    )
    structural_water_pressure = sigmoid(
        LEVEL_WEIGHT * features["water_level"]
        + DROUGHT_WEIGHT * features["drought_score"]
        + UNCERTAINTY_WEIGHT * features["water_uncertainty"]
    )
    food_pressure = sigmoid(
        SLOPE_WEIGHT * features["food_slope"]
        + LEVEL_WEIGHT * features["food_level"]
        + 0.5 * features["drought_score"]
        + UNCERTAINTY_WEIGHT * features["food_uncertainty"]
    )
    fuel_pressure = sigmoid(
        SLOPE_WEIGHT * features["fuel_slope"]
        + LEVEL_WEIGHT * features["fuel_level"]
        + 0.5 * np.mean([water_pressure, structural_water_pressure, food_pressure])
        + UNCERTAINTY_WEIGHT * features["fuel_uncertainty"]
    )

    return {
        "water_supplies": WEIGHT_WATER_SUPPLIES * water_pressure,
        "water_infrastructure": WEIGHT_WATER_INFRASTRUCTURE * structural_water_pressure,
        "food_supplies": WEIGHT_FOOD_SUPPLIES * food_pressure,
        "fuel": WEIGHT_FUEL * fuel_pressure,
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
    budget_pressure = {
        cargo_type: scores[cargo_type] * costs[cargo_type]
        for cargo_type in CARGO_TYPES
    }
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
) -> AgentDecision:
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
        reasoning=None,
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
    )

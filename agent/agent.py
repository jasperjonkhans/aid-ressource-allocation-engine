from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CARGO_TYPES = (
    "water_supplies",
    "water_infrastructure",
    "food_supplies",
    "fuel",
)

WEIGHT_WATER_SUPPLIES = 1.20
WEIGHT_WATER_INFRASTRUCTURE = 0.85
WEIGHT_FOOD_SUPPLIES = 1.15
WEIGHT_FUEL = 0.90

SLOPE_WEIGHT = 2.5
LEVEL_WEIGHT = 0.8
UNCERTAINTY_WEIGHT = 0.3
DROUGHT_WEIGHT = 0.7
SOFTMAX_TEMPERATURE = 1.5
TOTAL_UNITS = 100


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
    allocation: dict[str, int]
    scores: dict[str, float]
    selected_cargo: str
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


def allocate(scores: dict[str, float], total_units: int = TOTAL_UNITS) -> dict[str, int]:
    shares = softmax(scores)
    allocation = {cargo_type: int(round(shares.get(cargo_type, 0.0) * total_units)) for cargo_type in CARGO_TYPES}
    diff = total_units - sum(allocation.values())
    if diff:
        best_cargo = max(scores, key=scores.get)
        allocation[best_cargo] += diff
    return allocation


def decide(
    predictions: AgentPredictionBundle,
    *,
    total_units: int = TOTAL_UNITS,
) -> AgentDecision:
    features = compute_features(predictions)
    scores = compute_scores(features)
    allocation = allocate(scores, total_units=total_units)
    selected_cargo = max(scores, key=scores.get)

    return AgentDecision(
        region=predictions.region,
        allocation=allocation,
        scores={key: float(value) for key, value in scores.items()},
        selected_cargo=selected_cargo,
        reasoning=None,
    )


def make_aid_decision(
    *,
    region: str,
    water_prediction: Any,
    food_prediction: Any,
    fuel_prediction: Any | None = None,
    weather_prediction: Any | None = None,
    total_units: int = TOTAL_UNITS,
) -> AgentDecision:
    predictions = AgentPredictionBundle(
        region=region,
        water=water_prediction,
        food=food_prediction,
        fuel=fuel_prediction,
        weather=weather_prediction,
    )
    return decide(predictions, total_units=total_units)

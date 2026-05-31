from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project.agent import REGION_POPULATIONS, make_aid_decision  # noqa: E402
from project.agent.agent import AgentPredictionBundle, compute_features  # noqa: E402
from project.helper.predictions import (  # noqa: E402
    predict_gedo_weather,
    predict_global_fuel_price,
    predict_regional_food_price,
    predict_regional_water_price,
)


DEFAULT_REGION = "Buur Hakaba"
DEFAULT_BUDGET = 100.0
DEFAULT_RAINFALL_END_FACTOR = 0.005
DEFAULT_HUMIDITY_END_FACTOR = 0.35
DEFAULT_TEMPERATURE_INCREASE_C = 4.0
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass(frozen=True)
class ScenarioInputs:
    weather: Any
    water: Any
    food: Any
    fuel: Any


@dataclass(frozen=True)
class ScenarioDecision:
    label: str
    decision: Any
    features: dict[str, float]


@dataclass(frozen=True)
class RainfallShockResult:
    region: str
    rainfall_end_factor: float
    humidity_end_factor: float
    temperature_increase_c: float
    baseline: ScenarioDecision
    rainfall_shock: ScenarioDecision
    modified_rainfall: pd.DataFrame
    modified_humidity: pd.DataFrame
    modified_temperature: pd.DataFrame
    comparison: pd.DataFrame
    summary: dict[str, Any]


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def top_reason_for_cargo(decision: Any, cargo_type: str) -> str:
    if not decision.reasoning:
        return ""
    reasons = decision.reasoning.get(cargo_type) or []
    if not reasons:
        return ""
    return str(reasons[0]["text"])


def load_cached_prediction_inputs(region: str) -> ScenarioInputs:
    """Load cached weather and local cached-compatible market forecasts for one region."""
    return ScenarioInputs(
        weather=predict_gedo_weather(data_source="cache", forecast_source="cache"),
        water=predict_regional_water_price(region, source="local"),
        food=predict_regional_food_price(region, source="local"),
        fuel=predict_global_fuel_price(source="local"),
    )


def make_agent_decision_for_weather(
    *,
    label: str,
    region: str,
    budget: float,
    inputs: ScenarioInputs,
    weather: Any,
) -> ScenarioDecision:
    """Run the deterministic allocation agent against a specific weather forecast."""
    decision = make_aid_decision(
        region=region,
        water_prediction=inputs.water,
        food_prediction=inputs.food,
        fuel_prediction=inputs.fuel,
        weather_prediction=weather,
        budget=budget,
        population=REGION_POPULATIONS.get(region),
        reasoning="reasons",
    )
    features = compute_features(
        AgentPredictionBundle(region, inputs.water, inputs.food, inputs.fuel, weather)
    )
    return ScenarioDecision(label=label, decision=decision, features=features)


def build_declining_rainfall_forecast(
    rainfall: pd.DataFrame,
    *,
    end_factor: float,
) -> pd.DataFrame:
    """Create a copy where rainfall steadily declines to a fraction of month-one rainfall."""
    shocked = rainfall.sort_values("date").reset_index(drop=True).copy()
    start_value = max(float(shocked.loc[0, "forecast"]), 0.1)
    end_value = max(start_value * end_factor, 0.001)
    steps = max(len(shocked) - 1, 1)
    shocked["baseline_forecast"] = shocked["forecast"]
    shocked["forecast"] = [
        start_value + (end_value - start_value) * index / steps
        for index in range(len(shocked))
    ]

    for column in ("q05", "q10", "q25", "q50", "q75", "q90", "q95"):
        if column in shocked.columns:
            shocked[column] = shocked["forecast"]
    return shocked


def build_declining_humidity_forecast(
    humidity: pd.DataFrame,
    *,
    end_factor: float,
) -> pd.DataFrame:
    """Create a copy where relative humidity declines throughout the horizon."""
    shocked = humidity.sort_values("date").reset_index(drop=True).copy()
    start_value = max(float(shocked.loc[0, "forecast"]), 1.0)
    end_value = max(start_value * end_factor, 1.0)
    steps = max(len(shocked) - 1, 1)
    shocked["baseline_forecast"] = shocked["forecast"]
    shocked["forecast"] = [
        start_value + (end_value - start_value) * index / steps
        for index in range(len(shocked))
    ]

    for column in ("q05", "q10", "q25", "q50", "q75", "q90", "q95"):
        if column in shocked.columns:
            shocked[column] = shocked["forecast"]
    return shocked


def build_warming_temperature_forecast(
    temperature: pd.DataFrame,
    *,
    increase_c: float,
) -> pd.DataFrame:
    """Create a copy where average temperature rises by a fixed Celsius delta."""
    shocked = temperature.sort_values("date").reset_index(drop=True).copy()
    start_value = float(shocked.loc[0, "forecast"])
    end_value = start_value + increase_c
    steps = max(len(shocked) - 1, 1)
    shocked["baseline_forecast"] = shocked["forecast"]
    shocked["forecast"] = [
        start_value + (end_value - start_value) * index / steps
        for index in range(len(shocked))
    ]

    for column in ("q05", "q10", "q25", "q50", "q75", "q90", "q95"):
        if column in shocked.columns:
            shocked[column] = shocked["forecast"]
    return shocked


def replace_weather_stress_forecasts(
    weather: Any,
    *,
    modified_rainfall: pd.DataFrame,
    modified_humidity: pd.DataFrame,
    modified_temperature: pd.DataFrame,
) -> Any:
    """Return a WeatherPrediction copy with drought-stress weather forecasts replaced."""
    forecasts = {key: value.copy() for key, value in weather.forecasts.items()}
    forecasts["rainfall_mm_per_day"] = modified_rainfall.drop(
        columns=["baseline_forecast"],
        errors="ignore",
    )
    forecasts["relative_humidity_pct"] = modified_humidity.drop(
        columns=["baseline_forecast"],
        errors="ignore",
    )
    forecasts["temperature_avg_c"] = modified_temperature.drop(
        columns=["baseline_forecast"],
        errors="ignore",
    )
    return replace(weather, forecasts=forecasts)


def build_rainfall_shock_weather(
    weather: Any,
    *,
    end_factor: float,
    humidity_end_factor: float = DEFAULT_HUMIDITY_END_FACTOR,
    temperature_increase_c: float = DEFAULT_TEMPERATURE_INCREASE_C,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a compound drought weather shock plus comparison tables."""
    modified_rainfall = build_declining_rainfall_forecast(
        weather.forecasts["rainfall_mm_per_day"],
        end_factor=end_factor,
    )
    modified_humidity = build_declining_humidity_forecast(
        weather.forecasts["relative_humidity_pct"],
        end_factor=humidity_end_factor,
    )
    modified_temperature = build_warming_temperature_forecast(
        weather.forecasts["temperature_avg_c"],
        increase_c=temperature_increase_c,
    )
    shocked_weather = replace_weather_stress_forecasts(
        weather,
        modified_rainfall=modified_rainfall,
        modified_humidity=modified_humidity,
        modified_temperature=modified_temperature,
    )
    return shocked_weather, modified_rainfall, modified_humidity, modified_temperature


def decision_rows(scenario: ScenarioDecision) -> list[dict[str, Any]]:
    rows = []
    for cargo_type, amount in scenario.decision.budget_allocation.items():
        rows.append(
            {
                "scenario": scenario.label,
                "region": scenario.decision.region,
                "cargo_type": cargo_type,
                "budget_percent": round(float(amount), 4),
                "score": round(float(scenario.decision.scores[cargo_type]), 6),
                "top_reason": top_reason_for_cargo(scenario.decision, cargo_type),
                "drought_score": round(float(scenario.features["drought_score"]), 6),
            }
        )
    return rows


def build_allocation_comparison(
    baseline: ScenarioDecision,
    rainfall_shock: ScenarioDecision,
) -> pd.DataFrame:
    """Create a tidy comparison table for allocation, score, drought, and reasoning."""
    comparison = pd.DataFrame(
        decision_rows(baseline) + decision_rows(rainfall_shock)
    )
    comparison["budget_percent_delta_vs_baseline"] = (
        comparison.groupby("cargo_type")["budget_percent"].diff().fillna(0.0)
    )
    return comparison


def changed_allocations(
    baseline: ScenarioDecision,
    rainfall_shock: ScenarioDecision,
) -> list[str]:
    return [
        cargo_type
        for cargo_type in baseline.decision.budget_allocation
        if round(baseline.decision.budget_allocation[cargo_type], 2)
        != round(rainfall_shock.decision.budget_allocation[cargo_type], 2)
    ]


def changed_top_reasons(
    baseline: ScenarioDecision,
    rainfall_shock: ScenarioDecision,
) -> list[str]:
    return [
        cargo_type
        for cargo_type in baseline.decision.budget_allocation
        if top_reason_for_cargo(baseline.decision, cargo_type)
        != top_reason_for_cargo(rainfall_shock.decision, cargo_type)
    ]


def build_scenario_summary(
    *,
    region: str,
    rainfall_end_factor: float,
    humidity_end_factor: float,
    temperature_increase_c: float,
    baseline: ScenarioDecision,
    rainfall_shock: ScenarioDecision,
) -> dict[str, Any]:
    return {
        "region": region,
        "rainfall_end_factor": rainfall_end_factor,
        "humidity_end_factor": humidity_end_factor,
        "temperature_increase_c": temperature_increase_c,
        "baseline_drought_score": baseline.features["drought_score"],
        "rainfall_shock_drought_score": rainfall_shock.features["drought_score"],
        "baseline_allocation": baseline.decision.budget_allocation,
        "rainfall_shock_allocation": rainfall_shock.decision.budget_allocation,
        "baseline_top_reasons": {
            cargo_type: top_reason_for_cargo(baseline.decision, cargo_type)
            for cargo_type in baseline.decision.budget_allocation
        },
        "rainfall_shock_top_reasons": {
            cargo_type: top_reason_for_cargo(rainfall_shock.decision, cargo_type)
            for cargo_type in rainfall_shock.decision.budget_allocation
        },
        "changed_allocations": changed_allocations(baseline, rainfall_shock),
        "changed_top_reasons": changed_top_reasons(baseline, rainfall_shock),
    }


def run_rainfall_shock_scenario(
    *,
    region: str = DEFAULT_REGION,
    budget: float = DEFAULT_BUDGET,
    rainfall_end_factor: float = DEFAULT_RAINFALL_END_FACTOR,
    humidity_end_factor: float = DEFAULT_HUMIDITY_END_FACTOR,
    temperature_increase_c: float = DEFAULT_TEMPERATURE_INCREASE_C,
) -> RainfallShockResult:
    """Run baseline and rainfall-shock decisions without changing original data files."""
    inputs = load_cached_prediction_inputs(region)
    baseline = make_agent_decision_for_weather(
        label="baseline",
        region=region,
        budget=budget,
        inputs=inputs,
        weather=inputs.weather,
    )
    (
        shocked_weather,
        modified_rainfall,
        modified_humidity,
        modified_temperature,
    ) = build_rainfall_shock_weather(
        inputs.weather,
        end_factor=rainfall_end_factor,
        humidity_end_factor=humidity_end_factor,
        temperature_increase_c=temperature_increase_c,
    )
    rainfall_shock = make_agent_decision_for_weather(
        label="rainfall_shock",
        region=region,
        budget=budget,
        inputs=inputs,
        weather=shocked_weather,
    )
    comparison = build_allocation_comparison(baseline, rainfall_shock)
    summary = build_scenario_summary(
        region=region,
        rainfall_end_factor=rainfall_end_factor,
        humidity_end_factor=humidity_end_factor,
        temperature_increase_c=temperature_increase_c,
        baseline=baseline,
        rainfall_shock=rainfall_shock,
    )
    return RainfallShockResult(
        region=region,
        rainfall_end_factor=rainfall_end_factor,
        humidity_end_factor=humidity_end_factor,
        temperature_increase_c=temperature_increase_c,
        baseline=baseline,
        rainfall_shock=rainfall_shock,
        modified_rainfall=modified_rainfall,
        modified_humidity=modified_humidity,
        modified_temperature=modified_temperature,
        comparison=comparison,
        summary=summary,
    )


def write_scenario_outputs(result: RainfallShockResult, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(output_path / "allocation_comparison.csv", index=False)
    result.modified_rainfall.to_csv(output_path / "modified_rainfall_forecast.csv", index=False)
    result.modified_humidity.to_csv(output_path / "modified_humidity_forecast.csv", index=False)
    result.modified_temperature.to_csv(output_path / "modified_temperature_forecast.csv", index=False)
    (output_path / "scenario_summary.json").write_text(
        json.dumps(result.summary, indent=2),
        encoding="utf-8",
    )
    return output_path


def print_decision_block(scenario: ScenarioDecision) -> None:
    print(f"\n=== {scenario.label} ===")
    print(f"region: {scenario.decision.region}")
    print(f"drought_score: {scenario.features['drought_score']:.4f}")
    print("allocation:")
    for cargo_type, amount in scenario.decision.budget_allocation.items():
        score = scenario.decision.scores[cargo_type]
        print(f"  {cargo_type}: {format_percent(amount)} (score={score:.4f})")
    print("top reasons:")
    for cargo_type in scenario.decision.budget_allocation:
        print(f"  {cargo_type}: {top_reason_for_cargo(scenario.decision, cargo_type)}")


def print_scenario_result(result: RainfallShockResult, output_dir: Path | None = None) -> None:
    print_decision_block(result.baseline)
    print_decision_block(result.rainfall_shock)
    print("\n=== Scenario Change Summary ===")
    print(f"allocation changed for: {', '.join(result.summary['changed_allocations']) or 'none'}")
    print(f"top reasoning changed for: {', '.join(result.summary['changed_top_reasons']) or 'none'}")
    if output_dir is not None:
        print(f"wrote: {output_dir}")


def run_scenario_from_args(args: argparse.Namespace) -> None:
    result = run_rainfall_shock_scenario(
        region=args.region,
        budget=args.budget,
        rainfall_end_factor=args.rainfall_end_factor,
        humidity_end_factor=args.humidity_end_factor,
        temperature_increase_c=args.temperature_increase_c,
    )
    output_dir = write_scenario_outputs(result, args.output_dir)
    print_scenario_result(result, output_dir=output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a cached rainfall-shock scenario that modifies the rainfall weather "
            "forecast plus drought-related weather variables in memory and compares "
            "agent allocation/reasoning."
        )
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--rainfall-end-factor",
        type=float,
        default=DEFAULT_RAINFALL_END_FACTOR,
        help="Final shocked rainfall as a fraction of the first forecast value.",
    )
    parser.add_argument(
        "--humidity-end-factor",
        type=float,
        default=DEFAULT_HUMIDITY_END_FACTOR,
        help="Final shocked humidity as a fraction of the first forecast value.",
    )
    parser.add_argument(
        "--temperature-increase-c",
        type=float,
        default=DEFAULT_TEMPERATURE_INCREASE_C,
        help="Final shocked temperature increase in Celsius versus the first forecast value.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run_scenario_from_args(parse_args())

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project.agent import (
    REGION_POPULATIONS,
    TOTAL_BUDGET,
    make_aid_decision,
    population_weighted_budget,
)  # noqa: E402
from project.clients.sybilion import SybilionClient, monthly_timeseries  # noqa: E402
from project.config import (  # noqa: E402
    CONFIG_PATH,
    flatten_config_values,
    format_config_value,
    get_config_value,
    human_readable_agent_config,
    load_config,
    parse_config_value,
    set_config_value,
)
from project.domain.regions import water_region_districts  # noqa: E402
from project.helper.predictions import (  # noqa: E402
    DEFAULT_PRICE_HORIZON,
    DEFAULT_WEATHER_HORIZON,
    predict_gedo_weather,
    predict_global_fuel_price,
    predict_regional_food_price,
    predict_regional_water_price,
    predict_somalia_cmb,
    predict_somalia_water_prices,
    summarize_weather,
)

DEFAULT_AGENT_REGIONS = tuple(water_region_districts)


def status(message: str) -> None:
    print(f"[app] {message}", flush=True)


def format_usd(value: float) -> str:
    formatted = f"{value:,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"${formatted}"


def format_percent(value: float) -> str:
    formatted = f"{value:.2f}".replace(".", ",")
    return f"{formatted}%"


def is_percent_budget(total_budget: float) -> bool:
    return abs(total_budget - 100.0) < 0.005


def format_budget_value(value: float, *, percent_mode: bool) -> str:
    return format_percent(value) if percent_mode else format_usd(value)


def display_region_name(region_name: str) -> str:
    return region_name.replace("_", " ").replace("-", " ").title()


def _normalized_region_key(region_name: str) -> str:
    return region_name.lower().replace("_", " ").replace("-", " ").strip()


def normalize_agent_regions(args: argparse.Namespace) -> list[str]:
    selected = args.agent_regions or ([args.agent_region] if args.agent_region else DEFAULT_AGENT_REGIONS)
    return [display_region_name(region_name) for region_name in selected]


def print_series_summary(name: str, history: pd.DataFrame, forecast: pd.DataFrame) -> None:
    last_observed = history.sort_values("date").iloc[-1]
    first_forecast = forecast.sort_values("date").iloc[0]
    forecast_mean = float(pd.to_numeric(forecast["forecast"], errors="coerce").mean())
    print(
        f"{name}: last={last_observed['date'].date()} "
        f"value={float(last_observed['value']):.4f}; "
        f"next={first_forecast['date'].date()} "
        f"forecast={float(first_forecast['forecast']):.4f}; "
        f"horizon_mean={forecast_mean:.4f}"
    )


def print_pipeline_metrics(weather_result, water_result, cmb_result, fuel_result) -> None:
    print("\n=== Kennwerte ===")
    for key, value in summarize_weather(weather_result.history).items():
        print(f"weather.{key}: {value:.4f}")

    for metric_name, forecast in weather_result.forecasts.items():
        history = monthly_timeseries(weather_result.history, "month", metric_name)
        print_series_summary(f"weather.{metric_name}", history, forecast)

    print_series_summary("water.national_price", water_result.history, water_result.forecast)
    print_series_summary("food.cmb_national_usd", cmb_result.history, cmb_result.forecast)
    print_series_summary("fuel.global_proxy", fuel_result.history, fuel_result.forecast)


def print_agent_decision(decision, *, percent_mode: bool = False) -> None:
    print(f"region: {decision.region}")
    print(f"regional_budget: {format_budget_value(decision.regional_budget, percent_mode=percent_mode)}")
    print("budget_allocation:")
    for cargo_type, amount in decision.budget_allocation.items():
        print(f"  {cargo_type}: {format_budget_value(amount, percent_mode=percent_mode)}")
    print("scores:")
    for cargo_type, score in decision.scores.items():
        print(f"  {cargo_type}: {score:.4f}")


def _top_reason_for(decision, cargo_type: str) -> str | None:
    if not decision.reasoning:
        return None
    reasons = decision.reasoning.get(cargo_type) or []
    if not reasons:
        return None
    return str(reasons[0]["text"])


def _cargo_totals(decisions: dict[str, object]) -> dict[str, float]:
    cargo_types = next(iter(decisions.values())).budget_allocation.keys()
    return {
        cargo_type: sum(decision.budget_allocation[cargo_type] for decision in decisions.values())
        for cargo_type in cargo_types
    }


def print_overall_reasoning(decisions: dict[str, object], *, percent_mode: bool) -> None:
    if not decisions:
        return

    cargo_totals = _cargo_totals(decisions)
    top_cargo = max(cargo_totals, key=cargo_totals.get)
    top_cargo_region = max(decisions.values(), key=lambda decision: decision.budget_allocation[top_cargo])
    top_region = max(decisions.values(), key=lambda decision: decision.regional_budget)
    top_region_cargo = max(top_region.budget_allocation, key=top_region.budget_allocation.get)

    print("\n=== Overall Reasoning ===")
    print(
        "1. Most resources go to "
        f"{top_cargo}: {format_budget_value(cargo_totals[top_cargo], percent_mode=percent_mode)} "
        "overall."
    )
    top_cargo_reason = _top_reason_for(top_cargo_region, top_cargo)
    if top_cargo_reason:
        print(f"   Strongest concrete driver: {top_cargo_region.region}: {top_cargo_reason}")
    print(
        "2. The largest regional share goes to "
        f"{top_region.region}: {format_budget_value(top_region.regional_budget, percent_mode=percent_mode)}."
    )
    if top_region.population is not None:
        print(f"   This is driven by the population-weighted regional split: population={top_region.population}.")
    print(
        "3. Inside that region, the largest goods allocation is "
        f"{top_region_cargo}: "
        f"{format_budget_value(top_region.budget_allocation[top_region_cargo], percent_mode=percent_mode)}."
    )
    top_region_reason = _top_reason_for(top_region, top_region_cargo)
    if top_region_reason:
        print(f"   Strongest concrete driver: {top_region_reason}")


def print_formulas_once(decisions: dict[str, object]) -> None:
    formulas = None
    for decision in decisions.values():
        formulas = decision.formulas
        if formulas:
            break
    if not formulas:
        return
    print("\n=== Formulas ===")
    for name, formula in formulas.items():
        print(f"{name}: {formula}")


def print_agent_decisions(decisions: dict[str, object], *, reasoning_mode: str = "reasons") -> None:
    print("\n=== Agent Decisions ===")
    total_budget = sum(decision.regional_budget for decision in decisions.values())
    percent_mode = is_percent_budget(total_budget)
    for decision in decisions.values():
        print("")
        print_agent_decision(decision, percent_mode=percent_mode)
    if reasoning_mode == "reasons":
        print_overall_reasoning(decisions, percent_mode=percent_mode)
    elif reasoning_mode == "formula":
        print_formulas_once(decisions)


def weather_for_agent_region(region_name: str, weather_result):
    return weather_result if _normalized_region_key(region_name) == "gedo" else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Somalia forecasts for weather, water prices, and food/CMB prices. "
            "By default, the run command fetches fresh weather data and submits live Sybilion forecasts. "
            "Use config-show to print agent constants or config-set <key> <value> to update one."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["run", "config-show", "config-set"],
        default="run",
        help="Command to execute. Defaults to run for backwards-compatible pipeline usage.",
    )
    parser.add_argument(
        "config_key",
        nargs="?",
        help="Dotted config key, e.g. agent.total_budget.",
    )
    parser.add_argument(
        "config_value",
        nargs="?",
        help='New JSON value for config-set, e.g. 12000000 or "[\\"Gedo\\", \\"Buur Hakaba\\"]".',
    )
    parser.add_argument(
        "--mode",
        choices=["live", "cached"],
        default="live",
        help="Run live by default, or use cached data and cached/local forecasts.",
    )
    parser.add_argument("--weather-horizon", type=int, default=DEFAULT_WEATHER_HORIZON)
    parser.add_argument("--price-horizon", type=int, default=DEFAULT_PRICE_HORIZON)
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--water-aggregation", choices=["mean", "median"], default="mean")
    parser.add_argument("--fuel-aggregation", choices=["mean", "median"], default="median")
    parser.add_argument(
        "--agent-region",
        help="Single region for the agent. Kept for quick one-region runs.",
    )
    parser.add_argument(
        "--agent-regions",
        nargs="+",
        help='Regions for the agent. Defaults to all configured water regions: "Buur Hakaba" Bakool Gedo.',
    )
    parser.add_argument(
        "--agent-units",
        type=float,
        default=TOTAL_BUDGET,
        help="Deprecated alias for --agent-budget.",
    )
    parser.add_argument(
        "--agent-budget",
        type=float,
        help="Total money budget for all agent regions. Budget is first weighted by population.",
    )
    parser.add_argument(
        "--agent-forecast-source",
        choices=["live", "cache", "local"],
        default="live",
        help="Forecast source for regional agent water/food inputs.",
    )
    parser.add_argument(
        "--fuel-forecast-source",
        choices=["live", "cache", "local"],
        default="live",
        help="Fuel forecast source.",
    )
    parser.add_argument(
        "--reasoning",
        choices=["reasons", "formula", "off"],
        default="reasons",
        help="Print deterministic allocation reasons, formulas, or no reasoning.",
    )
    parser.add_argument("--top-drivers", type=int, default=8)
    return parser.parse_args()


def show_config(args: argparse.Namespace) -> None:
    config = load_config()
    if args.config_key:
        try:
            value = get_config_value(config, args.config_key)
        except KeyError as exc:
            raise SystemExit(f"Unknown config key: {exc.args[0]}") from exc
        if args.config_key == "agent":
            print(human_readable_agent_config(config))
            return
        if isinstance(value, dict):
            rows = flatten_config_values(value, args.config_key)
            key_width = max(len(key) for key, _ in rows)
            for key, row_value in rows:
                print(f"{key:<{key_width}}  {format_config_value(row_value)}")
            return
        print(f"{args.config_key}  {format_config_value(value)}")
        return
    print(human_readable_agent_config(config))


def set_config(args: argparse.Namespace) -> None:
    if not args.config_key or args.config_value is None:
        raise SystemExit("config-set requires <key> and <value>.")

    value = parse_config_value(args.config_value)
    try:
        old_value, new_value = set_config_value(args.config_key, value)
    except KeyError as exc:
        raise SystemExit(f"Unknown config key: {exc.args[0]}") from exc
    except PermissionError as exc:
        raise SystemExit(str(exc)) from exc
    except TypeError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Updated {args.config_key} in {CONFIG_PATH}:")
    print(f"  old: {old_value!r}")
    print(f"  new: {new_value!r}")


def run_pipeline(args: argparse.Namespace):
    live_mode = args.mode == "live"
    data_source = "live" if live_mode else "cache"
    forecast_source = "live" if live_mode else "cache"
    agent_forecast_source = args.agent_forecast_source
    if not live_mode and agent_forecast_source == "live":
        agent_forecast_source = "local"
    fuel_forecast_source = args.fuel_forecast_source
    if not live_mode and fuel_forecast_source == "live":
        fuel_forecast_source = "local"
    agent_budget = args.agent_budget if args.agent_budget is not None else args.agent_units
    status(f"mode={args.mode}, data_source={data_source}, forecast_source={forecast_source}")
    needs_live_client = "live" in {forecast_source, agent_forecast_source, fuel_forecast_source}
    client = SybilionClient() if needs_live_client else None

    weather_result = predict_gedo_weather(
        data_source=data_source,
        forecast_source=forecast_source,
        client=client,
        horizon=args.weather_horizon,
        poll_s=args.poll_s,
        timeout_s=args.timeout_s,
        overwrite_weather=live_mode,
    )
    status(
        "Weather loaded: "
        f"rows={len(weather_result.history)}, "
        f"path={weather_result.weather_path}, "
        f"range={weather_result.used_range[0]}-{weather_result.used_range[1]}"
    )

    water_result = predict_somalia_water_prices(
        source=forecast_source,
        client=client,
        horizon=args.price_horizon,
        poll_s=args.poll_s,
        timeout_s=args.timeout_s,
        aggregation=args.water_aggregation,
        top_drivers=args.top_drivers,
    )

    cmb_result = predict_somalia_cmb(
        source=forecast_source,
        client=client,
        horizon=args.price_horizon,
        poll_s=args.poll_s,
        timeout_s=args.timeout_s,
        top_drivers=args.top_drivers,
    )

    fuel_result = predict_global_fuel_price(
        source=fuel_forecast_source,
        client=client if fuel_forecast_source == "live" else None,
        horizon=args.price_horizon,
        poll_s=args.poll_s,
        timeout_s=args.timeout_s,
        aggregation=args.fuel_aggregation,
        top_drivers=args.top_drivers,
    )

    print_pipeline_metrics(weather_result, water_result, cmb_result, fuel_result)

    agent_decisions = {}
    agent_regions = normalize_agent_regions(args)
    status(
        "Building agent decision from regional predictions: "
        f"regions={', '.join(agent_regions)}, source={agent_forecast_source}"
    )
    region_budgets = population_weighted_budget(agent_regions, total_budget=agent_budget)
    for agent_region in agent_regions:
        regional_water_result = predict_regional_water_price(
            agent_region,
            source=agent_forecast_source,
            client=client if agent_forecast_source == "live" else None,
            horizon=args.price_horizon,
            poll_s=args.poll_s,
            timeout_s=args.timeout_s,
            top_drivers=args.top_drivers,
        )
        regional_food_result = predict_regional_food_price(
            agent_region,
            source=agent_forecast_source,
            client=client if agent_forecast_source == "live" else None,
            horizon=args.price_horizon,
            poll_s=args.poll_s,
            timeout_s=args.timeout_s,
            top_drivers=args.top_drivers,
        )
        agent_decisions[agent_region] = make_aid_decision(
            region=agent_region,
            water_prediction=regional_water_result,
            food_prediction=regional_food_result,
            fuel_prediction=fuel_result,
            weather_prediction=weather_for_agent_region(agent_region, weather_result),
            budget=region_budgets[agent_region],
            population=REGION_POPULATIONS[agent_region],
            reasoning=args.reasoning,
        )
    print_agent_decisions(agent_decisions, reasoning_mode=args.reasoning)

    status("Pipeline finished.")
    return weather_result, water_result, cmb_result, fuel_result, agent_decisions


def main() -> None:
    args = parse_args()
    if args.command == "config-show":
        show_config(args)
        return
    if args.command == "config-set":
        set_config(args)
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project.agent import make_aid_decision  # noqa: E402
from project.clients.sybilion import SybilionClient, monthly_timeseries  # noqa: E402
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


def status(message: str) -> None:
    print(f"[app] {message}", flush=True)


def display_region_name(region_name: str) -> str:
    return region_name.replace("_", " ").replace("-", " ").title()


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


def print_agent_decision(decision) -> None:
    print("\n=== Agent Decision ===")
    print(f"region: {decision.region}")
    print(f"selected_cargo: {decision.selected_cargo}")
    print("allocation:")
    for cargo_type, units in decision.allocation.items():
        print(f"  {cargo_type}: {units}")
    print("scores:")
    for cargo_type, score in decision.scores.items():
        print(f"  {cargo_type}: {score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Somalia forecasts for weather, water prices, and food/CMB prices. "
            "--mode cache uses existing data and cached Sybilion forecasts; "
            "--mode refresh fetches Copernicus weather and submits live Sybilion jobs."
        )
    )
    parser.add_argument("--mode", choices=["cache", "refresh"], default="cache")
    parser.add_argument("--data-source", choices=["cache", "live"], help="Override weather data source.")
    parser.add_argument("--forecast-source", choices=["cache", "live"], help="Override Sybilion forecast source.")
    parser.add_argument("--overwrite-weather", action="store_true", help="Redownload Copernicus NetCDF even if cached.")
    parser.add_argument("--weather-horizon", type=int, default=DEFAULT_WEATHER_HORIZON)
    parser.add_argument("--price-horizon", type=int, default=DEFAULT_PRICE_HORIZON)
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--water-aggregation", choices=["mean", "median"], default="mean")
    parser.add_argument("--fuel-aggregation", choices=["mean", "median"], default="median")
    parser.add_argument("--agent-region", default="Gedo")
    parser.add_argument("--agent-units", type=int, default=100)
    parser.add_argument(
        "--agent-forecast-source",
        choices=["cache", "live", "local"],
        help="Override forecast source for regional agent water/food inputs.",
    )
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument(
        "--fuel-forecast-source",
        choices=["cache", "live", "local"],
        default="local",
        help="Fuel forecast source. Defaults to a local seasonal baseline until a Sybilion fuel cache exists.",
    )
    parser.add_argument("--top-drivers", type=int, default=8)
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace):
    data_source = args.data_source or ("live" if args.mode == "refresh" else "cache")
    forecast_source = args.forecast_source or ("live" if args.mode == "refresh" else "cache")
    agent_forecast_source = args.agent_forecast_source or forecast_source
    status(f"mode={args.mode}, data_source={data_source}, forecast_source={forecast_source}")

    needs_live_client = "live" in {
        forecast_source,
        args.fuel_forecast_source,
        agent_forecast_source,
    }
    client = SybilionClient() if needs_live_client else None

    weather_result = predict_gedo_weather(
        data_source=data_source,
        forecast_source=forecast_source,
        client=client,
        horizon=args.weather_horizon,
        poll_s=args.poll_s,
        timeout_s=args.timeout_s,
        overwrite_weather=args.overwrite_weather,
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
        source=args.fuel_forecast_source,
        client=client if args.fuel_forecast_source == "live" else None,
        horizon=args.price_horizon,
        poll_s=args.poll_s,
        timeout_s=args.timeout_s,
        aggregation=args.fuel_aggregation,
        top_drivers=args.top_drivers,
    )

    print_pipeline_metrics(weather_result, water_result, cmb_result, fuel_result)

    agent_decision = None
    if not args.skip_agent:
        agent_region = display_region_name(args.agent_region)
        status(
            "Building agent decision from regional predictions: "
            f"region={agent_region}, source={agent_forecast_source}"
        )
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
        agent_decision = make_aid_decision(
            region=agent_region,
            water_prediction=regional_water_result,
            food_prediction=regional_food_result,
            fuel_prediction=fuel_result,
            weather_prediction=weather_result,
            total_units=args.agent_units,
        )
        print_agent_decision(agent_decision)

    status("Pipeline finished.")
    return weather_result, water_result, cmb_result, fuel_result, agent_decision


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()

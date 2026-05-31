from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project.clients.sybilion import (
    SybilionClient,
    build_forecast_body,
    forecast_to_frame,
    monthly_timeseries,
)
from project.config import CONFIG, project_path
from project.domain.regions import GEDO_POLYGON, water_region_districts

_PREDICTIONS_CONFIG = CONFIG["predictions"]

WEATHER_DIR = project_path(_PREDICTIONS_CONFIG["weather_dir"])
WEATHER_OUT_DIR = project_path(_PREDICTIONS_CONFIG["weather_out_dir"])

WATER_RAW_PATH = project_path(_PREDICTIONS_CONFIG["water_raw_path"])
WATER_OUT_DIR = project_path(_PREDICTIONS_CONFIG["water_out_dir"])
REGIONAL_WATER_OUT_DIR = project_path(_PREDICTIONS_CONFIG["regional_water_out_dir"])

CMB_RAW_PATH = project_path(_PREDICTIONS_CONFIG["cmb_raw_path"])
CMB_OUT_DIR = project_path(_PREDICTIONS_CONFIG["cmb_out_dir"])

FUEL_RAW_PATH = project_path(_PREDICTIONS_CONFIG["fuel_raw_path"])
FUEL_OUT_DIR = project_path(_PREDICTIONS_CONFIG["fuel_out_dir"])
REGIONAL_FOOD_RAW_PATH = project_path(_PREDICTIONS_CONFIG["regional_food_raw_path"])
REGIONAL_FOOD_OUT_DIR = project_path(_PREDICTIONS_CONFIG["regional_food_out_dir"])

DEFAULT_WEATHER_HORIZON = int(_PREDICTIONS_CONFIG["default_weather_horizon"])
DEFAULT_PRICE_HORIZON = int(_PREDICTIONS_CONFIG["default_price_horizon"])

WEATHER_METRICS = dict(_PREDICTIONS_CONFIG["weather_metrics"])
WATER_KEYWORDS = list(_PREDICTIONS_CONFIG["water_keywords"])
CMB_KEYWORDS = list(_PREDICTIONS_CONFIG["cmb_keywords"])
FUEL_KEYWORDS = list(_PREDICTIONS_CONFIG["fuel_keywords"])
REGIONAL_FOOD_KEYWORDS = list(_PREDICTIONS_CONFIG["regional_food_keywords"])
REGIONAL_FOOD_PROXY_REGIONS = {
    "buur_hakaba": "Bay",
}


@dataclass(frozen=True)
class PricePrediction:
    history: pd.DataFrame
    forecast: pd.DataFrame
    forecast_payload: dict[str, Any]
    out_dir: Path
    job_id: str | None = None


@dataclass(frozen=True)
class WeatherPrediction:
    history: pd.DataFrame
    forecasts: dict[str, pd.DataFrame]
    out_dir: Path
    weather_path: Path
    used_range: tuple[int, int]
    forecast_payloads: dict[str, dict[str, Any]]
    job_ids: dict[str, str | None]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status(message: str) -> None:
    print(f"[predictions] {message}", flush=True)


def _validate_aggregation(aggregation: str) -> None:
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be either 'mean' or 'median'.")


def _aggregate_monthly(
    frame: pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
    aggregation: str,
) -> pd.DataFrame:
    _validate_aggregation(aggregation)
    clean = frame.copy()
    clean[date_column] = pd.to_datetime(clean[date_column], errors="coerce")
    clean[value_column] = pd.to_numeric(clean[value_column], errors="coerce")
    clean = clean.dropna(subset=[date_column, value_column])
    clean[date_column] = clean[date_column].dt.to_period("M").dt.to_timestamp()

    observed = getattr(clean.groupby(date_column, as_index=False)[value_column], aggregation)()
    return (
        observed.rename(columns={date_column: "date", value_column: "value"})
        .dropna(subset=["value"])
        .sort_values("date")
    )


def _complete_monthly_series(
    observed: pd.DataFrame,
    *,
    interpolation: bool = True,
    fill_edges: bool = True,
    extra_columns: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, int, int]:
    if observed.empty:
        raise RuntimeError("Cannot complete an empty monthly series.")

    full_dates = pd.date_range(observed["date"].min(), observed["date"].max(), freq="MS")
    ts = observed.set_index("date").reindex(full_dates)
    ts.index.name = "date"
    missing_before = int(ts["value"].isna().sum())
    if interpolation:
        ts["value"] = ts["value"].interpolate(limit_direction="both")
    if fill_edges:
        ts["value"] = ts["value"].ffill().bfill()
    missing_after = int(ts["value"].isna().sum())
    ts = ts.reset_index()
    ts["source"] = "observed"
    for column, value in (extra_columns or {}).items():
        ts[column] = value
    return ts, missing_before, missing_after


def _extend_to_current_month(
    ts: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    aggregation: str,
    label: str,
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame:
    current_month = pd.Timestamp.today().to_period("M").to_timestamp()
    last_observed_month = ts["date"].max()
    if last_observed_month >= current_month:
        return ts

    monthly_profile = (
        observed.assign(month_num=observed["date"].dt.month)
        .groupby("month_num")["value"]
        .agg(aggregation)
    )
    extension_dates = pd.date_range(
        last_observed_month + pd.offsets.MonthBegin(1),
        current_month,
        freq="MS",
    )
    extension = pd.DataFrame(
        {
            "date": extension_dates,
            "value": [
                float(monthly_profile.get(date.month, ts["value"].iloc[-1]))
                for date in extension_dates
            ],
            "source": "seasonal_extension",
        }
    )
    for column, value in (extra_columns or {}).items():
        extension[column] = value

    _status(
        f"Extended {label} with seasonal monthly {aggregation}: "
        f"{extension_dates.min().date()} to {extension_dates.max().date()}."
    )
    return pd.concat([ts, extension], ignore_index=True)


def normalize_polygon_local(polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points = [(float(lon), float(lat)) for lon, lat in polygon]
    if points[0] != points[-1]:
        points.append(points[0])
    return points


def polygon_bbox_local(polygon: list[tuple[float, float]], padding: float = 0.0) -> list[float]:
    points = normalize_polygon_local(polygon)
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    return [
        max(lats) + padding,
        min(lons) - padding,
        min(lats) - padding,
        max(lons) + padding,
    ]


def summarize_weather(history: pd.DataFrame) -> dict[str, float]:
    return {
        "rainfall_mm_per_day_mean": float(history["rainfall_mm_per_day"].mean()),
        "temperature_avg_c_mean": float(history["temperature_avg_c"].mean()),
        "relative_humidity_pct_mean": float(history["relative_humidity_pct"].mean()),
    }


def weather_ranges(current_year: int | None = None) -> list[tuple[int, int]]:
    current_year = current_year or pd.Timestamp.today().year
    target_end_year = current_year - 1
    return [
        (current_year - 20, target_end_year),
        (current_year - 15, target_end_year),
        (current_year - 10, target_end_year),
        (current_year - 5, target_end_year),
    ]


def load_or_fetch_gedo_weather(
    *,
    source: str = "cache",
    overwrite: bool = False,
    data_dir: Path = WEATHER_DIR,
    polygon: list[tuple[float, float]] = GEDO_POLYGON,
) -> tuple[pd.DataFrame, Path, tuple[int, int]]:
    data_dir.mkdir(parents=True, exist_ok=True)

    _status(f"Gedo polygon: {normalize_polygon_local(polygon)}")
    _status(f"CDS bbox [north, west, south, east]: {polygon_bbox_local(polygon, padding=0.1)}")

    errors: list[str] = []
    for start_year, end_year in weather_ranges():
        csv_path = data_dir / f"gedo_monthly_weather_{start_year}_{end_year}.csv"

        if source == "cache" and csv_path.exists() and not overwrite:
            history = pd.read_csv(csv_path, parse_dates=["month"])
            _status(f"Loaded cached weather data: {csv_path}")
            return history, csv_path, (start_year, end_year)

        if source == "cache":
            continue

        if source != "live":
            raise ValueError("source must be either 'cache' or 'live'.")

        try:
            _status(f"Requesting Gedo weather data from Copernicus: {start_year}-{end_year}")
            from project.clients.copernicus import fetch_monthly_weather_for_polygon

            history = fetch_monthly_weather_for_polygon(
                polygon=polygon,
                start_year=start_year,
                end_year=end_year,
                region_name="gedo",
                data_dir=data_dir,
                grid=0.1,
                overwrite=overwrite,
            )
            history.to_csv(csv_path, index=False)
            _status(f"Saved weather data: {csv_path}")
            return history, csv_path, (start_year, end_year)
        except Exception as exc:
            message = f"{start_year}-{end_year}: {type(exc).__name__}: {exc}"
            errors.append(message)
            lower_message = str(exc).lower()
            if any(word in lower_message for word in ("cost", "rate", "too large")):
                _status("CDS rejected this request size; trying a shorter range.")
                continue
            raise

    if source == "cache":
        raise FileNotFoundError(
            f"No cached weather CSV found in {data_dir}. Run with data_source='live'."
        )
    raise RuntimeError("All configured weather ranges failed: " + " | ".join(errors))


def build_weather_body(
    history: pd.DataFrame,
    metric_name: str,
    config: dict[str, Any],
    *,
    horizon: int = DEFAULT_WEATHER_HORIZON,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ts = monthly_timeseries(history, "month", metric_name)
    model_ts = ts.copy()
    if config.get("transform") == "log1p":
        model_ts["value"] = np.log1p(model_ts["value"].clip(lower=0))

    body = build_forecast_body(
        model_ts,
        title=config["title"],
        description=(
            f"Monthly ERA5-Land {config['label']} for the Gedo region in Somalia. "
            "The source region polygon is project.domain.regions.GEDO_POLYGON. "
            f"Model transform: {config.get('transform') or 'none'}."
        ),
        keywords=["Somalia", "Gedo", "weather", "ERA5-Land", metric_name],
        horizon=horizon,
        frequency="monthly",
        backtest=True,
        recency_factor=0.75,
        strictly_positive=config["strictly_positive"],
    )
    return ts, body


def load_somalia_water_prices(
    path: Path = WATER_RAW_PATH,
    *,
    aggregation: str = "mean",
    extend_to_current: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"month", "water_price"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Water CSV is missing columns: {sorted(missing)}")

    observed = _aggregate_monthly(
        df,
        date_column="month",
        value_column="water_price",
        aggregation=aggregation,
    )
    ts, missing_before, missing_after = _complete_monthly_series(
        observed,
        interpolation=True,
        fill_edges=False,
    )

    if extend_to_current:
        ts = _extend_to_current_month(
            ts,
            observed,
            aggregation=aggregation,
            label="stale water series",
        )

    _status(
        "Loaded Somalia water price proxy: "
        f"{ts['date'].min().date()} to {ts['date'].max().date()}, "
        f"{len(ts)} rows, aggregation={aggregation}, "
        f"missing {missing_before}->{missing_after}."
    )
    return ts


def _slugify_region_name(region_name: str) -> str:
    return (
        region_name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def regional_food_source_region(region_name: str) -> str:
    region_key = _slugify_region_name(region_name)
    return REGIONAL_FOOD_PROXY_REGIONS.get(region_key, region_name)


def load_regional_water_prices(
    region_name: str,
    districts: list[str],
    path: Path = WATER_RAW_PATH,
    *,
    extend_to_current: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"District", "month", "water_price"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Water CSV is missing columns: {sorted(missing)}")

    selected_districts = set(districts)
    region_df = df[df["District"].isin(selected_districts)].copy()
    if region_df.empty:
        raise RuntimeError(
            f"No water-price rows found for region {region_name!r} "
            f"with districts: {sorted(selected_districts)}"
        )

    found_districts = set(region_df["District"].dropna().unique())
    missing_districts = sorted(selected_districts - found_districts)
    if missing_districts:
        _status(
            f"Region {region_name} is missing water-price districts in source CSV: "
            f"{missing_districts}"
        )

    observed = _aggregate_monthly(
        region_df,
        date_column="month",
        value_column="water_price",
        aggregation="mean",
    )
    if observed.empty:
        raise RuntimeError(f"No numeric water-price rows found for region {region_name!r}.")

    ts, missing_before, missing_after = _complete_monthly_series(
        observed,
        interpolation=True,
        fill_edges=False,
    )

    if extend_to_current:
        ts = _extend_to_current_month(
            ts,
            observed,
            aggregation="mean",
            label=f"stale {region_name} water series",
        )

    _status(
        f"Loaded {region_name} regional water price proxy: "
        f"{ts['date'].min().date()} to {ts['date'].max().date()}, "
        f"{len(ts)} rows, districts={len(found_districts)}, "
        f"missing {missing_before}->{missing_after}."
    )
    return ts


def load_somalia_cmb_usd(path: Path = CMB_RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1])

    rows = []
    for idx in range(1, len(df.columns), 2):
        date_label = str(df.columns[idx][0])
        usd_idx = idx + 1
        if usd_idx >= len(df.columns):
            continue
        if str(df.columns[usd_idx][1]).strip().upper() != "USD":
            continue

        date = pd.to_datetime(date_label, errors="coerce")
        if pd.isna(date):
            continue

        values = pd.to_numeric(df.iloc[:, usd_idx], errors="coerce")
        rows.append({"date": date.to_period("M").to_timestamp(), "value": values.mean()})

    if not rows:
        raise RuntimeError(f"No monthly USD CMB columns found in {path}.")

    ts = pd.DataFrame(rows).dropna().sort_values("date")
    ts = ts.groupby("date", as_index=False)["value"].mean()

    full_dates = pd.date_range(ts["date"].min(), ts["date"].max(), freq="MS")
    ts = ts.set_index("date").reindex(full_dates)
    ts.index.name = "date"
    missing_before = int(ts["value"].isna().sum())
    ts["value"] = ts["value"].interpolate(limit_direction="both")
    missing_after = int(ts["value"].isna().sum())
    ts = ts.reset_index()
    ts["source"] = "observed"

    _status(
        "Loaded Somalia CMB USD series: "
        f"{ts['date'].min().date()} to {ts['date'].max().date()}, "
        f"{len(ts)} rows, missing {missing_before}->{missing_after}."
    )
    return ts


def load_global_fuel_prices(
    path: Path = FUEL_RAW_PATH,
    *,
    aggregation: str = "median",
    extend_to_current: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "commodity", "price", "usdprice"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"WFP price CSV is missing columns: {sorted(missing)}")

    fuel = df[df["commodity"].astype(str).str.contains("fuel", case=False, na=False)].copy()
    if fuel.empty:
        raise RuntimeError(f"No fuel commodity rows found in {path}.")

    observed = _aggregate_monthly(
        fuel,
        date_column="date",
        value_column="usdprice",
        aggregation=aggregation,
    )
    ts, missing_before, missing_after = _complete_monthly_series(
        observed,
        interpolation=True,
        fill_edges=True,
    )

    if extend_to_current:
        ts = _extend_to_current_month(
            ts,
            observed,
            aggregation=aggregation,
            label="fuel series",
        )

    _status(
        "Loaded global fuel price proxy from WFP Somalia fuel rows: "
        f"{ts['date'].min().date()} to {ts['date'].max().date()}, "
        f"{len(ts)} rows, aggregation={aggregation}, "
        f"missing {missing_before}->{missing_after}."
    )
    return ts


def load_regional_food_prices(
    region_name: str,
    path: Path = REGIONAL_FOOD_RAW_PATH,
    *,
    region_column: str = "admin1",
    aggregation: str = "median",
    extend_to_current: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", region_column, "category", "commodity", "usdprice"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"WFP food price CSV is missing columns: {sorted(missing)}")

    food = df[
        (df[region_column].astype(str) == region_name)
        & (df["category"].astype(str) != "non-food")
        & (df["commodity"].astype(str) != "Exchange rate")
    ].copy()
    if food.empty:
        raise RuntimeError(f"No WFP food-price rows found for {region_column}={region_name!r}.")

    observed = _aggregate_monthly(
        food,
        date_column="date",
        value_column="usdprice",
        aggregation=aggregation,
    )
    if observed.empty:
        raise RuntimeError(f"No numeric WFP USD food-price rows found for {region_name!r}.")

    ts, missing_before, missing_after = _complete_monthly_series(
        observed,
        interpolation=True,
        fill_edges=True,
        extra_columns={"region": region_name},
    )

    if extend_to_current:
        ts = _extend_to_current_month(
            ts,
            observed,
            aggregation=aggregation,
            label=f"{region_name} food series",
            extra_columns={"region": region_name},
        )

    _status(
        f"Loaded {region_name} regional food price proxy: "
        f"{ts['date'].min().date()} to {ts['date'].max().date()}, "
        f"{len(ts)} rows, aggregation={aggregation}, "
        f"missing {missing_before}->{missing_after}."
    )
    return ts


def list_wfp_food_regions(
    path: Path = REGIONAL_FOOD_RAW_PATH,
    *,
    region_column: str = "admin1",
) -> list[str]:
    df = pd.read_csv(path, usecols=[region_column, "category", "commodity", "usdprice"])
    food = df[
        (df["category"].astype(str) != "non-food")
        & (df["commodity"].astype(str) != "Exchange rate")
        & pd.to_numeric(df["usdprice"], errors="coerce").notna()
    ]
    return sorted(food[region_column].dropna().astype(str).unique())


def build_somalia_water_body(
    history: pd.DataFrame,
    *,
    horizon: int = DEFAULT_PRICE_HORIZON,
    aggregation: str = "mean",
) -> dict[str, Any]:
    return build_forecast_body(
        history,
        title="Somalia national water price proxy",
        description=(
            f"Monthly Somalia water price proxy computed as the national {aggregation} "
            "district water_price from the Somalia water price panel, expressed in the "
            "source dataset price units. The source CSV ends in 2022; later months may "
            "be filled using the historical seasonal monthly profile for API recency."
        ),
        keywords=WATER_KEYWORDS,
        horizon=horizon,
        frequency="monthly",
        backtest=True,
        recency_factor=0.75,
        strictly_positive=True,
    )


def build_regional_water_body(
    history: pd.DataFrame,
    *,
    region_name: str,
    districts: list[str],
    horizon: int = DEFAULT_PRICE_HORIZON,
) -> dict[str, Any]:
    display_region = region_name.replace("_", " ").title()
    return build_forecast_body(
        history,
        title=f"{display_region} regional water price proxy",
        description=(
            f"Monthly water price proxy for {display_region}, Somalia. "
            "The series is computed as the simple monthly average of district "
            f"water_price values for these districts: {', '.join(districts)}. "
            "The source CSV ends in 2022; later months may be filled using the "
            "historical seasonal monthly mean for API recency."
        ),
        keywords=(WATER_KEYWORDS + [display_region, "regional water prices"])[:20],
        horizon=horizon,
        frequency="monthly",
        backtest=True,
        recency_factor=0.75,
        strictly_positive=True,
    )


def build_somalia_cmb_body(
    history: pd.DataFrame,
    *,
    horizon: int = DEFAULT_PRICE_HORIZON,
) -> dict[str, Any]:
    body = build_forecast_body(
        history,
        title="Somalia national Cost of Minimum Basket USD",
        description=(
            "Monthly Somalia Cost of Minimum Basket, national proxy computed as "
            "the regional average of FSNAU Total Basket CMB with red sorghum as "
            "the main cereal, expressed in USD."
        ),
        keywords=CMB_KEYWORDS,
        horizon=horizon,
        frequency="monthly",
        backtest=True,
        recency_factor=0.75,
        strictly_positive=True,
    )
    body["filters"] = {"limit": 1000}
    return body


def build_global_fuel_body(
    history: pd.DataFrame,
    *,
    horizon: int = DEFAULT_PRICE_HORIZON,
    aggregation: str = "median",
) -> dict[str, Any]:
    keywords = [
        "Somalia fuel prices",
        "diesel prices Somalia",
        "petrol prices Somalia",
        "Somalia transport costs",
        "Somalia food market fuel costs",
        "diesel fuel imports Somalia",
        "petrol transport costs Somalia",
        "Somali shilling USD exchange rate",
        "Horn of Africa logistics",
        "Somalia inflation consumer prices",
        "Somalia port import volumes",
        "Somalia drought conditions",
    ]
    body = build_forecast_body(
        history,
        title="Somalia national fuel price proxy",
        description=(
            f"Monthly fuel price proxy computed as the {aggregation} of WFP Somalia "
            "fuel commodity rows, including diesel and super petrol. USD prices are "
            "used, and missing months are primitively filled by interpolation with "
            "forward/backward fill at the edges."
        ),
        keywords=keywords,
        horizon=horizon,
        frequency="monthly",
        backtest=True,
        recency_factor=0.75,
        strictly_positive=True,
    )
    body["filters"] = {"limit": 1000}
    return body


def build_regional_food_body(
    history: pd.DataFrame,
    *,
    region_name: str,
    source_region_name: str | None = None,
    horizon: int = DEFAULT_PRICE_HORIZON,
    aggregation: str = "median",
) -> dict[str, Any]:
    source_region_name = source_region_name or region_name
    source_note = (
        f" WFP rows are filtered with admin1={source_region_name} as a proxy."
        if source_region_name != region_name
        else ""
    )
    return build_forecast_body(
        history[["date", "value"]],
        title=f"{region_name} regional food price proxy",
        description=(
            f"Monthly regional food price proxy for {region_name}, Somalia. "
            f"The series is computed as the {aggregation} monthly WFP USD food price "
            "across food commodities and markets in the region. Non-food rows and "
            "exchange-rate rows are excluded. Missing months are primitively filled "
            "by interpolation with forward/backward fill at the edges."
            f"{source_note}"
        ),
        keywords=REGIONAL_FOOD_KEYWORDS + [region_name],
        horizon=horizon,
        frequency="monthly",
        backtest=True,
        recency_factor=0.75,
        strictly_positive=True,
    )


def _source_mask(history: pd.DataFrame, source: str) -> pd.Series:
    if "source" not in history:
        return pd.Series([source == "observed"] * len(history), index=history.index)
    return history["source"] == source


def plot_price_prediction(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    observed = history[_source_mask(history, "observed")]
    extension = history[~_source_mask(history, "observed")]

    ax.plot(observed["date"], observed["value"], label="Observed")
    if not extension.empty:
        ax.plot(
            extension["date"],
            extension["value"],
            linestyle="--",
            color="#666666",
            label="Seasonal extension",
        )

    ax.plot(forecast["date"], forecast["forecast"], marker="o", label="Sybilion forecast")

    if "q05" in forecast and "q95" in forecast and forecast["q05"].notna().any():
        ax.fill_between(
            forecast["date"],
            forecast["q05"].astype(float),
            forecast["q95"].astype(float),
            alpha=0.16,
            label="q05-q95",
        )

    if "q10" in forecast and "q90" in forecast and forecast["q10"].notna().any():
        ax.fill_between(
            forecast["date"],
            forecast["q10"].astype(float),
            forecast["q90"].astype(float),
            alpha=0.24,
            label="q10-q90",
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    _status(f"Saved prediction plot: {out_path}")


def _resolve_forecast(
    *,
    body: dict[str, Any],
    client: SybilionClient | None,
    source: str,
    out_dir: Path,
    forecast_json_path: Path,
    driver_prefix: str,
    poll_s: float,
    timeout_s: float,
    top_drivers: int,
) -> tuple[dict[str, Any], str | None]:
    if source in {"local", "naive"}:
        _status(f"Building local seasonal forecast: {driver_prefix}")
        forecast = local_seasonal_forecast_payload(body)
        save_json(forecast_json_path, forecast)
        return forecast, None

    if source == "cache":
        if not forecast_json_path.exists():
            raise FileNotFoundError(
                f"Missing cached Sybilion forecast: {forecast_json_path}. "
                "Run with source='live' first or source='local' for a local baseline."
            )
        _status(f"Loaded cached forecast: {forecast_json_path}")
        return load_json(forecast_json_path), None

    if source != "live":
        raise ValueError("source must be 'cache', 'live', or 'local'.")
    if client is None:
        client = SybilionClient()

    _status(f"Submitting live Sybilion forecast: {driver_prefix}")
    job, forecast = client.run_forecast(
        body,
        poll_s=poll_s,
        timeout_s=timeout_s,
        driver_out_dir=out_dir,
        driver_prefix=driver_prefix,
        top_drivers=top_drivers,
        print_drivers=True,
        plot_drivers=True,
    )
    save_json(out_dir / "forecast_job_status.json", job.payload or {})
    save_json(forecast_json_path, forecast)
    return forecast, job.job_id


def _finalize_price_prediction(
    *,
    history: pd.DataFrame,
    forecast_payload: dict[str, Any],
    out_dir: Path,
    csv_path: Path,
    plot_title: str,
    plot_ylabel: str,
    plot_path: Path,
    job_id: str | None,
) -> PricePrediction:
    forecast = forecast_to_frame(forecast_payload)
    forecast.to_csv(csv_path, index=False)
    plot_price_prediction(
        history,
        forecast,
        title=plot_title,
        ylabel=plot_ylabel,
        out_path=plot_path,
    )
    return PricePrediction(history, forecast, forecast_payload, out_dir, job_id)


def local_seasonal_forecast_payload(body: dict[str, Any]) -> dict[str, Any]:
    raw_series = body.get("timeseries", {})
    if not raw_series:
        raise RuntimeError("Cannot build a local forecast without body.timeseries.")

    history = pd.DataFrame(
        {
            "date": pd.to_datetime(list(raw_series.keys())),
            "value": pd.to_numeric(list(raw_series.values()), errors="coerce"),
        }
    ).dropna().sort_values("date")
    if history.empty:
        raise RuntimeError("Cannot build a local forecast from an empty timeseries.")

    horizon = int(body.get("soft_horizon", DEFAULT_PRICE_HORIZON))
    last_date = history["date"].max()
    forecast_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    monthly_profile = history.assign(month_num=history["date"].dt.month).groupby("month_num")[
        "value"
    ].mean()
    fallback_value = float(history["value"].iloc[-1])

    forecast_series = {}
    for date in forecast_dates:
        value = float(monthly_profile.get(date.month, fallback_value))
        low = max(0.0, value * 0.9) if body.get("strictly_positive") else value * 0.9
        high = value * 1.1
        forecast_series[date.strftime("%Y-%m-%d")] = {
            "forecast": value,
            "quantile_forecast": {
                "0.05": low,
                "0.10": low,
                "0.25": value * 0.95,
                "0.50": value,
                "0.75": value * 1.05,
                "0.90": high,
                "0.95": high,
            },
        }

    return {
        "data": {
            "forecast_series": forecast_series,
            "model": "local_seasonal_baseline",
        }
    }


def _resolve_weather_forecast(
    *,
    history: pd.DataFrame,
    metric_name: str,
    config: dict[str, Any],
    client: SybilionClient | None,
    source: str,
    out_dir: Path,
    horizon: int,
    poll_s: float,
    timeout_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast_json_path = out_dir / f"{metric_name}_forecast.json"
    forecast_csv_path = out_dir / f"{metric_name}_forecast.csv"
    body_path = out_dir / f"{metric_name}_request.json"
    job_path = out_dir / f"{metric_name}_job.json"

    ts, body = build_weather_body(history, metric_name, config, horizon=horizon)
    ts.to_csv(out_dir / f"{metric_name}_history.csv", index=False)
    save_json(body_path, body)

    if source == "cache":
        if not forecast_json_path.exists():
            raise FileNotFoundError(
                f"Missing cached weather forecast: {forecast_json_path}. "
                "Run with forecast_source='live' first."
            )
        forecast_payload = load_json(forecast_json_path)
        job_id = None
        _status(f"Loaded cached weather forecast for {metric_name}: {forecast_json_path}")
    elif source == "live":
        if client is None:
            client = SybilionClient()
        _status(f"Submitting live Sybilion weather forecast: {metric_name}")
        job, forecast_payload = client.run_forecast(
            body,
            poll_s=poll_s,
            timeout_s=timeout_s,
            verbose=True,
            print_drivers=False,
            plot_drivers=False,
        )
        job_id = job.job_id
        save_json(job_path, job.payload or {})
        save_json(forecast_json_path, forecast_payload)
    else:
        raise ValueError("source must be either 'cache' or 'live'.")

    forecast = forecast_to_frame(forecast_payload)
    if config.get("transform") == "log1p":
        value_columns = ["forecast", "q05", "q10", "q25", "q50", "q75", "q90", "q95"]
        existing = [column for column in value_columns if column in forecast.columns]
        forecast[existing] = np.expm1(forecast[existing].astype(float)).clip(lower=0)

    forecast.insert(0, "metric", metric_name)
    forecast.to_csv(forecast_csv_path, index=False)
    return ts, forecast, forecast_payload, job_id


def predict_gedo_weather(
    *,
    data_source: str = "cache",
    forecast_source: str = "cache",
    client: SybilionClient | None = None,
    horizon: int = DEFAULT_WEATHER_HORIZON,
    poll_s: float = 5.0,
    timeout_s: float = 900.0,
    overwrite_weather: bool = False,
    data_dir: Path = WEATHER_DIR,
    out_dir: Path = WEATHER_OUT_DIR,
) -> WeatherPrediction:
    history, weather_path, used_range = load_or_fetch_gedo_weather(
        source=data_source,
        overwrite=overwrite_weather,
        data_dir=data_dir,
    )

    forecasts: dict[str, pd.DataFrame] = {}
    forecast_payloads: dict[str, dict[str, Any]] = {}
    job_ids: dict[str, str | None] = {}

    for metric_name, config in WEATHER_METRICS.items():
        _, forecast, payload, job_id = _resolve_weather_forecast(
            history=history,
            metric_name=metric_name,
            config=config,
            client=client,
            source=forecast_source,
            out_dir=out_dir,
            horizon=horizon,
            poll_s=poll_s,
            timeout_s=timeout_s,
        )
        forecasts[metric_name] = forecast
        forecast_payloads[metric_name] = payload
        job_ids[metric_name] = job_id

    combined = pd.concat(forecasts.values(), ignore_index=True)
    combined.to_csv(out_dir / "gedo_weather_sybilion_forecasts.csv", index=False)
    return WeatherPrediction(
        history=history,
        forecasts=forecasts,
        out_dir=out_dir,
        weather_path=weather_path,
        used_range=used_range,
        forecast_payloads=forecast_payloads,
        job_ids=job_ids,
    )


def predict_somalia_water_prices(
    *,
    source: str = "cache",
    client: SybilionClient | None = None,
    horizon: int = DEFAULT_PRICE_HORIZON,
    poll_s: float = 5.0,
    timeout_s: float = 900.0,
    aggregation: str = "mean",
    extend_to_current: bool = True,
    out_dir: Path = WATER_OUT_DIR,
    top_drivers: int = 8,
) -> PricePrediction:
    out_dir.mkdir(parents=True, exist_ok=True)
    history = load_somalia_water_prices(
        WATER_RAW_PATH,
        aggregation=aggregation,
        extend_to_current=extend_to_current,
    )
    history.to_csv(out_dir / "somalia_water_price_monthly.csv", index=False)

    body = build_somalia_water_body(history, horizon=horizon, aggregation=aggregation)
    save_json(out_dir / "forecast_body.json", body)

    forecast_payload, job_id = _resolve_forecast(
        body=body,
        client=client,
        source=source,
        out_dir=out_dir,
        forecast_json_path=out_dir / "somalia_water_sybilion_forecast.json",
        driver_prefix="somalia_water_sybilion",
        poll_s=poll_s,
        timeout_s=timeout_s,
        top_drivers=top_drivers,
    )

    return _finalize_price_prediction(
        history=history,
        forecast_payload=forecast_payload,
        out_dir=out_dir,
        csv_path=out_dir / "somalia_water_sybilion_forecast.csv",
        plot_title="Sybilion Forecast - Somalia Water Prices",
        plot_ylabel="Water price",
        plot_path=out_dir / "somalia_water_sybilion_forecast.png",
        job_id=job_id,
    )


def predict_regional_water_price(
    region_name: str,
    districts: list[str] | None = None,
    *,
    source: str = "local",
    client: SybilionClient | None = None,
    horizon: int = DEFAULT_PRICE_HORIZON,
    poll_s: float = 5.0,
    timeout_s: float = 900.0,
    extend_to_current: bool = True,
    out_dir: Path | None = None,
    top_drivers: int = 8,
) -> PricePrediction:
    region_key = _slugify_region_name(region_name)
    if districts is None:
        try:
            districts = water_region_districts[region_key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown regional water area {region_name!r}. "
                f"Known regions: {sorted(water_region_districts)}"
            ) from exc

    out_dir = out_dir or REGIONAL_WATER_OUT_DIR / region_key
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load_regional_water_prices(
        region_key,
        districts,
        WATER_RAW_PATH,
        extend_to_current=extend_to_current,
    )
    history.to_csv(out_dir / f"{region_key}_water_price_monthly.csv", index=False)

    body = build_regional_water_body(
        history,
        region_name=region_key,
        districts=districts,
        horizon=horizon,
    )
    save_json(out_dir / "forecast_body.json", body)

    forecast_payload, job_id = _resolve_forecast(
        body=body,
        client=client,
        source=source,
        out_dir=out_dir,
        forecast_json_path=out_dir / f"{region_key}_water_sybilion_forecast.json",
        driver_prefix=f"{region_key}_water_sybilion",
        poll_s=poll_s,
        timeout_s=timeout_s,
        top_drivers=top_drivers,
    )

    return _finalize_price_prediction(
        history=history,
        forecast_payload=forecast_payload,
        out_dir=out_dir,
        csv_path=out_dir / f"{region_key}_water_sybilion_forecast.csv",
        plot_title=f"Forecast - {region_key.replace('_', ' ').title()} Water Prices",
        plot_ylabel="Water price",
        plot_path=out_dir / f"{region_key}_water_sybilion_forecast.png",
        job_id=job_id,
    )


def predict_regional_water_prices(
    *,
    regions: dict[str, list[str]] | None = None,
    source: str = "local",
    client: SybilionClient | None = None,
    horizon: int = DEFAULT_PRICE_HORIZON,
    poll_s: float = 5.0,
    timeout_s: float = 900.0,
    extend_to_current: bool = True,
    out_dir: Path = REGIONAL_WATER_OUT_DIR,
    top_drivers: int = 8,
) -> dict[str, PricePrediction]:
    regions = regions or water_region_districts
    results: dict[str, PricePrediction] = {}
    for region_name, districts in regions.items():
        region_key = _slugify_region_name(region_name)
        results[region_key] = predict_regional_water_price(
            region_key,
            districts,
            source=source,
            client=client,
            horizon=horizon,
            poll_s=poll_s,
            timeout_s=timeout_s,
            extend_to_current=extend_to_current,
            out_dir=out_dir / region_key,
            top_drivers=top_drivers,
        )
    return results


def predict_somalia_cmb(
    *,
    source: str = "cache",
    client: SybilionClient | None = None,
    horizon: int = DEFAULT_PRICE_HORIZON,
    poll_s: float = 5.0,
    timeout_s: float = 900.0,
    out_dir: Path = CMB_OUT_DIR,
    top_drivers: int = 8,
) -> PricePrediction:
    out_dir.mkdir(parents=True, exist_ok=True)
    history = load_somalia_cmb_usd()
    history.to_csv(out_dir / "somalia_cmb_usd_monthly.csv", index=False)

    body = build_somalia_cmb_body(history, horizon=horizon)
    save_json(out_dir / "forecast_body.json", body)

    forecast_payload, job_id = _resolve_forecast(
        body=body,
        client=client,
        source=source,
        out_dir=out_dir,
        forecast_json_path=out_dir / "forecast.json",
        driver_prefix="somalia_cmb_sybilion",
        poll_s=poll_s,
        timeout_s=timeout_s,
        top_drivers=top_drivers,
    )

    return _finalize_price_prediction(
        history=history,
        forecast_payload=forecast_payload,
        out_dir=out_dir,
        csv_path=out_dir / "somalia_cmb_sybilion_forecast.csv",
        plot_title="Sybilion Forecast - Somalia Cost of Minimum Basket",
        plot_ylabel="USD",
        plot_path=out_dir / "somalia_cmb_sybilion_forecast.png",
        job_id=job_id,
    )


def predict_global_fuel_price(
    *,
    source: str = "cache",
    client: SybilionClient | None = None,
    horizon: int = DEFAULT_PRICE_HORIZON,
    poll_s: float = 5.0,
    timeout_s: float = 900.0,
    aggregation: str = "median",
    extend_to_current: bool = True,
    out_dir: Path = FUEL_OUT_DIR,
    top_drivers: int = 8,
) -> PricePrediction:
    out_dir.mkdir(parents=True, exist_ok=True)
    history = load_global_fuel_prices(
        FUEL_RAW_PATH,
        aggregation=aggregation,
        extend_to_current=extend_to_current,
    )
    history.to_csv(out_dir / "global_fuel_price_monthly.csv", index=False)

    body = build_global_fuel_body(history, horizon=horizon, aggregation=aggregation)
    save_json(out_dir / "forecast_body.json", body)

    forecast_payload, job_id = _resolve_forecast(
        body=body,
        client=client,
        source=source,
        out_dir=out_dir,
        forecast_json_path=out_dir / "global_fuel_sybilion_forecast.json",
        driver_prefix="global_fuel_sybilion",
        poll_s=poll_s,
        timeout_s=timeout_s,
        top_drivers=top_drivers,
    )

    return _finalize_price_prediction(
        history=history,
        forecast_payload=forecast_payload,
        out_dir=out_dir,
        csv_path=out_dir / "global_fuel_sybilion_forecast.csv",
        plot_title="Forecast - Global Fuel Price Proxy",
        plot_ylabel="Fuel price proxy",
        plot_path=out_dir / "global_fuel_sybilion_forecast.png",
        job_id=job_id,
    )


def predict_regional_food_price(
    region_name: str,
    *,
    source: str = "local",
    client: SybilionClient | None = None,
    horizon: int = DEFAULT_PRICE_HORIZON,
    poll_s: float = 5.0,
    timeout_s: float = 900.0,
    region_column: str = "admin1",
    aggregation: str = "median",
    extend_to_current: bool = True,
    out_dir: Path | None = None,
    top_drivers: int = 8,
) -> PricePrediction:
    region_key = _slugify_region_name(region_name)
    out_dir = out_dir or REGIONAL_FOOD_OUT_DIR / region_key
    out_dir.mkdir(parents=True, exist_ok=True)
    source_region_name = regional_food_source_region(region_name)
    if source_region_name != region_name:
        _status(f"Using {source_region_name} WFP food-price proxy for {region_name}.")

    history = load_regional_food_prices(
        source_region_name,
        REGIONAL_FOOD_RAW_PATH,
        region_column=region_column,
        aggregation=aggregation,
        extend_to_current=extend_to_current,
    )
    history["region"] = region_name
    history["source_region"] = source_region_name
    history.to_csv(out_dir / f"{region_key}_food_price_monthly.csv", index=False)

    body = build_regional_food_body(
        history,
        region_name=region_name,
        source_region_name=source_region_name,
        horizon=horizon,
        aggregation=aggregation,
    )
    save_json(out_dir / "forecast_body.json", body)

    forecast_payload, job_id = _resolve_forecast(
        body=body,
        client=client,
        source=source,
        out_dir=out_dir,
        forecast_json_path=out_dir / f"{region_key}_food_sybilion_forecast.json",
        driver_prefix=f"{region_key}_food_sybilion",
        poll_s=poll_s,
        timeout_s=timeout_s,
        top_drivers=top_drivers,
    )

    return _finalize_price_prediction(
        history=history,
        forecast_payload=forecast_payload,
        out_dir=out_dir,
        csv_path=out_dir / f"{region_key}_food_sybilion_forecast.csv",
        plot_title=f"Forecast - {region_name} Food Prices",
        plot_ylabel="Food price proxy, USD",
        plot_path=out_dir / f"{region_key}_food_sybilion_forecast.png",
        job_id=job_id,
    )


def predict_regional_food_prices(
    *,
    regions: list[str] | None = None,
    source: str = "local",
    client: SybilionClient | None = None,
    horizon: int = DEFAULT_PRICE_HORIZON,
    poll_s: float = 5.0,
    timeout_s: float = 900.0,
    region_column: str = "admin1",
    aggregation: str = "median",
    extend_to_current: bool = True,
    out_dir: Path = REGIONAL_FOOD_OUT_DIR,
    top_drivers: int = 8,
) -> dict[str, PricePrediction]:
    regions = regions or list_wfp_food_regions(region_column=region_column)
    results: dict[str, PricePrediction] = {}
    for region_name in regions:
        region_key = _slugify_region_name(region_name)
        results[region_key] = predict_regional_food_price(
            region_name,
            source=source,
            client=client,
            horizon=horizon,
            poll_s=poll_s,
            timeout_s=timeout_s,
            region_column=region_column,
            aggregation=aggregation,
            extend_to_current=extend_to_current,
            out_dir=out_dir / region_key,
            top_drivers=top_drivers,
        )
    return results


def predict_somalia_food_prices(**kwargs) -> PricePrediction:
    """Alias for the Somalia CMB food-basket forecast."""
    return predict_somalia_cmb(**kwargs)


def predict_somalia_weather(**kwargs) -> WeatherPrediction:
    """Alias for the currently configured Gedo weather forecast."""
    return predict_gedo_weather(**kwargs)

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from html import escape
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

PREDICTION_UI_CHOICES = ("water", "cmb", "fuel", "regional-water", "regional-food", "weather")
WEATHER_UI_METRICS = ("rainfall_mm_per_day", "temperature_avg_c", "relative_humidity_pct")
ANSI_PATTERN = re.compile(r"\033\[[0-9;]*m")
DEFAULT_DASHBOARD_PATH = PROJECT_ROOT / "reports/aid_dashboard.html"


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


def format_compact_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}k"
    return f"{value:.2f}"


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

    print("\n=== Agent Reasoning Summary ===")
    print(
        "1. Overall allocation prioritizes "
        f"{top_cargo}: {format_budget_value(cargo_totals[top_cargo], percent_mode=percent_mode)} "
        "across all selected regions."
    )
    top_cargo_reason = _top_reason_for(top_cargo_region, top_cargo)
    if top_cargo_reason:
        print(f"   Strongest concrete driver: {top_cargo_region.region}: {top_cargo_reason}")
    print(
        "2. The largest regional budget goes to "
        f"{top_region.region}: {format_budget_value(top_region.regional_budget, percent_mode=percent_mode)}."
    )
    top_region_reason = _top_reason_for(top_region, top_region_cargo)
    if top_region_reason:
        print(
            f"   Forecasts in that region put the strongest pressure on "
            f"{top_region_cargo}: {top_region_reason}"
        )
    print(
        "3. Within that region, the largest predicted need is "
        f"{top_region_cargo}: "
        f"{format_budget_value(top_region.budget_allocation[top_region_cargo], percent_mode=percent_mode)}."
    )


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


def _region_slug(region_name: str) -> str:
    return region_name.lower().replace("-", " ").replace("_", " ").strip().replace(" ", "_")


def _prediction_ui_path(args: argparse.Namespace) -> tuple[Path, str, str]:
    prediction = args.prediction
    if prediction == "water":
        return (
            PROJECT_ROOT / "data/somalia/water/sybilion/somalia_water_sybilion_forecast.csv",
            "National water price",
            "SOS water proxy",
        )
    if prediction == "cmb":
        return (
            PROJECT_ROOT / "data/somalia/sybilion_cmb/somalia_cmb_sybilion_forecast.csv",
            "Cost of minimum basket",
            "USD basket cost",
        )
    if prediction == "fuel":
        return (
            PROJECT_ROOT / "data/somalia/sybilion_fuel/global_fuel_sybilion_forecast.csv",
            "Global fuel proxy",
            "Market proxy",
        )
    if prediction == "regional-water":
        slug = _region_slug(args.region)
        return (
            PROJECT_ROOT
            / f"data/somalia/water/sybilion/regional/{slug}/{slug}_water_sybilion_forecast.csv",
            f"{display_region_name(args.region)} water price",
            "Regional water proxy",
        )
    if prediction == "regional-food":
        slug = _region_slug(args.region)
        return (
            PROJECT_ROOT
            / f"data/somalia/sybilion_regional_food/{slug}/{slug}_food_sybilion_forecast.csv",
            f"{display_region_name(args.region)} food price",
            "Regional food proxy",
        )
    if prediction == "weather":
        metric = args.weather_metric
        return (
            PROJECT_ROOT / f"data/weather/sybilion_gedo/{metric}_forecast.csv",
            f"Gedo {metric.replace('_', ' ')}",
            "Weather forecast",
        )
    raise SystemExit(f"Unsupported prediction: {prediction}")


def _ansi(code: str, enabled: bool) -> str:
    return f"\033[{code}m" if enabled else ""


def _prediction_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    low_column = "q10" if "q10" in frame.columns else "q05" if "q05" in frame.columns else None
    high_column = "q90" if "q90" in frame.columns else "q95" if "q95" in frame.columns else None
    return low_column, high_column


def _fit_text(text: str, width: int) -> str:
    if _visible_len(text) <= width:
        return text
    if width <= 1:
        return ANSI_PATTERN.sub("", text)[:width]
    return f"{ANSI_PATTERN.sub('', text)[: width - 1]}."


def _visible_len(text: str) -> int:
    return len(ANSI_PATTERN.sub("", text))


def _pad_right(text: str, width: int) -> str:
    return text + " " * max(width - _visible_len(text), 0)


def _pad_left(text: str, width: int) -> str:
    return " " * max(width - _visible_len(text), 0) + text


def _ui_rule(width: int) -> str:
    return "+" + "-" * (width - 2) + "+"


def _ui_line(text: str, width: int) -> str:
    text = _fit_text(text, width - 4)
    return f"| {_pad_right(text, width - 4)} |"


def _ui_split(left: str, right: str, width: int) -> str:
    body_width = width - 4
    gap = 2
    left_width = max((body_width - gap) // 2, 10)
    right_width = max(body_width - gap - left_width, 10)
    left = _fit_text(left, left_width)
    right = _fit_text(right, right_width)
    return f"| {_pad_right(left, left_width)}{' ' * gap}{_pad_left(right, right_width)} |"


def _bar(value: float, min_value: float, max_value: float, width: int = 24) -> str:
    if max_value <= min_value:
        filled = width
    else:
        filled = int(round(((value - min_value) / (max_value - min_value)) * width))
    filled = min(max(filled, 1), width)
    return "#" * filled + "." * (width - filled)


def print_prediction_ui(args: argparse.Namespace) -> None:
    path, title, subtitle = _prediction_ui_path(args)
    if not path.exists():
        raise SystemExit(f"Missing cached prediction file: {path}")

    frame = pd.read_csv(path)
    if "date" not in frame.columns or "forecast" not in frame.columns:
        raise SystemExit(f"Prediction file must contain date and forecast columns: {path}")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["forecast"] = pd.to_numeric(frame["forecast"], errors="coerce")
    row_count = max(args.rows, 1)
    frame = frame.dropna(subset=["date", "forecast"]).sort_values("date").head(row_count)
    if frame.empty:
        raise SystemExit(f"Prediction file contains no printable forecast rows: {path}")

    low_column, high_column = _prediction_columns(frame)
    if low_column:
        frame[low_column] = pd.to_numeric(frame[low_column], errors="coerce")
    if high_column:
        frame[high_column] = pd.to_numeric(frame[high_column], errors="coerce")

    first = float(frame["forecast"].iloc[0])
    last = float(frame["forecast"].iloc[-1])
    mean = float(frame["forecast"].mean())
    change = ((last - first) / max(abs(first), 1.0)) * 100.0
    band = None
    if low_column and high_column:
        band = float((frame[high_column] - frame[low_column]).mean())

    color_enabled = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    accent = _ansi("36", color_enabled)
    muted = _ansi("2", color_enabled)
    reset = _ansi("0", color_enabled)
    green = _ansi("32", color_enabled)
    red = _ansi("31", color_enabled)
    trend_color = green if change >= 0 else red

    width = min(max(shutil.get_terminal_size((88, 24)).columns, 72), 100)
    print(_ui_rule(width))
    print(_ui_line(f"{accent}{title.upper()}{reset}", width))
    print(_ui_line(f"{muted}{subtitle}  |  {path.relative_to(PROJECT_ROOT)}{reset}", width))
    print(_ui_rule(width))
    print(_ui_split(f"next {frame['date'].iloc[0].date()}  {format_compact_number(first)}", f"avg {format_compact_number(mean)}", width))
    band_text = f"band {format_compact_number(band)}" if band is not None else "band n/a"
    print(_ui_split(f"last {frame['date'].iloc[-1].date()}  {format_compact_number(last)}", f"trend {trend_color}{change:+.2f}%{reset}", width))
    print(_ui_split(f"horizon {len(frame)} months", band_text, width))
    print(_ui_rule(width))

    values = frame["forecast"].astype(float)
    min_value = float(values.min())
    max_value = float(values.max())
    print(_ui_line("date        forecast      interval        shape", width))
    for _, row in frame.iterrows():
        date = row["date"].date().isoformat()
        forecast = format_compact_number(float(row["forecast"]))
        if low_column and high_column and pd.notna(row[low_column]) and pd.notna(row[high_column]):
            interval = f"{format_compact_number(float(row[low_column]))}-{format_compact_number(float(row[high_column]))}"
        else:
            interval = "n/a"
        line = f"{date}  {forecast:>10}  {interval:>15}  {_bar(float(row['forecast']), min_value, max_value)}"
        print(_ui_line(line, width))
    print(_ui_rule(width))


def _json_ready(value):
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _forecast_records(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, object]]:
    if frame.empty:
        return []
    columns = [column for column in ("date", "forecast", "q10", "q90", "q05", "q95") if column in frame.columns]
    clean = frame[columns].copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    for column in columns:
        if column != "date":
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=["date", "forecast"]).sort_values("date")
    if limit is not None:
        clean = clean.head(limit)
    records = []
    for record in clean.to_dict(orient="records"):
        records.append({key: _json_ready(value) for key, value in record.items() if pd.notna(value)})
    return records


def _history_records(frame: pd.DataFrame, *, limit: int | None = 72) -> list[dict[str, object]]:
    if frame.empty or "date" not in frame.columns or "value" not in frame.columns:
        return []
    clean = frame[["date", "value"]].copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna(subset=["date", "value"]).sort_values("date")
    if limit is not None:
        clean = clean.tail(limit)
    records = []
    for record in clean.to_dict(orient="records"):
        records.append({key: _json_ready(value) for key, value in record.items() if pd.notna(value)})
    return records


def _chart_payload(
    *,
    chart_id: str,
    label: str,
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    history_limit: int = 72,
) -> dict[str, object]:
    return {
        "id": chart_id,
        "label": label,
        "history": _history_records(history, limit=history_limit),
        "forecast": _forecast_records(forecast),
    }


def _decision_records(decisions: dict[str, object], *, percent_mode: bool) -> list[dict[str, object]]:
    records = []
    for decision in decisions.values():
        top_reason = _top_reason_for(decision, decision.selected_cargo)
        records.append(
            {
                "region": decision.region,
                "selectedCargo": decision.selected_cargo,
                "regionalBudget": decision.regional_budget,
                "regionalBudgetLabel": format_budget_value(decision.regional_budget, percent_mode=percent_mode),
                "population": decision.population,
                "accessibility": decision.accessibility,
                "budgetAllocation": decision.budget_allocation,
                "scores": decision.scores,
                "topReason": top_reason,
            }
        )
    return records


def _dashboard_payload(args: argparse.Namespace) -> dict[str, object]:
    live_mode = args.mode == "live"
    data_source = "live" if live_mode else "cache"
    forecast_source = "live" if live_mode else "cache"
    agent_forecast_source = args.agent_forecast_source
    if not live_mode and agent_forecast_source == "live":
        agent_forecast_source = "local"
    fuel_forecast_source = args.fuel_forecast_source
    if not live_mode and fuel_forecast_source == "live":
        fuel_forecast_source = "local"

    client = SybilionClient() if "live" in {forecast_source, agent_forecast_source, fuel_forecast_source} else None
    agent_budget = args.agent_budget if args.agent_budget is not None else args.agent_units

    status(f"Building dashboard data: mode={args.mode}, forecast_source={forecast_source}")
    weather_result = predict_gedo_weather(
        data_source=data_source,
        forecast_source=forecast_source,
        client=client,
        horizon=args.weather_horizon,
        poll_s=args.poll_s,
        timeout_s=args.timeout_s,
        overwrite_weather=live_mode,
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

    agent_regions = normalize_agent_regions(args)
    region_budgets = population_weighted_budget(agent_regions, total_budget=agent_budget)
    agent_decisions = {}
    regional_forecasts = []
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
        regional_forecasts.extend(
            [
                _chart_payload(
                    chart_id=f"{_region_slug(agent_region)}-water",
                    label=f"{agent_region} water",
                    history=regional_water_result.history,
                    forecast=regional_water_result.forecast,
                ),
                _chart_payload(
                    chart_id=f"{_region_slug(agent_region)}-food",
                    label=f"{agent_region} food",
                    history=regional_food_result.history,
                    forecast=regional_food_result.forecast,
                ),
            ]
        )
        agent_decisions[agent_region] = make_aid_decision(
            region=agent_region,
            water_prediction=regional_water_result,
            food_prediction=regional_food_result,
            fuel_prediction=fuel_result,
            weather_prediction=weather_for_agent_region(agent_region, weather_result),
            budget=region_budgets[agent_region],
            population=REGION_POPULATIONS[agent_region],
            reasoning="reasons",
        )

    total_budget = sum(decision.regional_budget for decision in agent_decisions.values())
    percent_mode = is_percent_budget(total_budget)
    return {
        "generatedAt": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "mode": args.mode,
        "budgetLabel": format_budget_value(total_budget, percent_mode=percent_mode),
        "decisions": _decision_records(agent_decisions, percent_mode=percent_mode),
        "forecasts": [
            _chart_payload(
                chart_id="water",
                label="National water",
                history=water_result.history,
                forecast=water_result.forecast,
            ),
            _chart_payload(
                chart_id="cmb",
                label="Minimum basket",
                history=cmb_result.history,
                forecast=cmb_result.forecast,
            ),
            _chart_payload(
                chart_id="fuel",
                label="Fuel proxy",
                history=fuel_result.history,
                forecast=fuel_result.forecast,
            ),
            *regional_forecasts,
            *[
                _chart_payload(
                    chart_id=f"weather-{metric_name}",
                    label=metric_name.replace("_", " "),
                    history=monthly_timeseries(weather_result.history, "month", metric_name),
                    forecast=forecast,
                )
                for metric_name, forecast in weather_result.forecasts.items()
            ],
        ],
    }


def _dashboard_html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=True)
    title = "Aid Allocation Dashboard"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #1d2424;
      --muted: #68706c;
      --line: #d9ddd5;
      --panel: #ffffff;
      --a: #176b87;
      --b: #b94747;
      --c: #738a3d;
      --d: #6d5d93;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      padding: 24px 28px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
    }}
    h1 {{ margin: 0; font-size: 28px; font-weight: 720; }}
    .meta {{ color: var(--muted); font-size: 13px; text-align: right; }}
    main {{ padding: 20px 28px 32px; display: grid; gap: 20px; }}
    section {{ display: grid; gap: 12px; }}
    h2 {{ margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 12px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }}
    .span-4 {{ grid-column: span 4; }}
    .span-6 {{ grid-column: span 6; }}
    .span-8 {{ grid-column: span 8; }}
    .span-12 {{ grid-column: span 12; }}
    .decision-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: start; margin-bottom: 12px; }}
    .region {{ font-size: 18px; font-weight: 720; }}
    .selected {{ color: var(--a); font-size: 13px; font-weight: 680; text-align: right; }}
    .number {{ font-size: 24px; font-weight: 760; margin-bottom: 10px; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(112px, 1fr) 2fr 74px; align-items: center; gap: 8px; margin: 7px 0; font-size: 13px; }}
    .bar {{ height: 10px; background: #eceee8; border-radius: 999px; overflow: hidden; }}
    .fill {{ height: 100%; width: 0; background: var(--a); }}
    .fill.food_supplies {{ background: var(--b); }}
    .fill.water_infrastructure {{ background: var(--c); }}
    .fill.fuel {{ background: var(--d); }}
    .reason {{ border-top: 1px solid var(--line); padding-top: 10px; margin-top: 12px; color: var(--muted); font-size: 13px; line-height: 1.4; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    button {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 10px;
      cursor: pointer;
      font: inherit;
      font-size: 13px;
    }}
    button.active {{ border-color: var(--a); color: var(--a); }}
    svg {{ width: 100%; height: 380px; display: block; }}
    .chart-labels {{ display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; margin-top: 8px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; color: var(--muted); font-size: 12px; margin-top: 10px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 22px; height: 3px; display: inline-block; background: var(--a); }}
    .swatch.history {{ background: #6f7772; }}
    .swatch.band {{ background: #b8d6df; }}
    @media (max-width: 900px) {{
      header {{ display: grid; align-items: start; padding: 20px; }}
      .meta {{ text-align: left; }}
      main {{ padding: 16px 20px 28px; }}
      .span-4, .span-6, .span-8 {{ grid-column: span 12; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 5px; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Aid Allocation Dashboard</h1>
      <div class="muted">Agent output and forecast signals for Somalia allocation planning</div>
    </div>
    <div class="meta">
      <div id="generated"></div>
      <div id="mode"></div>
    </div>
  </header>
  <main>
    <section>
      <h2>Agent Output</h2>
      <div class="grid" id="decisions"></div>
    </section>
    <section>
      <h2>Predictions</h2>
      <div class="tabs" id="tabs"></div>
      <div class="panel span-12">
        <svg id="chart" role="img" aria-label="Forecast chart"></svg>
        <div class="legend">
          <span><i class="swatch history"></i>History</span>
          <span><i class="swatch"></i>Forecast</span>
          <span><i class="swatch band"></i>Uncertainty band</span>
        </div>
        <div class="chart-labels"><span id="chartStart"></span><span id="chartEnd"></span></div>
      </div>
    </section>
  </main>
  <script>
    const data = {payload_json};
    const cargoLabels = {{
      water_supplies: "Water supplies",
      water_infrastructure: "Water infrastructure",
      food_supplies: "Food supplies",
      fuel: "Fuel"
    }};
    document.getElementById("generated").textContent = `Generated ${{data.generatedAt}}`;
    document.getElementById("mode").textContent = `Mode ${{data.mode}} | Budget ${{data.budgetLabel}}`;

    const decisions = document.getElementById("decisions");
    const globalAllocationMax = Math.max(
      ...data.decisions.flatMap((decision) =>
        Object.values(decision.budgetAllocation).map((value) => Number(value) || 0)
      ),
      1
    );
    data.decisions.forEach((decision) => {{
      const panel = document.createElement("article");
      panel.className = "panel span-4";
      const allocations = Object.entries(decision.budgetAllocation);
      panel.innerHTML = `
        <div class="decision-head">
          <div>
            <div class="region">${{decision.region}}</div>
            <div class="muted">Population ${{Number(decision.population).toLocaleString()}} | Access ${{decision.accessibility.toFixed(2)}}</div>
          </div>
          <div class="selected">${{cargoLabels[decision.selectedCargo] || decision.selectedCargo}}</div>
        </div>
        <div class="number">${{decision.regionalBudgetLabel}}</div>
        ${{allocations.map(([key, value]) => `
          <div class="bar-row">
            <div>${{cargoLabels[key] || key}}</div>
            <div class="bar"><div class="fill ${{key}}" style="width:${{Math.max((value / globalAllocationMax) * 100, 3)}}%"></div></div>
            <div class="muted">${{Number(value).toFixed(2)}}</div>
          </div>
        `).join("")}}
        <div class="reason">${{decision.topReason || "No dominant driver available."}}</div>
      `;
      decisions.appendChild(panel);
    }});

    const tabs = document.getElementById("tabs");
    const chart = document.getElementById("chart");
    const chartStart = document.getElementById("chartStart");
    const chartEnd = document.getElementById("chartEnd");

    function fmt(value) {{
      const abs = Math.abs(value);
      if (abs >= 1000000) return `${{(value / 1000000).toFixed(1)}}M`;
      if (abs >= 1000) return `${{(value / 1000).toFixed(1)}}k`;
      if (abs >= 10) return value.toFixed(1);
      return value.toFixed(2);
    }}

    function drawChart(forecast) {{
      const history = forecast.history || [];
      const forecastSeries = forecast.forecast || [];
      const combined = [
        ...history.map((point) => ({{...point, kind: "history", main: point.value}})),
        ...forecastSeries.map((point) => ({{...point, kind: "forecast", main: point.forecast}}))
      ].filter((point) => Number.isFinite(point.main));
      chart.innerHTML = "";
      if (!combined.length) return;
      const width = 960;
      const height = 380;
      const padLeft = 72;
      const padRight = 28;
      const padTop = 42;
      const padBottom = 58;
      chart.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      const dates = combined.map((point) => new Date(point.date).getTime());
      const minDate = Math.min(...dates);
      const maxDate = Math.max(...dates);
      const values = combined.flatMap((point) => [point.main, point.q10, point.q90, point.q05, point.q95].filter((value) => Number.isFinite(value)));
      const rawMin = Math.min(...values);
      const rawMax = Math.max(...values);
      const rawSpan = Math.max(rawMax - rawMin, 1);
      const min = rawMin - rawSpan * 0.08;
      const max = rawMax + rawSpan * 0.08;
      const span = Math.max(max - min, 1);
      const xDate = (date) => padLeft + ((new Date(date).getTime() - minDate) / Math.max(maxDate - minDate, 1)) * (width - padLeft - padRight);
      const y = (value) => height - padBottom - ((value - min) / span) * (height - padTop - padBottom);
      const line = (points, key) => points
        .filter((point) => Number.isFinite(point[key]))
        .map((point) => `${{xDate(point.date)}},${{y(point[key])}}`)
        .join(" ");
      const yTicks = [0, .25, .5, .75, 1].map((step) => min + step * span);
      const grid = yTicks.map((tick) => {{
        const yy = y(tick);
        return `
          <line x1="${{padLeft}}" y1="${{yy}}" x2="${{width - padRight}}" y2="${{yy}}" stroke="#e1e5dd" />
          <text x="${{padLeft - 10}}" y="${{yy + 4}}" text-anchor="end" fill="#4f5753" font-size="12">${{fmt(tick)}}</text>
        `;
      }}).join("");
      const xTickPoints = [combined[0], combined[Math.floor(combined.length / 2)], combined[combined.length - 1]];
      const xTicks = xTickPoints.map((point) => {{
        const xx = xDate(point.date);
        const label = String(point.date).slice(0, 7);
        return `
          <line x1="${{xx}}" y1="${{height - padBottom}}" x2="${{xx}}" y2="${{height - padBottom + 6}}" stroke="#1d2424" />
          <text x="${{xx}}" y="${{height - 24}}" text-anchor="middle" fill="#4f5753" font-size="12">${{label}}</text>
        `;
      }}).join("");
      const lower = forecastSeries.some((point) => Number.isFinite(point.q10)) ? "q10" : "q05";
      const upper = forecastSeries.some((point) => Number.isFinite(point.q90)) ? "q90" : "q95";
      const hasBand = forecastSeries.some((point) => Number.isFinite(point[lower]) && Number.isFinite(point[upper]));
      const band = hasBand ? `
        <polyline points="${{line(forecastSeries, upper)}}" fill="none" stroke="#b8d6df" stroke-width="2.5" />
        <polyline points="${{line(forecastSeries, lower)}}" fill="none" stroke="#b8d6df" stroke-width="2.5" />
      ` : "";
      const historyLine = history.length ? `<polyline points="${{line(history, "value")}}" fill="none" stroke="#6f7772" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />` : "";
      const forecastLine = forecastSeries.length ? `<polyline points="${{line(forecastSeries, "forecast")}}" fill="none" stroke="#176b87" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />` : "";
      const divider = history.length && forecastSeries.length ? `<line x1="${{xDate(forecastSeries[0].date)}}" y1="${{padTop}}" x2="${{xDate(forecastSeries[0].date)}}" y2="${{height - padBottom}}" stroke="#9ba49d" stroke-dasharray="5 6" />` : "";
      chart.innerHTML = `
        ${{grid}}
        <line x1="${{padLeft}}" y1="${{padTop}}" x2="${{padLeft}}" y2="${{height - padBottom}}" stroke="#1d2424" stroke-width="1.5" />
        <line x1="${{padLeft}}" y1="${{height - padBottom}}" x2="${{width - padRight}}" y2="${{height - padBottom}}" stroke="#1d2424" stroke-width="1.5" />
        ${{xTicks}}
        ${{divider}}
        ${{historyLine}}
        ${{band}}
        ${{forecastLine}}
        ${{forecastSeries.map((point) => `<circle cx="${{xDate(point.date)}}" cy="${{y(point.forecast)}}" r="4" fill="#176b87" />`).join("")}}
        <text x="${{padLeft}}" y="24" fill="#1d2424" font-size="16" font-weight="700">${{forecast.label}}</text>
        <text x="${{padLeft}}" y="${{height - 8}}" fill="#68706c" font-size="12">x: month | y: value</text>
        <text x="${{width - padRight}}" y="${{height - 8}}" text-anchor="end" fill="#68706c" font-size="12">range ${{fmt(rawMin)}} to ${{fmt(rawMax)}}</text>
      `;
      chartStart.textContent = `History starts ${{combined[0].date}}`;
      chartEnd.textContent = `Forecast ends ${{combined[combined.length - 1].date}}`;
    }}

    const defaultForecastIndex = Math.max(
      data.forecasts.findIndex((forecast) => forecast.id === "weather-relative_humidity_pct"),
      0
    );

    data.forecasts.forEach((forecast, index) => {{
      const button = document.createElement("button");
      button.textContent = forecast.label;
      button.type = "button";
      button.addEventListener("click", () => {{
        document.querySelectorAll("button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        drawChart(forecast);
      }});
      if (index === defaultForecastIndex) button.classList.add("active");
      tabs.appendChild(button);
    }});
    drawChart(data.forecasts[defaultForecastIndex]);
  </script>
</body>
</html>
"""


def write_dashboard(args: argparse.Namespace) -> Path:
    output_path = Path(args.dashboard_output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    payload = _dashboard_payload(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_dashboard_html(payload), encoding="utf-8")
    print(f"Dashboard written to: {output_path}")
    return output_path


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
        choices=["run", "dashboard", "prediction-ui", "config-show", "config-set"],
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
    parser.add_argument(
        "--prediction",
        choices=PREDICTION_UI_CHOICES,
        default="water",
        help="Prediction to print with prediction-ui.",
    )
    parser.add_argument(
        "--region",
        default="Gedo",
        help="Region for regional-water or regional-food prediction-ui output.",
    )
    parser.add_argument(
        "--weather-metric",
        choices=WEATHER_UI_METRICS,
        default="rainfall_mm_per_day",
        help="Weather metric for prediction-ui --prediction weather.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Number of forecast rows to show in prediction-ui.",
    )
    parser.add_argument(
        "--dashboard-output",
        default=str(DEFAULT_DASHBOARD_PATH.relative_to(PROJECT_ROOT)),
        help="HTML output path for dashboard.",
    )
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
    if args.command == "prediction-ui":
        print_prediction_ui(args)
        return
    if args.command == "dashboard":
        if "--mode" not in sys.argv:
            args.mode = "cached"
        if "--agent-forecast-source" not in sys.argv:
            args.agent_forecast_source = "cache"
        if "--fuel-forecast-source" not in sys.argv:
            args.fuel_forecast_source = "cache"
        write_dashboard(args)
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()

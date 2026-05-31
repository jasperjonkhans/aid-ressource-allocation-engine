from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from project.config import CONFIG

_SYBILION_CONFIG = CONFIG["clients"]["sybilion"]

API_BASE = _SYBILION_CONFIG["api_base"]
DEFAULT_TOKEN_PATHS = tuple(_SYBILION_CONFIG["default_token_paths"])


@dataclass(frozen=True)
class ForecastJob:
    job_id: str
    status: str | None = None
    settled: bool | None = None
    payload: dict | None = None


def load_token(
    env_var: str = "SYBILION_API_TOKEN",
    token_paths: Iterable[str | Path] = DEFAULT_TOKEN_PATHS,
) -> str:
    token = os.environ.get(env_var)
    if token:
        return token.strip()

    for token_path in token_paths:
        path = Path(token_path).expanduser()
        if path.exists():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token

    raise RuntimeError(
        f"Set {env_var} or put the Sybilion token in one of: "
        + ", ".join(str(path) for path in token_paths)
    )


def _headers(token: str, json_body: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def monthly_timeseries(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    *,
    interpolate: bool = True,
) -> pd.DataFrame:
    series = df[[date_column, value_column]].copy()
    series[date_column] = pd.to_datetime(series[date_column])
    series[value_column] = pd.to_numeric(series[value_column], errors="coerce")
    series = series.dropna(subset=[date_column]).sort_values(date_column)
    series[date_column] = series[date_column].dt.to_period("M").dt.to_timestamp()
    series = series.groupby(date_column, as_index=False)[value_column].mean()

    full_months = pd.date_range(series[date_column].min(), series[date_column].max(), freq="MS")
    series = series.set_index(date_column).reindex(full_months)
    series.index.name = date_column

    if interpolate:
        series[value_column] = series[value_column].interpolate(limit_direction="both")

    return series.reset_index().rename(
        columns={date_column: "date", value_column: "value"}
    )


def build_forecast_body(
    ts: pd.DataFrame,
    *,
    title: str,
    description: str,
    keywords: list[str] | None = None,
    horizon: int = 12,
    frequency: str = "monthly",
    backtest: bool = True,
    recency_factor: float = 0.75,
    strictly_positive: bool = False,
) -> dict:
    required = {"date", "value"}
    missing = required - set(ts.columns)
    if missing:
        raise ValueError(f"Timeseries is missing columns: {sorted(missing)}")

    clean = ts.dropna(subset=["date", "value"]).copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna(subset=["value"]).sort_values("date")

    timeseries = {
        row["date"].strftime("%Y-%m-%d"): round(float(row["value"]), 6)
        for _, row in clean.iterrows()
    }

    return {
        "pipeline_version": "v1",
        "soft_horizon": horizon,
        "frequency": frequency,
        "backtest": backtest,
        "recency_factor": recency_factor,
        "strictly_positive": strictly_positive,
        "timeseries_metadata": {
            "title": title,
            "description": description,
            "keywords": keywords or [],
        },
        "timeseries": timeseries,
    }


def forecast_to_frame(forecast: dict) -> pd.DataFrame:
    series = forecast.get("data", {}).get("forecast_series", {})
    rows = []

    for date, entry in series.items():
        quantiles = entry.get("quantile_forecast", {})
        row = {
            "date": pd.to_datetime(date),
            "forecast": entry.get("forecast", quantiles.get("0.50", quantiles.get("0.5"))),
        }

        for quantile in ("0.05", "0.10", "0.25", "0.50", "0.75", "0.90", "0.95"):
            compact = quantile.rstrip("0").rstrip(".")
            column = "q" + quantile.split(".")[1]
            row[column] = quantiles.get(quantile, quantiles.get(compact))

        rows.append(row)

    if not rows:
        raise RuntimeError("Forecast payload did not contain data.forecast_series.")

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _overall_metric(entry: dict, group: str, metric: str = "mean") -> float | None:
    value = entry.get(group, {}).get("overall", {}).get(metric)
    if value is None:
        return None
    return float(value)


def external_signals_to_frame(external_signals: dict) -> pd.DataFrame:
    signals = external_signals.get("data", {})
    rows = []

    for driver_id, entry in signals.items():
        rows.append(
            {
                "driver_id": driver_id,
                "driver_name": entry.get("driver_name", driver_id),
                "importance_mean": _overall_metric(entry, "importance"),
                "importance_max": _overall_metric(entry, "importance", "max"),
                "correlation_mean": _overall_metric(entry, "pearson_correlation"),
                "correlation_max": _overall_metric(entry, "pearson_correlation", "max"),
                "correlation_min": _overall_metric(entry, "pearson_correlation", "min"),
                "direction_mean": _overall_metric(entry, "direction"),
            }
        )

    if not rows:
        raise RuntimeError("External signals payload did not contain data.")

    df = pd.DataFrame(rows)
    df["importance_mean"] = df["importance_mean"].fillna(0.0)
    df["abs_correlation_mean"] = df["correlation_mean"].abs()
    return df.sort_values(
        ["importance_mean", "abs_correlation_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)


def print_top_drivers(drivers: pd.DataFrame, *, top_n: int = 8) -> None:
    columns = [
        "driver_name",
        "importance_mean",
        "correlation_mean",
        "direction_mean",
    ]
    printable = drivers.head(top_n).loc[:, columns].copy()
    print("\nTop Sybilion drivers")
    print(printable.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def save_driver_plot(
    drivers: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Sybilion forecast drivers",
    top_n: int = 8,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    top = drivers.head(top_n).copy()
    top["label"] = (
        top["driver_name"].str.slice(0, 54)
        + " ["
        + top["driver_id"].str.slice(0, 8)
        + "]"
    )
    top = top.iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, top_n * 0.55)))
    axes[0].barh(top["label"], top["importance_mean"], color="#2a9d8f")
    axes[0].set_title("Driver importance")
    axes[0].set_xlabel("Mean importance")
    axes[0].grid(axis="x", alpha=0.25)

    colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in top["correlation_mean"]]
    axes[1].barh(top["label"], top["correlation_mean"], color=colors)
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Pearson correlation")
    axes[1].set_xlabel("Mean correlation")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_driver_outputs(
    external_signals: dict,
    out_dir: Path,
    *,
    prefix: str = "sybilion",
    top_n: int = 8,
    print_drivers: bool = True,
    plot_drivers: bool = True,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_external_signals.json").write_text(
        json.dumps(external_signals, indent=2),
        encoding="utf-8",
    )
    drivers = external_signals_to_frame(external_signals)

    drivers.to_csv(out_dir / f"{prefix}_drivers.csv", index=False)

    if print_drivers:
        print_top_drivers(drivers, top_n=top_n)

    if plot_drivers:
        save_driver_plot(
            drivers,
            out_dir / f"{prefix}_drivers.png",
            title=f"Sybilion drivers - {prefix}",
            top_n=top_n,
        )

    return drivers


class SybilionClient:
    def __init__(self, token: str | None = None, api_base: str = API_BASE):
        self.token = token or load_token()
        self.api_base = api_base.rstrip("/")

    def submit_forecast(self, body: dict) -> ForecastJob:
        response = requests.post(
            f"{self.api_base}/forecasts",
            headers=_headers(self.token, json_body=True),
            json=body,
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Sybilion forecast submit failed ({response.status_code}): {response.text}"
            )

        payload = response.json()
        job_id = payload.get("job_id")
        if not job_id:
            raise RuntimeError(f"Sybilion response did not include job_id: {payload}")

        return ForecastJob(job_id=job_id, payload=payload)

    def get_forecast_job(self, job_id: str) -> ForecastJob:
        response = requests.get(
            f"{self.api_base}/forecasts/{job_id}",
            headers=_headers(self.token),
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Sybilion forecast poll failed ({response.status_code}): {response.text}"
            )

        payload = response.json()
        return ForecastJob(
            job_id=job_id,
            status=payload.get("status"),
            settled=payload.get("settled"),
            payload=payload,
        )

    def wait_for_forecast(
        self,
        job_id: str,
        *,
        poll_s: float = 5.0,
        timeout_s: float = 900.0,
        verbose: bool = True,
    ) -> ForecastJob:
        started = time.time()

        while True:
            job = self.get_forecast_job(job_id)
            if verbose:
                print(f"Job {job_id}: status={job.status}, settled={job.settled}")

            if job.settled is True or job.status in {"completed", "failed", "canceled"}:
                return job

            if time.time() - started > timeout_s:
                raise TimeoutError(f"Forecast job did not settle within {timeout_s} seconds.")

            time.sleep(poll_s)

    def download_artifact(self, job_id: str, artifact_name: str) -> bytes:
        response = requests.get(
            f"{self.api_base}/forecasts/{job_id}/artifacts/{artifact_name}",
            headers=_headers(self.token),
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Sybilion artifact download failed ({response.status_code}): {response.text}"
            )

        return response.content

    def download_json_artifact(self, job_id: str, artifact_name: str) -> dict:
        return json.loads(self.download_artifact(job_id, artifact_name).decode("utf-8"))

    def run_forecast(
        self,
        body: dict,
        *,
        poll_s: float = 5.0,
        timeout_s: float = 900.0,
        verbose: bool = True,
        driver_out_dir: str | Path | None = None,
        driver_prefix: str = "sybilion",
        top_drivers: int = 8,
        print_drivers: bool = True,
        plot_drivers: bool = True,
    ) -> tuple[ForecastJob, dict]:
        submitted = self.submit_forecast(body)
        if verbose:
            print(f"Submitted Sybilion forecast job: {submitted.job_id}")

        job = self.wait_for_forecast(
            submitted.job_id,
            poll_s=poll_s,
            timeout_s=timeout_s,
            verbose=verbose,
        )
        if job.status in {"failed", "canceled"}:
            raise RuntimeError(f"Sybilion job ended with status={job.status}.")

        forecast = self.download_json_artifact(submitted.job_id, "forecast.json")
        try:
            external_signals = self.download_json_artifact(
                submitted.job_id,
                "external_signals.json",
            )
        except RuntimeError as exc:
            if verbose:
                print(f"Could not download Sybilion drivers: {exc}")
        else:
            if driver_out_dir is not None:
                out_dir = Path(driver_out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{driver_prefix}_external_signals.json").write_text(
                    json.dumps(external_signals, indent=2),
                    encoding="utf-8",
                )
            try:
                drivers = external_signals_to_frame(external_signals)
            except RuntimeError as exc:
                if verbose:
                    print(f"Could not parse Sybilion drivers: {exc}")
            else:
                if print_drivers:
                    print_top_drivers(drivers, top_n=top_drivers)
                if driver_out_dir is not None:
                    save_driver_outputs(
                        external_signals,
                        Path(driver_out_dir),
                        prefix=driver_prefix,
                        top_n=top_drivers,
                        print_drivers=False,
                        plot_drivers=plot_drivers,
                    )

        return job, forecast

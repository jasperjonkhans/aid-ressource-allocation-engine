from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


API_BASE = "https://api.sybilion.dev/api/v1"
DEFAULT_TOKEN_PATHS = ("API_KEY.txt", "project/API_KEY.txt")


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
        return job, forecast

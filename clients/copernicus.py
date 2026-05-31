from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from project.config import CONFIG, project_path

_COPERNICUS_CONFIG = CONFIG["clients"]["copernicus"]

CDS_URL = _COPERNICUS_CONFIG["cds_url"]
ERA5_LAND_MONTHLY_DATASET = _COPERNICUS_CONFIG["era5_land_monthly_dataset"]
DEFAULT_DATA_DIR = project_path(_COPERNICUS_CONFIG["default_data_dir"])
MONTHLY_VARIABLES = list(_COPERNICUS_CONFIG["monthly_variables"])


@dataclass(frozen=True)
class CopernicusCredentials:
    url: str
    key: str


def _read_cdsapirc(path: str | Path) -> CopernicusCredentials | None:
    path = Path(path).expanduser()
    if not path.exists():
        return None

    values = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()

    api_key = values.get("key")
    if not api_key:
        return None

    return CopernicusCredentials(url=values.get("url", CDS_URL), key=api_key)


def get_credentials() -> CopernicusCredentials:
    env_key = os.environ.get("CDSAPI_KEY")
    if env_key:
        return CopernicusCredentials(
            url=os.environ.get("CDSAPI_URL", CDS_URL),
            key=env_key,
        )

    for path in (".cdsapirc", "aid-ressource-allocation-engine/.cdsapirc", "~/.cdsapirc"):
        credentials = _read_cdsapirc(path)
        if credentials:
            return credentials

    raise RuntimeError(
        "CDS API credentials are missing. Set CDSAPI_KEY or create .cdsapirc "
        "with 'url: https://cds.climate.copernicus.eu/api' and 'key: <token>'."
    )


def get_client() -> cdsapi.Client:
    credentials = get_credentials()
    return cdsapi.Client(url=credentials.url, key=credentials.key)


def normalize_polygon(
    polygon: Iterable[Sequence[float] | dict] | dict,
) -> list[tuple[float, float]]:
    """
    Normalize polygon coordinates to [(lon, lat), ...].

    Accepted point formats:
    - (lon, lat)
    - {"lon": ..., "lat": ...}
    - {"longitude": ..., "latitude": ...}
    - GeoJSON Polygon
    - GeoJSON Feature with Polygon geometry
    """
    if isinstance(polygon, dict) and polygon.get("type") == "Feature":
        polygon = polygon["geometry"]

    if isinstance(polygon, dict) and polygon.get("type") == "Polygon":
        polygon = polygon["coordinates"][0]

    points = []
    for point in polygon:
        if isinstance(point, dict):
            lon = point.get("lon", point.get("longitude"))
            lat = point.get("lat", point.get("latitude"))
        else:
            lon, lat = point[0], point[1]

        if lon is None or lat is None:
            raise ValueError(f"Invalid polygon point: {point!r}")

        points.append((float(lon), float(lat)))

    if len(points) < 3:
        raise ValueError("A polygon needs at least 3 points.")

    if points[0] != points[-1]:
        points.append(points[0])

    return points


def polygon_bbox(polygon: Iterable[Sequence[float] | dict], padding=0.0) -> list[float]:
    points = normalize_polygon(polygon)
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]

    north = max(lats) + padding
    west = min(lons) - padding
    south = min(lats) - padding
    east = max(lons) + padding

    return [north, west, south, east]


def _point_in_polygon(lon: float, lat: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1

    for i, point in enumerate(polygon):
        lon_i, lat_i = point
        lon_j, lat_j = polygon[j]

        crosses_lat = (lat_i > lat) != (lat_j > lat)
        if crosses_lat:
            intersect_lon = (lon_j - lon_i) * (lat - lat_i) / (lat_j - lat_i) + lon_i
            if lon < intersect_lon:
                inside = not inside
        j = i

    return inside


def _polygon_mask(
    latitudes: xr.DataArray,
    longitudes: xr.DataArray,
    polygon: list[tuple[float, float]],
) -> xr.DataArray:
    mask_values = np.array(
        [
            [_point_in_polygon(float(lon), float(lat), polygon) for lon in longitudes.values]
            for lat in latitudes.values
        ],
        dtype=bool,
    )

    if not mask_values.any():
        raise ValueError(
            "No ERA5-Land grid-cell centers fall inside the polygon. "
            "Use a larger polygon or a finer grid."
        )

    return xr.DataArray(
        mask_values,
        coords={"latitude": latitudes, "longitude": longitudes},
        dims=("latitude", "longitude"),
    )


def _time_coord_name(ds: xr.Dataset) -> str:
    for candidate in ("valid_time", "time"):
        if candidate in ds.coords or candidate in ds.dims:
            return candidate
    raise ValueError(f"Could not find a time coordinate. Available coords: {list(ds.coords)}")


def _relative_humidity_from_dewpoint(t_c: xr.DataArray, td_c: xr.DataArray) -> xr.DataArray:
    numerator = np.exp((17.625 * td_c) / (243.04 + td_c))
    denominator = np.exp((17.625 * t_c) / (243.04 + t_c))
    return (100 * numerator / denominator).clip(0, 100)


def _years(start_year: int, end_year: int) -> list[str]:
    return [str(year) for year in range(start_year, end_year + 1)]


def _months() -> list[str]:
    return [f"{month:02d}" for month in range(1, 13)]


def _cache_key(
    polygon: list[tuple[float, float]],
    start_year: int,
    end_year: int,
    grid: float,
) -> str:
    raw = repr((polygon, start_year, end_year, grid)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def download_monthly_era5_land_for_polygon(
    polygon: Iterable[Sequence[float] | dict],
    start_year: int,
    end_year: int,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    grid: float = 0.1,
    overwrite: bool = False,
) -> Path:
    """
    Download monthly ERA5-Land data for the polygon bounding box.

    CDS accepts rectangular areas for this dataset, so polygon clipping is applied
    locally when converting the downloaded NetCDF to a DataFrame.
    """
    normalized_polygon = normalize_polygon(polygon)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    target = data_dir / (
        f"era5_land_monthly_{start_year}_{end_year}_"
        f"{_cache_key(normalized_polygon, start_year, end_year, grid)}.nc"
    )
    if target.exists() and not overwrite:
        return target

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": MONTHLY_VARIABLES,
        "year": _years(start_year, end_year),
        "month": _months(),
        "time": ["00:00"],
        "area": polygon_bbox(normalized_polygon, padding=grid),
        "grid": [grid, grid],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    get_client().retrieve(ERA5_LAND_MONTHLY_DATASET, request, str(target))
    return target


def monthly_weather_dataframe_from_netcdf(
    netcdf_path: str | Path,
    polygon: Iterable[Sequence[float] | dict],
    region_name: str | None = None,
) -> pd.DataFrame:
    polygon = normalize_polygon(polygon)
    ds = xr.open_dataset(netcdf_path)
    time_name = _time_coord_name(ds)

    mask = _polygon_mask(ds["latitude"], ds["longitude"], polygon)

    t2m_c = (ds["t2m"] - 273.15).where(mask).mean(
        dim=("latitude", "longitude"),
        skipna=True,
    )
    d2m_c = (ds["d2m"] - 273.15).where(mask).mean(
        dim=("latitude", "longitude"),
        skipna=True,
    )
    rainfall_mm_per_day = (ds["tp"] * 1000).where(mask).mean(
        dim=("latitude", "longitude"),
        skipna=True,
    )
    relative_humidity_pct = _relative_humidity_from_dewpoint(t2m_c, d2m_c)

    frame = xr.Dataset(
        {
            "rainfall_mm_per_day": rainfall_mm_per_day,
            "temperature_avg_c": t2m_c,
            "relative_humidity_pct": relative_humidity_pct,
        }
    ).to_dataframe().reset_index()

    frame = frame.rename(columns={time_name: "month"})
    frame["month"] = pd.to_datetime(frame["month"]).dt.to_period("M").dt.to_timestamp()

    if region_name is not None:
        frame.insert(0, "region", region_name)

    columns = [
        column
        for column in (
            "region",
            "month",
            "rainfall_mm_per_day",
            "temperature_avg_c",
            "relative_humidity_pct",
        )
        if column in frame.columns
    ]
    return frame[columns].sort_values("month").reset_index(drop=True)


def fetch_monthly_weather_for_polygon(
    polygon: Iterable[Sequence[float] | dict],
    start_year: int,
    end_year: int,
    region_name: str | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    grid: float = 0.1,
    overwrite: bool = False,
) -> pd.DataFrame:
    netcdf_path = download_monthly_era5_land_for_polygon(
        polygon=polygon,
        start_year=start_year,
        end_year=end_year,
        data_dir=data_dir,
        grid=grid,
        overwrite=overwrite,
    )
    return monthly_weather_dataframe_from_netcdf(
        netcdf_path=netcdf_path,
        polygon=polygon,
        region_name=region_name,
    )


def summarize_weather(df: pd.DataFrame) -> dict[str, float]:
    return {
        "rainfall_mm_per_day_mean": float(df["rainfall_mm_per_day"].mean()),
        "temperature_avg_c_mean": float(df["temperature_avg_c"].mean()),
        "relative_humidity_pct_mean": float(df["relative_humidity_pct"].mean()),
    }

# CLI Usage

Run all commands from the repository root. In GitHub / Codespaces this is expected to be `/projects`.

## Setup

Install Python dependencies:

```bash
python -m pip install -r project/requirements.txt
```

Live Sybilion runs require a token in `SYBILION_API_TOKEN`, `API_KEY.txt`, or `project/API_KEY.txt`.

Live Copernicus weather refreshes require CDS credentials via `CDSAPI_KEY` / `CDSAPI_URL` or a `.cdsapirc` file.

## Main Pipeline

Default cached run:

```bash
python project/app.py
python project/app.py run --mode cache
```

This loads cached weather and cached Sybilion outputs where available, builds the fuel local baseline by default, and prints agent allocation decisions.

Refresh data and submit live forecasts:

```bash
python project/app.py run --mode refresh
```

Useful overrides:

```bash
python project/app.py run --data-source cache --forecast-source live
python project/app.py run --overwrite-weather
python project/app.py run --weather-horizon 12 --price-horizon 6
python project/app.py run --poll-s 5 --timeout-s 900
python project/app.py run --water-aggregation median --fuel-aggregation median
python project/app.py run --fuel-forecast-source local
python project/app.py run --top-drivers 8
```

## Agent Runs

Run the full pipeline but skip the agent layer:

```bash
python project/app.py run --skip-agent
```

Run one region:

```bash
python project/app.py run --agent-region Gedo
```

Run multiple regions:

```bash
python project/app.py run --agent-regions Bay Bakool Gedo
```

Set a total money budget. The app splits it across selected regions by population before allocating cargo classes:

```bash
python project/app.py run --agent-budget 12000000
```

Use regional local seasonal baselines for water/food forecasts:

```bash
python project/app.py run --agent-forecast-source local
```

Use cached or live regional forecasts:

```bash
python project/app.py run --agent-forecast-source cache
python project/app.py run --agent-forecast-source live
```

`--agent-units` is kept as a deprecated alias for `--agent-budget`.

## Config

Print the full agent config:

```bash
python project/app.py config-show
```

Print one section or value:

```bash
python project/app.py config-show agent
python project/app.py config-show agent.total_budget
python project/app.py config-show agent.weights
```

Update one value. Values are parsed as JSON, so numbers, booleans, lists, and objects keep their type:

```bash
python project/app.py config-set agent.total_budget 12000000
python project/app.py config-set agent.weights.water_supplies 1.35
```

## Legacy Water Forecast Script

Clean and plot the national water history without submitting to Sybilion:

```bash
python somalia_water_sybilion.py --no-submit
```

Submit a national water forecast:

```bash
python somalia_water_sybilion.py --aggregation mean --horizon 6
python somalia_water_sybilion.py --aggregation median --horizon 6
```

Poll and download an existing job:

```bash
python somalia_water_sybilion.py --job-id <job-id>
```

Other options:

```bash
python somalia_water_sybilion.py --poll-s 5 --timeout-s 900
python somalia_water_sybilion.py --out-dir project/data/somalia/water/sybilion
python somalia_water_sybilion.py --top-drivers 8
python somalia_water_sybilion.py --no-stale-extension
```

## Legacy CMB Forecast Script

Submit a Cost of Minimum Basket forecast:

```bash
python somalia_cmb_sybilion.py --horizon 6
```

Poll and download an existing job:

```bash
python somalia_cmb_sybilion.py --job-id <job-id>
```

Other options:

```bash
python somalia_cmb_sybilion.py --poll-s 5 --timeout-s 900
python somalia_cmb_sybilion.py --out-dir somalia/data/sybilion_cmb
python somalia_cmb_sybilion.py --top-drivers 8
```

## Help

Print CLI help directly from each entrypoint:

```bash
python project/app.py --help
python somalia_water_sybilion.py --help
python somalia_cmb_sybilion.py --help
```

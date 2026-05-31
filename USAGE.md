# CLI Usage

Run all commands from the repository root. In GitHub / Codespaces this is expected to be `/projects/aid-ressource-allocation-engine`.

## Setup

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Live Sybilion runs require a token in `SYBILION_API_TOKEN`, `API_KEY.txt`, or `aid-ressource-allocation-engine/API_KEY.txt`.

Live Copernicus weather refreshes require CDS credentials via `CDSAPI_KEY` / `CDSAPI_URL` or a `.cdsapirc` file.

## Main Pipeline

Default cached run:

```bash
python app.py
python app.py run --mode cache
```

This loads cached weather, water, and CMB forecasts, builds local seasonal baselines for fuel and regional agent inputs by default, and prints agent allocation decisions.

Refresh data and submit live forecasts:

```bash
python app.py run --mode refresh
```

Useful overrides:

```bash
python app.py run --data-source cache --forecast-source live
python app.py run --overwrite-weather
python app.py run --weather-horizon 12 --price-horizon 6
python app.py run --poll-s 5 --timeout-s 900
python app.py run --water-aggregation median --fuel-aggregation median
python app.py run --fuel-forecast-source local
python app.py run --top-drivers 8
```

## Agent Runs

Run the full pipeline but skip the agent layer:

```bash
python app.py run --skip-agent
```

Run one region:

```bash
python app.py run --agent-region Gedo
```

Run multiple regions:

```bash
python app.py run --agent-regions Bay Bakool Gedo
```

Set a total money budget. The app splits it across selected regions by population before allocating cargo classes:

```bash
python app.py run --agent-budget 12000000
```

Use regional local seasonal baselines for water/food forecasts:

```bash
python app.py run --agent-forecast-source local
```

Use cached or live regional forecasts:

```bash
python app.py run --agent-forecast-source cache
python app.py run --agent-forecast-source live
```

`--agent-units` is kept as a deprecated alias for `--agent-budget`.

## Config

Print the full agent config:

```bash
python app.py config-show
```

Print one section or value:

```bash
python app.py config-show agent
python app.py config-show agent.total_budget
python app.py config-show agent.weights
```

Update one value. Values are parsed as JSON, so numbers, booleans, lists, and objects keep their type:

```bash
python app.py config-set agent.total_budget 12000000
python app.py config-set agent.weights.water_supplies 1.35
```

## Help

Print CLI help directly:

```bash
python app.py --help
```

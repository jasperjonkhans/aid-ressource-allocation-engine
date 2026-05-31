# CLI Usage

Run all commands from the repository root. In GitHub / Codespaces this is expected to be `/projects/aid-ressource-allocation-engine`.

## Setup

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Live Sybilion runs require a token in `SYBILION_API_TOKEN`, `API_KEY.txt`, or `aid-ressource-allocation-engine/API_KEY.txt`.

Fresh Copernicus weather fetches require CDS credentials via `CDSAPI_KEY` / `CDSAPI_URL` or a `.cdsapirc` file.

## Main Pipeline

Default live run:

```bash
python app.py
python app.py run
```

This fetches fresh Gedo weather data from Copernicus, submits new Sybilion forecasts for all pipeline signals, and prints agent allocation decisions.

Cached run:

```bash
python app.py --mode cached
python app.py run --mode cached
```

This uses cached weather and cached national forecasts. Regional agent inputs and fuel fall back to local seasonal baselines unless a source is set explicitly.

Useful overrides:

```bash
python app.py run --mode cached
python app.py run --weather-horizon 12 --price-horizon 6
python app.py run --poll-s 5 --timeout-s 900
python app.py run --water-aggregation median --fuel-aggregation median
python app.py run --fuel-forecast-source local
python app.py run --reasoning formula
python app.py run --reasoning off
python app.py run --top-drivers 8
```

## Agent Runs

Run one region:

```bash
python app.py run --agent-region Gedo
```

Run multiple regions:

```bash
python app.py run --agent-regions "Buur Hakaba" Bakool Gedo
```

`Buur Hakaba` must be quoted because it contains a space. The water forecast is district-specific; the food forecast uses the available Bay WFP food-price proxy because the current WFP CSV has no Buur Hakaba food rows.

Set a custom total money budget. The default `100` is treated as percent allocation; any other custom budget is printed as money. The app splits the value across selected regions by population before allocating cargo classes:

```bash
python app.py run --agent-budget 12000000
```

Use regional local seasonal baselines for water/food forecasts:

```bash
python app.py run --agent-forecast-source local
```

Use cached regional forecasts:

```bash
python app.py run --agent-forecast-source cache
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
python app.py config-set agent.total_budget 100
python app.py config-set agent.weights.water_supplies 1.35
```

## Help

Print CLI help directly:

```bash
python app.py --help
```

# Rainfall Weather Shock Scenario

This scenario keeps the original cached forecast files unchanged. It loads the cached
weather forecast, copies the drought-related weather forecasts in memory, and
replaces them with a compound drought shock: collapsing rainfall, falling
humidity, and rising temperature. The agent is then run twice:

1. baseline cached weather
2. modified compound drought-shock weather

The output shows whether the allocation and the top deterministic reasons changed.

```bash
python scenarios/rainfall_weather_shock/run_rainfall_shock.py
```

For a narrated walkthrough, open:

```text
scenarios/rainfall_weather_shock/rainfall_weather_shock.ipynb
```

Useful options:

```bash
python scenarios/rainfall_weather_shock/run_rainfall_shock.py --region "Buur Hakaba"
python scenarios/rainfall_weather_shock/run_rainfall_shock.py --rainfall-end-factor 0.005
python scenarios/rainfall_weather_shock/run_rainfall_shock.py --humidity-end-factor 0.35 --temperature-increase-c 4
python scenarios/rainfall_weather_shock/run_rainfall_shock.py --budget 100
```

Generated comparison files are written to `scenarios/rainfall_weather_shock/output/`.

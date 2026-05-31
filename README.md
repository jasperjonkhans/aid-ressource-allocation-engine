# Humanitarian Aid Resource Allocation Agent

## Somalia

Somalia has suffered from prolonged conflict, resulting in weak infrastructure on top of that recurrent droughts, and heat waves plaege the region. As a result wate

Humanitarian organisations often struggle to detect regional crises early enough, which can lead to poor timing, inefficient resource allocation, and avoidable pressure on already limited aid budgets.

## The Data Edge

Somalia has a useful advantage for this approach: time-series data for food, water, and fuel prices is surprisingly available. Because food and water are sourced regionally, regional weather data can be used as an early indicator for drought pressure and future market stress.


## Idea

We combine regional and national market time series for food, water, and fuel prices with satellite-based weather data.

The resulting time series are sent to Sybilion for forecasting. A deterministic, rule-based agentic layer then turns those forecasts into interpretable resource-allocation signals. -> where and which trucks are sent to

## Case Study 

we focus on ... as they are the most effected regions, especially by drought which most of the humanitarian aid ressources go to

## Data And Sybilion

### Somali Markets

Regional markets are used for water prices and, where available, food-price pressure.

National market proxies are used for the Cost of Minimum Basket and fuel prices.

### Weather Data

Weather data comes from the Copernicus Climate Data Store, using ERA5-Land monthly averaged reanalysis.

- rainfall
- relative humidity
- average temperature

![Gedo weather Sybilion forecasts](README/img/gedo_weather_sybilion_forecasts.png)

## Usage

Run commands from the repository root. In GitHub / Codespaces this working directory is expected to be `/projects/aid-ressource-allocation-engine`.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the full live pipeline, including fresh weather data, new Sybilion forecasts, and agent allocation. This fetches Gedo weather data from Copernicus and submits new Sybilion jobs for weather, water, CMB, fuel, regional water, and regional food forecasts.

```bash
python app.py
python app.py run
```

Run from cached data and cached/local forecasts:

```bash
python app.py --mode cached
python app.py run --mode cached
```

Run the agent for selected regions or a custom budget:

```bash
python app.py run --agent-region Gedo
python app.py run --agent-regions Bay Bakool Gedo --agent-budget 12000000
```

Use local seasonal baselines for regional agent water/food inputs instead of live Sybilion forecasts:

```bash
python app.py run --agent-forecast-source local
```

Inspect and update agent configuration:

```bash
python app.py config-show
python app.py config-show agent.total_budget
python app.py config-set agent.total_budget 12000000
python app.py config-set agent.weights.water_supplies 1.35
```

See [USAGE.md](USAGE.md) for the full CLI documentation.


## Agent Layer

Funding is the bottleneck. The agent receives a constant money budget, distributes that budget across regions by population, then distributes each regional budget across aid classes by forecast pressure and delivery cost. The current default total budget is 10,000,000.

All agent constants live in `config.json`. Print the editable agent constants with short descriptions:

```bash
python app.py config-show
```

Print one constant or section with a dotted key:

```bash
python app.py config-show agent.total_budget
python app.py config-show agent.weights
```

Update one agent constant with `config-set`; values are parsed as JSON, so numbers, booleans, lists, and objects keep their type. Only `agent.*` keys can be changed from the CLI:

```bash
python app.py config-set agent.total_budget 12000000
python app.py config-set agent.weights.water_supplies 1.35
```

https://logcluster.org/sites/default/files/public/2026-03/logistics-clustersomaliaoperation-overviewnovember-2025.pdf

The agent layer receives forecasts rather than pulling data itself. For the current pipeline it gets:

- regional water-price forecasts for Bay, Bakool, and Gedo
- regional food-price forecasts for Bay, Bakool, and Gedo
- global fuel-price forecast
- Gedo weather forecasts for humidity, rainfall, and temperature

It returns a deterministic allocation decision across the aid classes humanitarian organisations typically deliver:

- water supplies
- water infrastructure equipment such as spare parts and pumps
- food supplies
- fuel supplies

Reasoning is deterministic and uses the same score components as the allocation logic. Each cargo class returns the top five ranked reasons by default. Use `--reasoning formula` to print the formulas or `--reasoning off` to suppress reasoning output. The CLI defaults to all three configured agent regions; use `--agent-regions Bay Bakool Gedo` for an explicit run or `--agent-region Gedo` for a one-region run. The `--agent-budget` value is distributed across regions by population before each region's cargo allocation is computed. `--agent-units` still works as a deprecated alias.

Current population weights:

| region | population |
| --- | ---: |
| Bay | 1,297,550 |
| Bakool | 564,958 |
| Gedo | 1,014,335 |

Current aid-class unit costs:

| aid class | base unit cost |
| --- | ---: |
| water supplies | 1.0 |
| water infrastructure | 6.0 |
| food supplies | 2.0 |
| fuel | 3.0 |

Current accessibility coefficients:

| region | accessibility |
| --- | ---: |
| Bay | 1.0 |
| Bakool | 1.0 |
| Gedo | 0.7 |

Effective unit cost is computed as `base_unit_cost / accessibility`. Gedo therefore treats all goods as `1 / 0.7` times as expensive for now.

```text
drought_index = sigmoid(
    + w_1 * temperature_slope
    - w_2 * rainfall_slope
    - w_3 * humidity_slope
)

water_supplies_score = WEIGHT_WATER_SUPPLIES * sigmoid(
    water_price_slope + water_price_level + drought_index
)

water_infrastructure_score = WEIGHT_WATER_INFRA * sigmoid(
    water_price_level + drought_index - water_price_slope
)

food_supplies_score = WEIGHT_FOOD_SUPPLIES * sigmoid(
    food_price_slope + food_price_level + drought_index
)

fuel_score = WEIGHT_FUEL * sigmoid(
    global_fuel_price_slope
    + global_fuel_price_level
    + average(water_supplies_score, water_infrastructure_score, food_supplies_score)
)

budget_pressure = cargo_scores * effective_unit_costs
budget_allocation = softmax(budget_pressure) * regional_budget
unit_allocation = budget_allocation / effective_unit_costs
```

Steep water or food increases push emergency supplies up. High but stable water stress pushes infrastructure. Fuel pressure is treated as a cross-sector multiplier because it affects transport, pumping, and distribution.

## technicalities

Food and fuel prices come from local market data published by humanitarian and regional organisations. Water demand is represented through water-price proxies, both nationally and for selected regions.

Copernicus satellite weather data is used to anticipate drought pressure by forecasting humidity, rainfall, and average temperature separately. Fuel is modelled globally because fuel price pressure tends to propagate across regions through transport and energy costs.

## data

The pipeline uses national market-price proxies together with regional weather signals. Every dataset used by the current code is listed with its CSV path.

### raw input csvs

| scope | signal | csv path | description |
| --- | --- | --- | --- |
| national / district panel | water prices | `data/somalia/water/som_water_price_2011_2022.csv` | Somalia water-price observations by `Region`, `District`, `month`, and `water_price`. The pipeline aggregates this to a national monthly water-shortage proxy using mean or median. |
| national | Cost of Minimum Basket (CMB) | `data/somalia/fsnau_cmb_total_basket_cmb_sorghum.csv` | FSNAU Total Basket CMB with red sorghum as the main cereal. The pipeline extracts monthly USD columns and averages across regions to estimate what a Somali household pays for the minimum food basket. |
| national / market panel | fuel prices | `data/somalia/wfp_food_prices_som.csv` | WFP Somalia market-price data. The pipeline filters fuel commodities such as diesel and petrol, then aggregates USD prices into a national monthly fuel-cost proxy. |
| regional / market panel | food prices | `data/somalia/wfp_food_prices_som.csv` | WFP Somalia market-price data. The regional food pipeline filters `admin1`, removes non-food and exchange-rate rows, and aggregates `usdprice` into monthly regional food-price proxies. |
| regional, Gedo | weather | `data/weather/gedo_monthly_weather_2006_2025.csv` | Monthly ERA5-Land weather features clipped to `GEDO_POLYGON`: rainfall in mm/day, average temperature in C, and relative humidity in %. Used to anticipate drought stress and future yield pressure. |

### additional raw fsnau csvs

These files are available for deeper CMB analysis, even though the current forecasting pipeline mainly uses the total-basket CSV above.

| signal | csv path | description |
| --- | --- | --- |
| essential items | `data/somalia/fsnau_cmb_esssential_items_cmb_sorghum_.csv` | Essential-item CMB components with red sorghum basis. Useful for decomposing the total basket. |
| food items | `data/somalia/fsnau_cmb_food_items_cmb_sorghum.csv` | Food-item CMB components. Useful for separating food inflation from non-food costs. |
| non-food items | `data/somalia/fsnau_cmb_non-food_items_cmb.csv` | Non-food CMB components. Useful for household cost drivers outside food. |
| exchange data | `data/somalia/fsnau_cmb_exch-data.csv` | Supporting exchange-rate data from the FSNAU CMB source files. |
| exchange rate | `data/somalia/fsnau_cmb_exch_rate.csv` | Exchange-rate series used as additional context for USD/SOS price interpretation. |
| notes | `data/somalia/fsnau_cmb_notes.csv` | Source notes and metadata exported from the FSNAU workbook. |

### derived monthly csvs

| scope | signal | csv path | description |
| --- | --- | --- | --- |
| national | water-price monthly proxy | `data/somalia/water/sybilion/somalia_water_price_monthly.csv` | Clean monthly `date`, `value`, `source` series produced from the raw water-price panel. Missing months are interpolated; stale data can be seasonally extended. |
| regional, Bay | water-price monthly proxy | `data/somalia/water/sybilion/regional/bay/bay_water_price_monthly.csv` | Clean monthly Bay water-price proxy, computed as the simple average across the Bay districts listed in `domain/regions.py`. |
| regional, Bakool | water-price monthly proxy | `data/somalia/water/sybilion/regional/bakool/bakool_water_price_monthly.csv` | Clean monthly Bakool water-price proxy, computed as the simple average across the Bakool districts listed in `domain/regions.py`. |
| regional, Gedo | water-price monthly proxy | `data/somalia/water/sybilion/regional/gedo/gedo_water_price_monthly.csv` | Clean monthly Gedo water-price proxy, computed as the simple average across the Gedo districts listed in `domain/regions.py`. |
| national | CMB monthly USD proxy | `data/somalia/sybilion_cmb/somalia_cmb_usd_monthly.csv` | Clean monthly `date`, `value`, `source` series extracted from the FSNAU CMB table and averaged across regions. |
| national | fuel monthly proxy | `data/somalia/sybilion_fuel/global_fuel_price_monthly.csv` | Clean monthly `date`, `value`, `source` series from WFP fuel rows, aggregated with the configured mean or median. |
| regional, Bay | food-price monthly proxy | `data/somalia/sybilion_regional_food/bay/bay_food_price_monthly.csv` | Clean monthly Bay food-price proxy from WFP `admin1=Bay` food rows, aggregated with the configured mean or median. |
| regional, Bakool | food-price monthly proxy | `data/somalia/sybilion_regional_food/bakool/bakool_food_price_monthly.csv` | Clean monthly Bakool food-price proxy from WFP `admin1=Bakool` food rows, aggregated with the configured mean or median. |
| regional, Gedo | food-price monthly proxy | `data/somalia/sybilion_regional_food/gedo/gedo_food_price_monthly.csv` | Clean monthly Gedo food-price proxy from WFP `admin1=Gedo` food rows, aggregated with the configured mean or median. |
| regional, Gedo | rainfall history | `data/weather/sybilion_gedo/rainfall_mm_per_day_history.csv` | Sybilion-ready monthly history for Gedo rainfall. |
| regional, Gedo | temperature history | `data/weather/sybilion_gedo/temperature_avg_c_history.csv` | Sybilion-ready monthly history for Gedo average temperature. |
| regional, Gedo | humidity history | `data/weather/sybilion_gedo/relative_humidity_pct_history.csv` | Sybilion-ready monthly history for Gedo relative humidity. |

### forecast csvs

| scope | forecast | csv path | description |
| --- | --- | --- | --- |
| national | water demand / shortage proxy | `data/somalia/water/sybilion/somalia_water_sybilion_forecast.csv` | Sybilion forecast for the national water-price proxy, including point forecasts and quantiles. |
| regional, Bay | water demand / shortage proxy | `data/somalia/water/sybilion/regional/bay/bay_water_sybilion_forecast.csv` | Regional forecast for the Bay water-price proxy, including point forecasts and quantiles. |
| regional, Bakool | water demand / shortage proxy | `data/somalia/water/sybilion/regional/bakool/bakool_water_sybilion_forecast.csv` | Regional forecast for the Bakool water-price proxy, including point forecasts and quantiles. |
| regional, Gedo | water demand / shortage proxy | `data/somalia/water/sybilion/regional/gedo/gedo_water_sybilion_forecast.csv` | Regional forecast for the Gedo water-price proxy, including point forecasts and quantiles. |
| national | food basket cost | `data/somalia/sybilion_cmb/somalia_cmb_sybilion_forecast.csv` | Sybilion forecast for national CMB in USD, including point forecasts and quantiles. |
| national | food basket cost, archived Sybilion series | `data/somalia/sybilion_cmb/somalia_cmb_0e8c76cc-c47f-4758-b86e-a86d16466796_forecast_series.csv` | Archived forecast-series export from an earlier Sybilion CMB job. Kept for comparison/debugging against the normalized current forecast CSV. |
| national | food basket forecast drivers | `data/somalia/sybilion_cmb/somalia_cmb_sybilion_drivers.csv` | Sybilion driver ranking for the CMB forecast, useful for explaining which external signals influenced the forecast. |
| national | fuel cost proxy | `data/somalia/sybilion_fuel/global_fuel_sybilion_forecast.csv` | Sybilion forecast for the aggregated fuel-price proxy, including point forecasts and quantiles. |
| regional, Bay | food-price proxy | `data/somalia/sybilion_regional_food/bay/bay_food_sybilion_forecast.csv` | Forecast for the Bay regional WFP food-price proxy, including point forecasts and quantiles. |
| regional, Bakool | food-price proxy | `data/somalia/sybilion_regional_food/bakool/bakool_food_sybilion_forecast.csv` | Forecast for the Bakool regional WFP food-price proxy, including point forecasts and quantiles. |
| regional, Gedo | food-price proxy | `data/somalia/sybilion_regional_food/gedo/gedo_food_sybilion_forecast.csv` | Forecast for the Gedo regional WFP food-price proxy, including point forecasts and quantiles. |
| regional, Gedo | combined weather forecast | `data/weather/sybilion_gedo/gedo_weather_sybilion_forecasts.csv` | Combined Sybilion weather forecasts for rainfall, temperature, and humidity. |
| regional, Gedo | rainfall forecast | `data/weather/sybilion_gedo/rainfall_mm_per_day_forecast.csv` | Forecast for Gedo rainfall in mm/day. |
| regional, Gedo | temperature forecast | `data/weather/sybilion_gedo/temperature_avg_c_forecast.csv` | Forecast for Gedo average temperature in C. |
| regional, Gedo | humidity forecast | `data/weather/sybilion_gedo/relative_humidity_pct_forecast.csv` | Forecast for Gedo relative humidity in %. |

### non-csv cached source data

| scope | file path | description |
| --- | --- | --- |
| regional, Gedo | `data/weather/era5_land_monthly_2006_2025_78511d14d4b57c13.nc` | Cached Copernicus ERA5-Land NetCDF download used to produce the Gedo weather CSV. |
| national | `data/somalia/fsnau_somalia_cmb_red_sorghum_apr_2026.xlsx` | Original FSNAU workbook from which the CMB CSV extracts were created. |

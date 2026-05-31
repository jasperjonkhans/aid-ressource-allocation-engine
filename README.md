# Humanitarian Aid Resource Allocation Agent

A decision-support engine for humanitarian aid allocation in Somalia. The system combines regional food, water, and fuel price forecasts with Copernicus weather indicators to allocate scarce aid capacity across food supplies, water supplies, fuel, and water infrastructure equipment, with transparent reasoning for every recommendation.

## Somalia

Somalia has suffered from prolonged conflict, resulting in weak infrastructure on top of that recurrent droughts, and heat waves plaege the region. As a result the somali population regularly suffers from water and food shortages often resulting in wide spread famines.

![](README/img/somalia_famine.png)


Thats where Humanitarian organisations step in, but - funding  is limited. Moreover is it often wasted by poor ressource management. Humanitarian organisations have trouble predicting where and in which quantity certain aids will be needed.


## The upside

Somalia has a useful advantage: time-series data for food, water, and fuel prices is surprisingly available. 
Because food and water are sourced regionally, regional weather data can be used as an early indicator for drought pressure and future market stress.


## Idea

Combine regional and national market time series for food, water, and fuel prices with satellite-based weather data.

The resulting time series are sent to Sybilion for forecasting. A deterministic, rule-based agentic layer then turns those forecasts into interpretable resource-allocation signals. -> where and in which quantities which ressources are allocated to

## Case Study 

we focus on the 3 districts which are most affected: Gedo, Bay/Buurhakaba and Bakool all regions which are especially hit by droughts

insert map of the districts

## Data And Sybilion

### Somali Markets

Regional markets are used for water prices and, where available, food-price pressure.

National market proxies are used for the Cost of Minimum Basket and fuel prices.

![Somalia common commodities](README/img/somalia_common_commodities.png)

### Weather Data

Weather data comes from the Copernicus Climate Data Store, using ERA5-Land monthly averaged reanalysis.

- rainfall
- relative humidity
- average temperature

![Gedo weather Sybilion forecasts](README/img/gedo_weather_sybilion_forecasts.png)

plug timeseries into sybilion API and retrieve forecasts

## Usage

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
python app.py config-set agent.total_budget 100
python app.py config-set agent.weights.water_supplies 1.35
```

See [USAGE.md](USAGE.md) for the full CLI documentation.


## Agent Layer

Funding is the bottleneck. The agent receives the funding and allocates ressources where they are needed the most. 

The agents logic comprised of a formula which you can see with.

```
python app.py --mode cashed --format formula
```

Weights and other constants are freely customizable.

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
python app.py config-set agent.total_budget 100
python app.py config-set agent.weights.water_supplies 1.35
```


## data

### input csvs

| scope | signal | csv path | description |
| --- | --- | --- | --- |
| national / district panel | water prices | `data/somalia/water/som_water_price_2011_2022.csv` | Somalia water-price observations by `Region`, `District`, `month`, and `water_price`. The pipeline aggregates this to a national monthly water-shortage proxy using mean or median. |
| national | Cost of Minimum Basket (CMB) | `data/somalia/fsnau_cmb_total_basket_cmb_sorghum.csv` | FSNAU Total Basket CMB with red sorghum as the main cereal. The pipeline extracts monthly USD columns and averages across regions to estimate what a Somali household pays for the minimum food basket. |
| national / regional market panel | fuel and food prices | `data/somalia/wfp_food_prices_som.csv` | WFP Somalia market-price data. The pipeline filters fuel commodities for the national fuel proxy and food rows by `admin1` for regional food-price proxies. |

### weather API

Weather features for rainfall, average temperature, and relative humidity are pulled from the **Copernicus Climate Data Store API**, using **ERA5-Land monthly averaged reanalysis** clipped to polygons.

### prediction series csvs

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
| national | water demand / shortage proxy | `data/somalia/water/sybilion/somalia_water_sybilion_forecast.csv` | Sybilion forecast for the national water-price proxy, including point forecasts and quantiles. |
| regional, Bay | water demand / shortage proxy | `data/somalia/water/sybilion/regional/bay/bay_water_sybilion_forecast.csv` | Regional forecast for the Bay water-price proxy, including point forecasts and quantiles. |
| regional, Bakool | water demand / shortage proxy | `data/somalia/water/sybilion/regional/bakool/bakool_water_sybilion_forecast.csv` | Regional forecast for the Bakool water-price proxy, including point forecasts and quantiles. |
| regional, Gedo | water demand / shortage proxy | `data/somalia/water/sybilion/regional/gedo/gedo_water_sybilion_forecast.csv` | Regional forecast for the Gedo water-price proxy, including point forecasts and quantiles. |
| national | food basket cost | `data/somalia/sybilion_cmb/somalia_cmb_sybilion_forecast.csv` | Sybilion forecast for national CMB in USD, including point forecasts and quantiles. |
| national | fuel cost proxy | `data/somalia/sybilion_fuel/global_fuel_sybilion_forecast.csv` | Sybilion forecast for the aggregated fuel-price proxy, including point forecasts and quantiles. |
| regional, Bay | food-price proxy | `data/somalia/sybilion_regional_food/bay/bay_food_sybilion_forecast.csv` | Forecast for the Bay regional WFP food-price proxy, including point forecasts and quantiles. |
| regional, Bakool | food-price proxy | `data/somalia/sybilion_regional_food/bakool/bakool_food_sybilion_forecast.csv` | Forecast for the Bakool regional WFP food-price proxy, including point forecasts and quantiles. |
| regional, Gedo | food-price proxy | `data/somalia/sybilion_regional_food/gedo/gedo_food_sybilion_forecast.csv` | Forecast for the Gedo regional WFP food-price proxy, including point forecasts and quantiles. |

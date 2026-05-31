# Humanitarian Aid Resource Allocation Agent

## Somalia

Somalia has suffered from prolonged conflict, weak infrastructure, recurrent droughts, and heat waves. These conditions create recurring food and water shortages and make humanitarian aid planning especially difficult.

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

![Gedo weather Sybilion forecasts](./img/gedo_weather_sybilion_forecasts.png)


## Agent Layer

The agent layer receives forecasts rather than pulling data itself. For the current pipeline it gets:

- regional water-price forecast
- regional food-price forecast
- global fuel-price forecast
- Gedo weather forecasts for humidity, rainfall, and temperature

It returns a deterministic allocation decision across the aid classes humanitarian organisations typically deliver:

- water supplies
- water infrastructure equipment such as spare parts and pumps
- food supplies
- fuel supplies

Reason text is intentionally disabled for now; the current return object keeps `reasoning=None` so explanations can be added later without changing the interface.

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

allocation = softmax(cargo_scores)
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
| national / district panel | water prices | `project/data/somalia/water/som_water_price_2011_2022.csv` | Somalia water-price observations by `Region`, `District`, `month`, and `water_price`. The pipeline aggregates this to a national monthly water-shortage proxy using mean or median. |
| national | Cost of Minimum Basket (CMB) | `somalia/data/fsnau_cmb_total_basket_cmb_sorghum.csv` | FSNAU Total Basket CMB with red sorghum as the main cereal. The pipeline extracts monthly USD columns and averages across regions to estimate what a Somali household pays for the minimum food basket. |
| national / market panel | fuel prices | `somalia/data/wfp_food_prices_som.csv` | WFP Somalia market-price data. The pipeline filters fuel commodities such as diesel and petrol, then aggregates USD prices into a national monthly fuel-cost proxy. |
| regional / market panel | food prices | `somalia/data/wfp_food_prices_som.csv` | WFP Somalia market-price data. The regional food pipeline filters `admin1`, removes non-food and exchange-rate rows, and aggregates `usdprice` into monthly regional food-price proxies. |
| regional, Gedo | weather | `project/data/weather/gedo_monthly_weather_2006_2025.csv` | Monthly ERA5-Land weather features clipped to `GEDO_POLYGON`: rainfall in mm/day, average temperature in C, and relative humidity in %. Used to anticipate drought stress and future yield pressure. |

### additional raw fsnau csvs

These files are available for deeper CMB analysis, even though the current forecasting pipeline mainly uses the total-basket CSV above.

| signal | csv path | description |
| --- | --- | --- |
| essential items | `somalia/data/fsnau_cmb_esssential_items_cmb_sorghum_.csv` | Essential-item CMB components with red sorghum basis. Useful for decomposing the total basket. |
| food items | `somalia/data/fsnau_cmb_food_items_cmb_sorghum.csv` | Food-item CMB components. Useful for separating food inflation from non-food costs. |
| non-food items | `somalia/data/fsnau_cmb_non-food_items_cmb.csv` | Non-food CMB components. Useful for household cost drivers outside food. |
| exchange data | `somalia/data/fsnau_cmb_exch-data.csv` | Supporting exchange-rate data from the FSNAU CMB source files. |
| exchange rate | `somalia/data/fsnau_cmb_exch_rate.csv` | Exchange-rate series used as additional context for USD/SOS price interpretation. |
| notes | `somalia/data/fsnau_cmb_notes.csv` | Source notes and metadata exported from the FSNAU workbook. |

### derived monthly csvs

| scope | signal | csv path | description |
| --- | --- | --- | --- |
| national | water-price monthly proxy | `project/data/somalia/water/sybilion/somalia_water_price_monthly.csv` | Clean monthly `date`, `value`, `source` series produced from the raw water-price panel. Missing months are interpolated; stale data can be seasonally extended. |
| regional, Bay | water-price monthly proxy | `project/data/somalia/water/sybilion/regional/bay/bay_water_price_monthly.csv` | Clean monthly Bay water-price proxy, computed as the simple average across the Bay districts listed in `project/domain/regions.py`. |
| regional, Bakool | water-price monthly proxy | `project/data/somalia/water/sybilion/regional/bakool/bakool_water_price_monthly.csv` | Clean monthly Bakool water-price proxy, computed as the simple average across the Bakool districts listed in `project/domain/regions.py`. |
| regional, Gedo | water-price monthly proxy | `project/data/somalia/water/sybilion/regional/gedo/gedo_water_price_monthly.csv` | Clean monthly Gedo water-price proxy, computed as the simple average across the Gedo districts listed in `project/domain/regions.py`. |
| national | CMB monthly USD proxy | `somalia/data/sybilion_cmb/somalia_cmb_usd_monthly.csv` | Clean monthly `date`, `value`, `source` series extracted from the FSNAU CMB table and averaged across regions. |
| national | fuel monthly proxy | `somalia/data/sybilion_fuel/global_fuel_price_monthly.csv` | Clean monthly `date`, `value`, `source` series from WFP fuel rows, aggregated with the configured mean or median. |
| regional, Gedo | food-price monthly proxy | `somalia/data/sybilion_regional_food/gedo/gedo_food_price_monthly.csv` | Clean monthly Gedo food-price proxy from WFP `admin1=Gedo` food rows, aggregated with the configured mean or median. The same function can build all available WFP `admin1` regions. |
| regional, Gedo | rainfall history | `project/data/weather/sybilion_gedo/rainfall_mm_per_day_history.csv` | Sybilion-ready monthly history for Gedo rainfall. |
| regional, Gedo | temperature history | `project/data/weather/sybilion_gedo/temperature_avg_c_history.csv` | Sybilion-ready monthly history for Gedo average temperature. |
| regional, Gedo | humidity history | `project/data/weather/sybilion_gedo/relative_humidity_pct_history.csv` | Sybilion-ready monthly history for Gedo relative humidity. |

### forecast csvs

| scope | forecast | csv path | description |
| --- | --- | --- | --- |
| national | water demand / shortage proxy | `project/data/somalia/water/sybilion/somalia_water_sybilion_forecast.csv` | Sybilion forecast for the national water-price proxy, including point forecasts and quantiles. |
| regional, Bay | water demand / shortage proxy | `project/data/somalia/water/sybilion/regional/bay/bay_water_sybilion_forecast.csv` | Regional forecast for the Bay water-price proxy, including point forecasts and quantiles. |
| regional, Bakool | water demand / shortage proxy | `project/data/somalia/water/sybilion/regional/bakool/bakool_water_sybilion_forecast.csv` | Regional forecast for the Bakool water-price proxy, including point forecasts and quantiles. |
| regional, Gedo | water demand / shortage proxy | `project/data/somalia/water/sybilion/regional/gedo/gedo_water_sybilion_forecast.csv` | Regional forecast for the Gedo water-price proxy, including point forecasts and quantiles. |
| national | food basket cost | `somalia/data/sybilion_cmb/somalia_cmb_sybilion_forecast.csv` | Sybilion forecast for national CMB in USD, including point forecasts and quantiles. |
| national | food basket cost, archived Sybilion series | `somalia/data/sybilion_cmb/somalia_cmb_0e8c76cc-c47f-4758-b86e-a86d16466796_forecast_series.csv` | Archived forecast-series export from an earlier Sybilion CMB job. Kept for comparison/debugging against the normalized current forecast CSV. |
| national | food basket forecast drivers | `somalia/data/sybilion_cmb/somalia_cmb_sybilion_drivers.csv` | Sybilion driver ranking for the CMB forecast, useful for explaining which external signals influenced the forecast. |
| national | fuel cost proxy | `somalia/data/sybilion_fuel/global_fuel_sybilion_forecast.csv` | Forecast for the aggregated fuel-price proxy. Defaults to a local seasonal baseline unless live Sybilion fuel forecasts are requested. |
| regional, Gedo | food-price proxy | `somalia/data/sybilion_regional_food/gedo/gedo_food_sybilion_forecast.csv` | Forecast for the Gedo regional WFP food-price proxy, including point forecasts and quantiles. The same function can produce forecasts for all available WFP `admin1` regions. |
| regional, Gedo | combined weather forecast | `project/data/weather/sybilion_gedo/gedo_weather_sybilion_forecasts.csv` | Combined Sybilion weather forecasts for rainfall, temperature, and humidity. |
| regional, Gedo | rainfall forecast | `project/data/weather/sybilion_gedo/rainfall_mm_per_day_forecast.csv` | Forecast for Gedo rainfall in mm/day. |
| regional, Gedo | temperature forecast | `project/data/weather/sybilion_gedo/temperature_avg_c_forecast.csv` | Forecast for Gedo average temperature in C. |
| regional, Gedo | humidity forecast | `project/data/weather/sybilion_gedo/relative_humidity_pct_forecast.csv` | Forecast for Gedo relative humidity in %. |

### non-csv cached source data

| scope | file path | description |
| --- | --- | --- |
| regional, Gedo | `project/data/weather/era5_land_monthly_2006_2025_78511d14d4b57c13.nc` | Cached Copernicus ERA5-Land NetCDF download used to produce the Gedo weather CSV. |
| national | `somalia/data/fsnau_somalia_cmb_red_sorghum_apr_2026.xlsx` | Original FSNAU workbook from which the CMB CSV extracts were created. |

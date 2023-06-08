# Current Status

## Data aggregation

### Data procurement

#### PV data
- The solar panel production is from a solar site in Bastorf.
- The data is in 15 min windows.
- Data for roughly 2 years.

#### Forecast data
- The forecasts are from ECMWF 
- The forecasts are generated every 12 hours at 12 am and pm.
- The forecasts are valid for 36 hours.
- In addition to this data from ECMWF we use clear sky irradiance which is
caclculated by a physical model and is a theoretical value.

#### Measurement data
- The measurements are from Rostock which is around 25 km away. 
- The data is from dwd.

### Data preprocessing

#### PV data
- The PV data is used to create features
- We create periodic features, with a period of e.g. 24 hours, 7 days, 31 days.
- These periodic features can be e.g. max values or mean.
- We create features that represent a shift of different time frames, e.g. 1H
2H, 6H, 12H, 24H.
- These shifts have different features, e.g. max, mean
- These features allows us to represent the relation between the current and 
the past while still using a feed forward neural net.

#### Forecast data
- The features are forecasted hourly, so we resample to 15 min windows.


#### Measurement data
- The measurements are in 10 min windows and then resampled to 15 mins.

## Models and Training
- As models we used XGBoost and a feed forward neural net. 
- We compared the performance of the models and we trained the models with 
different features.
- One XGBoost model is trained only with historical data
- One XGBoost model is trained with historical data and with the forecasts
- One XGBoost model is trained with historical data and measurement
- The neural net is trained with historical data and forecasts
- We trained one model with measurements to see the ceiling of the forecast 
feature as a perfect forecast would equal the measuremants.
- We also filtered out values under 7kW, because it is easy to forecast e.g. 
the night and we are interested in forecasting the production during the day 
when it matters.

## Results so far
- All models, no matter the features follow the trend that forecasting becomes
less accurate the further we forecast into the future.
- For the first hour the difference between all models is quite small.
- At the 2 hour mark the forecast and measurements start to increase the 
performance of the models
- Betweeen the 4h and 24h mark, the performance seems quite stable
- Even with the measurements, the predicted production is still off
- This indicates, that just the radiation and past production is not enough
to make a very accurate prediction

## Analyzing the features
- We also compared the features with the target production
- Generally, if the forecast is off, it predicts usually more production than
what is actually occuring
- ssrd and ssr values are very similar
- When the weather forecast is off, it usually also biases towards higher 
values


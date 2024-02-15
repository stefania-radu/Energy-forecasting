# Energy Consumption Forecasting Pipeline

![intro](analysis/intro.png)

## Setup

Create a virtual environment using venv: `python -m venv simpl_env` and activate it.

Install the requirements: `pip install -r requirements.txt`

Python version: 3.10 \
Forecasting libraries: nixtla, sklearn\
API: FastAPI, uvicorn


## Quick Start

Run the API: `uvicorn api:app --reload`

Send a request to the endpoint with the .json file. The result contains the forecasts for the future data using the model I found to work best (Linear regression):
`python demo.py`

    {"forecast":{"0":3922056.7903033737,"1":3889187.233931119.......}


![Future Forecasts](results\future_forecasts.png)

## End to End Walkthrough


### Analysing the data

The data shows the energy consumption over a period of approx 3 months. The recordings are done every 15 min and there are 9440 timestamps. 
That means there are 96 recordings a day and 96*7 recording per week.
We can already see some seasonality with respect to the day and week. Less power is used during the weekend than during the week days.

![Analysis Image](analysis\ed010d2a-1530-4bfa-b8a2-463aeb224c25.png)

There are 3 exogenous variables:
- shortwave_radiation
- temperature_2m
- windspeed_10m

![Exogenous Data](analysis\exog_data.png)

I performed 2 statistical tests with these variables, first to check their correlation with the response (y) and then to check for multicollinearity.

#### Pearson correlation test

Correlation between 'y' and shortwave_radiation: 0.4345828467580596\
Correlation between 'y' and temperature_2m: 0.17930982066296647\
Correlation between 'y' and windspeed_10m: 0.13608799120468934

All variables show positive correlation, but the highest one is shortwave_radiation, so I expect it to improve prediction the most. 
However, other variables might have a non-linear relationship with y, which is not tested here. An interaction test/plot might also be helpful.

#### Testing for multicollinearity:

|    variable            |     VIF     |
|------------------------|-------------|
|         const          |  19.953837  |
| shortwave_radiation    |  1.469424   |
|    temperature_2m      |  1.561450   |
|     windspeed_10m      |  1.076232   |

All the Variance Inflation Factor (VIF) values are quite small so the exogenous variables are not correlated.

#### Seasonality:

I performed decomposition on the signal to check for any seasonality and as expected, the results show that there is daily and weekly seasonality:

![Decomposed Image](analysis\decomposed.png)


### Data loading/preprocessing

Loading the data is done in the [DataLoader](src\DataLoader.py) class, where I used dataframes to extract history and future data, as well as any necessary parameters.
As some values are missing, I used a backword fill method to complete the dataset, since it gave the best results.
For testing purposes, I split the history data into a training (90%) and a validation dataset (10%). A much more reliable appraoch would have been to use 
cross validation.

History Data:  
|    ds    |       y       |              unique_id             | shortwave_radiation | temperature_2m | windspeed_10m |
|----------|---------------|------------------------------------|---------------------|----------------|---------------|
|2023-08-09|   5288480.0   |ed010d2a-1530-4bfa-b8a2-463aeb224c25|     459.653691      |   20.374219   |   4.978553    |
|2023-08-09|   5228632.0   |ed010d2a-1530-4bfa-b8a2-463aeb224c25|     455.728915      |   20.343750   |   4.950307    |
|2023-08-09|   5058448.0   |ed010d2a-1530-4bfa-b8a2-463aeb224c25|     448.189682      |   20.266406   |   4.884408    |
|2023-08-09|   5036708.0   |ed010d2a-1530-4bfa-b8a2-463aeb224c25|     436.000000      |   20.100000   |   4.790000    |
|2023-08-09|   4844940.0   |ed010d2a-1530-4bfa-b8a2-463aeb224c25|     403.148763      |   19.875000   |   4.677387    |

### Forecasting experiments

I used the nixtla library for all my experiment. The goal was to compare some statistical models with machine learning models, as well as looking at the 
effect of exogenous variables. The main class for the experiments is [ForecastExperiment](src\ForecastExperiment.py).

- Statistical models without exogenous variables

    The statistical models I used are: 
    - HistoricAverage()
    - SeasonalNaive()
    - DynamicOptimizedTheta()

    Here, whenever one seasonality was expected, I used the week, as it achieved the best results. No exogenous variables are used in this experiment.

- Multiple Seasonal-Trend decomposition with exogenous variables

    To test the effect of exogenous variables, I used Multiple Seasonal-Trend decomposition using LOESS (MSTD) with the AutoARIMA forecaster. This model allows
    for multiple seaonalities so I used both the day and the week here.

- Machine learning models with exogenous variables

    The machine learning models I used are:
    - LinearRegression()
    - MLPRegressor()

    I also used day and week as features using the lags parameteres, as well as target transformations.

**Run all experiments: `python run_experiments.py`**


### Evaluation

I evaluated my models on the validation dataset using 4 loss functions:
- Mean Absolute Error
- Mean Squared Error
- Mean Absolute Percentage Error (MAPE)
- Symmetric Mean Absolute Percentage Error (SMAPE)

I also computed the CPU time for fitting and predicting the models.


### Results

- Statistical models without exogenous variables

    ![Stats Models Results](results\stats_models.png)

- Multiple Seasonal-Trend decomposition with exogenous variables

    - shortwave radiation

    ![Exog Shortwave Radiation](results\forecasts_exog_shortwave_radiation.png)

    - temperature 2m

    ![Exog Temperature 2m](results\forecasts_exog_temperature_2m.png)

    - windspeed 10m

    ![Exog Windspeed 10m](results\forecasts_exog_windspeed_10m.png)

    - all variables

    ![Exog All Variables](results\forecasts_exog_all.png)


- Machine learning models with exogenous variables

    - shortwave radiation

    ![ML Shortwave Radiation](results\forecasts_ML_shortwave_radiation.png)

    - temperature 2m

    ![ML Temperature 2m](results\forecasts_ML_temperature_2m.png)

    - windspeed 10m

    ![ML Windspeed 10m](results\forecasts_ML_windspeed_10m.png)

    - all variables

    ![ML All Variables](results\forecasts_ML_all.png)


CPU time vs loss for some models:

|       Model               | Mean Absolute Error | Train Time (s) | Predict Time (s) |
|---------------------------|---------------------|----------------|------------------|
| SeasonalNaive             |      184908.69      |    3.640625    |     0.046875     |
| MSTL (all)                |      481275.99      |    28.000000   |     0.031250     |
| MLPRegressor  (all)       |      187603.85      |    1.234375    |     1.640625     |
| LinearRegression (all)    |      203105.34      |    0.000000    |     1.578125     |


According to the logs the best model is:

| Model                      | mean_absolute_error | mean_squared_error | mape | smape |
|----------------------------|---------------------|--------------------|------|-------|
| LinearRegression (weather) |      179766.14      |   6.783439e+10    | 3.61 | 3.66  |
| SeasonalNaive              |      184908.69      |   6.399835e+10    | 3.65 | 3.75  |
| MLPRegressor (all)         |      187603.85      |   7.359905e+10    | 3.88 | 3.80  |


### Predicting the test data

**Run prediction on test data: `python run_best_forecast.py`**

![Future Forecasts](results\future_forecasts.png)



import logging
import time
import pandas as pd
from src.utils import *

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

from statsforecast import StatsForecast
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from mlforecast.target_transforms import Differences
from statsforecast.models import DynamicOptimizedTheta, SeasonalNaive, MSTL, AutoARIMA, HistoricAverage
from utilsforecast.plotting import plot_series


logger = logging.getLogger(__name__)


class ForecastExperiment:
    """
    Main class for forecasting experiments. A series of predefined models are initalized for each experiment type:
    - stats_models: HistoricAverage(), SeasonalNaive(), DynamicOptimizedTheta()
    - exog_experiment: MSTL with AutoARIMA
    - ML_models: LinearRegression(),MLPRegressor()
    """
    def __init__(self, experiment_name, experiment_type='stats_models', freq='15min', season_length=[96], n_jobs=-1):
        self.experiment_name = experiment_name
        self.experiment_type = experiment_type
        self.freq = freq
        self.season_length = season_length
        self.n_jobs = n_jobs
        self.models = None
        self.sff = None
        self.forecasts_df = None
        self.train_cpu_time = None
        self.predict_cpu_time = None
        self.losses = None

        logger.info(f"=========Initializing experiment: {self.experiment_name} ===========")

        if experiment_type == "stats_models":
            self.models = [
                HistoricAverage(), 
                SeasonalNaive(season_length=self.season_length['WEEK']),
                DynamicOptimizedTheta(season_length=self.season_length['WEEK']),
            ]
            
        elif experiment_type == "exog_experiment":
            self.models = [
                MSTL(
                season_length=[self.season_length['DAY'], self.season_length['WEEK']], # seasonalities of the time series 
                trend_forecaster=AutoARIMA() # model used to forecast trend
                )
            ]

        elif experiment_type == "ML_models":
            self.models=[
                LinearRegression(),
                MLPRegressor(hidden_layer_sizes=256, batch_size=32, random_state=1)
            ]

        elif experiment_type == "BEST":
                self.models=[
                LinearRegression(),
            ]

        else:
            logger.info("Experiment not implemented")


    def run_pipeline(self, parameters, complete_data, history_data_train, history_data_validation,  exog_var_to_use=[], save_file_name="out"):
        """
        Executes the complete pipeline: running experiment and forecasting, saving the results, and evaluation
        """
        self.forecasts_df =  self.run_experiment(parameters, history_data_train, history_data_validation, exog_var_to_use)
        self.save_results(complete_data, save_file_name)
        self.evaluate_forecast(history_data_validation)


    def run_experiment(self, parameters, history_data, future_data=[], exog_var_to_use=[]):
        """
        Run a forecasting experiment based on the experiment type. The forecaster predicts new values based on some history data and past (or future) exogenous variables
        """
        
        logger.info(f"=========Running Experiment {self.experiment_name} with type={self.experiment_type}===========")
        logger.info(f"=========Models: {self.models} ===========")

        if self.experiment_type in ["stats_models", "exog_experiment"]:
            history_data = history_data.copy()
            history_data['ds'] = pd.to_datetime(history_data['ds']).dt.strftime('%Y-%m-%d %H:%M:%S')
            future_data = future_data.copy()
            future_data['ds'] = pd.to_datetime(future_data['ds'])
            
            self.sff = StatsForecast(
                models=self.models,
                freq=self.freq,
                n_jobs=self.n_jobs
            )

        elif self.experiment_type in ["ML_models", "BEST"]:

            history_data = history_data.copy()
            history_data['ds'] = pd.to_datetime(history_data['ds'])
            
            self.sff = MLForecast(
                models=self.models,
                freq=self.freq,
                lags=[self.season_length['DAY'], self.season_length['WEEK']],
                target_transforms=[Differences([self.season_length['DAY']])]
            )
            

        self.horizon = parameters["horizon"] if not len(future_data) else len(future_data)
        levels = parameters["levels"]
        
        self.forecasts_df = self._do_forecast(history_data, future_data, self.horizon, levels, exog_var_to_use)

        logger.info('==========Done forecasting===========')
        logger.info(self.forecasts_df.head())
        
        return self.forecasts_df


    def save_results(self, history_data, fig_name):
        """
        Save the results of the forecaster as a plot and .csv file
        """
    
        results_dir = "results"

        if self.experiment_type in ["BEST","ML_models"]:
            history_data = history_data.copy()
            history_data['ds'] = pd.to_datetime(history_data['ds'])
            plot_series(history_data[-96*10:], self.forecasts_df[-96*10:]).savefig(f"{results_dir}/{fig_name}.png")
        else:  
            self.sff.plot(history_data[-96*10:], self.forecasts_df[-96*10:]).savefig(f"{results_dir}/{fig_name}.png")
        self.forecasts_df.to_csv(f'{results_dir}/{fig_name}_forecasts.csv', index=False)


    def evaluate_forecast(self, history_data_validation):
        """
        Evaluate the forecasting based on a validation set which contains the true values. The losses computed are MAE, MSE, MAPE, SMAPE
        """
    
        logger.info("========Evaluating models:=============")

        metrics = [mean_absolute_error, mean_squared_error, mape, smape]

        history_data_validation_Y = history_data_validation[['unique_id','ds','y']]
        history_data_validation_Y = history_data_validation.copy()
        history_data_validation_Y['ds'] = pd.to_datetime(history_data_validation_Y['ds'])
        results = history_data_validation_Y.merge(self.forecasts_df, how='left', on=['unique_id', 'ds'])

        logger.info(f"RESULTS {results}")

        metric_values = {metric.__name__: [] for metric in metrics}
        for model in results.columns[6:]:  # skip the ds, y and exogenous variables columns
            y_true = results['y']
            y_pred = results[model]
            for metric in metrics:
                metric_value = metric(y_true, y_pred)
                metric_value = round(metric_value, 2)
                metric_values[metric.__name__].append(metric_value)

        self.losses = pd.DataFrame(metric_values, index=results.columns[6:])

        logger.info(self.losses)

        logger.info("*************************************")
        logger.info(f"train_cpu_time: {self.train_cpu_time}")
        logger.info(f"predict_cpu_time: {self.predict_cpu_time}")
        logger.info("*************************************")
        
        return self.losses
    

    def _do_forecast(self, history_data, future_data, horizon, levels, exog_var_to_use):
        """
        Perform forecasting (fit and predict) based on the history data, horizon, levels and a list of optional exogenous variables
        """

        if len(exog_var_to_use) > 0:
            
            logger.info(f"Using exogenous variables: {exog_var_to_use}")
        else:
            logger.info(f"NO exogenous variables used.")

        history_data_subset = history_data[['ds', 'unique_id', 'y'] + exog_var_to_use]
        future_data_subset = future_data[['ds', 'unique_id'] + exog_var_to_use]

        # fit and predict based on the experiment type
        if self.experiment_type in ["BEST", "ML_models"]:
            _, self.train_cpu_time = self._compute_cpu_time(self.sff.fit, df=history_data_subset, static_features=[], prediction_intervals=PredictionIntervals(n_windows=5, h=horizon, method="conformal_distribution" ))
            forecasts_df, self.predict_cpu_time = self._compute_cpu_time(self.sff.predict, h=horizon, level=levels, X_df=future_data_subset)

        elif self.experiment_type  == "exog_experiment":
            _, self.train_cpu_time = self._compute_cpu_time(self.sff.fit, df=history_data_subset)
            forecasts_df, self.predict_cpu_time = self._compute_cpu_time(self.sff.predict, h=horizon, level=levels, X_df=future_data_subset)
            # forecasts_df = self.sff.forecast(df=history_data_subset, h=horizon, level=levels, X_df=future_data_subset)
            
        else:
            _, self.train_cpu_time = self._compute_cpu_time(self.sff.fit, df=history_data)
            forecasts_df, self.predict_cpu_time = self._compute_cpu_time(self.sff.predict, h=horizon, level=levels)
            # forecasts_df = self.sff.forecast(df=history_data, h=horizon, level=levels)

        return forecasts_df


    def _compute_cpu_time(self, func, *args, **kwargs):
        """
        Compute the CPU time when executing a function
        """

        start_time = time.process_time()
        result = func(*args, **kwargs)  # Execute the function with provided arguments
        end_time = time.process_time()
        cpu_time = end_time - start_time

        return result, cpu_time

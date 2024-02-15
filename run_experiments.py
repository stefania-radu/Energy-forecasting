import os
import logging
from src.DataLoader import DataLoader
from src.ForecastExperiment import ForecastExperiment
from src.utils import *


results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


logging.basicConfig(filename='results/output.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


os.environ['NIXTLA_ID_AS_COL'] = '1'


SEASON_LENGTH = {'DAY': 96,
                  'WEEK': 96 * 7}



def main():

    file_path = 'request_body.json' 
    data_loader = DataLoader(file_path)
    data_loader.load_data()

    # Accessing the data
    history_data = data_loader.history_data
    history_data_train = data_loader.history_data_train
    history_data_validation = data_loader.history_data_validation
    
    parameters = data_loader.parameters
    exog_variables = data_loader.exog_variables

    # Log some information
    logger.info(f"Parameters: {parameters}")
    logger.info(f"History Data: {history_data}")
    logger.info(f"History Data Train: {history_data_train}")
    logger.info(f"History Data Validation: {history_data_validation}")
    logger.info(f"Exogenous Variables: {exog_variables}")

    # Test for correlation and multicullinearity
    test_correlation_exog_vars(history_data, exog_variables)
    test_multicollinearity(history_data, exog_variables)


    # Statistics models from StatsForecast: HistoricAverage(), SeasonalNaive(), DynamicOptimizedTheta(),
    
    stats_experiment = ForecastExperiment("Statistics experiment", experiment_type="stats_models", freq=parameters['frequency'], season_length=SEASON_LENGTH)

    stats_experiment.run_pipeline(parameters=parameters, 
                               complete_data=history_data, 
                               history_data_train=history_data_train, 
                               history_data_validation=history_data_validation,
                               save_file_name="stats_models")

   
    # Experiment with exogenous variables using MSTL and ARIMA

    exog_experiment = ForecastExperiment("Statistical models + Exogenous Variables Experiment", experiment_type="exog_experiment", freq=parameters['frequency'], season_length=SEASON_LENGTH)

    exog_experiment.run_pipeline(parameters=parameters, 
                               complete_data=history_data, 
                               history_data_train=history_data_train, 
                               history_data_validation=history_data_validation,
                               exog_var_to_use=["shortwave_radiation"],
                               save_file_name="forecasts_exog_shortwave_radiation")

    exog_experiment.run_pipeline(parameters=parameters, 
                               complete_data=history_data, 
                               history_data_train=history_data_train, 
                               history_data_validation=history_data_validation, 
                               exog_var_to_use=["windspeed_10m"],
                               save_file_name="forecasts_exog_windspeed_10m")    

    exog_experiment.run_pipeline(parameters=parameters, 
                               complete_data=history_data, 
                               history_data_train=history_data_train, 
                               history_data_validation=history_data_validation, 
                               exog_var_to_use=["temperature_2m"],
                               save_file_name="forecasts_exog_temperature_2m")

    exog_experiment.run_pipeline(parameters=parameters, 
                               complete_data=history_data, 
                               history_data_train=history_data_train, 
                               history_data_validation=history_data_validation,
                               exog_var_to_use=["shortwave_radiation", "windspeed_10m", "temperature_2m"],
                               save_file_name="forecasts_exog_all")


    # Experiment with Machine learning models and exogenous variables using MLForecast: LinearRegression(),MLPRegressor()

    ML_experiment = ForecastExperiment("ML models Experiment", experiment_type="ML_models", freq=parameters['frequency'], season_length=SEASON_LENGTH)

    ML_experiment.run_pipeline(parameters=parameters, 
                               complete_data=history_data, 
                               history_data_train=history_data_train, 
                               history_data_validation=history_data_validation,
                               exog_var_to_use=["shortwave_radiation"],
                               save_file_name="forecasts_ML_shortwave_radiation")

    ML_experiment.run_pipeline(parameters=parameters, 
                               complete_data=history_data, 
                               history_data_train=history_data_train, 
                               history_data_validation=history_data_validation, 
                               exog_var_to_use=["windspeed_10m"],
                               save_file_name="forecasts_ML_windspeed_10m")

    ML_experiment.run_pipeline(parameters=parameters, 
                               complete_data=history_data, 
                               history_data_train=history_data_train, 
                               history_data_validation=history_data_validation, 
                               exog_var_to_use=["temperature_2m"],
                               save_file_name="forecasts_ML_temperature_2m")

    ML_experiment.run_pipeline(parameters=parameters, 
                               complete_data=history_data, 
                               history_data_train=history_data_train, 
                               history_data_validation=history_data_validation,
                               exog_var_to_use=["shortwave_radiation", "windspeed_10m", "temperature_2m"],
                               save_file_name="forecasts_ML_all")

    # experiments = [stats_experiment, exog_experiment, ML_experiment]
    # table = create_performance_table(experiments)
    # plot_error_vs_time(table)


if __name__ == '__main__':
    main()
    
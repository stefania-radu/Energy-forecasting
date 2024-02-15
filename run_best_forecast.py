import os
import logging
from src.DataLoader import DataLoader
from src.ForecastExperiment import ForecastExperiment
from src.utils import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    future_data = data_loader.future_data
    parameters = data_loader.parameters

    # Run the best experiment - best not to hardcode it
    LR_experiment = ForecastExperiment("BEST Experiment", experiment_type="BEST", freq=parameters['frequency'], season_length=SEASON_LENGTH)

    forecasts = LR_experiment.run_experiment(parameters=parameters,  
                            history_data=history_data, 
                            future_data=future_data, 
                            exog_var_to_use=["temperature_2m"])

    print(forecasts['LinearRegression'])

    LR_experiment.save_results(history_data=history_data, fig_name="future_forecasts")
    

if __name__ == '__main__':
    main()
    
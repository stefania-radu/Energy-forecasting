import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from src.DataLoader import DataLoader
from src.ForecastExperiment import ForecastExperiment


app = FastAPI()


SEASON_LENGTH = {'DAY': 96, 'WEEK': 96 * 7}


@app.post("/forecast/")
async def generate_forecast(file: UploadFile = File(...)):
    
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
        content = await file.read()  # Read file content
        tmp_file.write(content)  # Write to the temporary file
        temp_file_path = tmp_file.name  # Get the temporary file path

    # Process the JSON data using the temporary file path
    data_loader = DataLoader(temp_file_path)
    data_loader.load_data()

    history_data = data_loader.history_data
    future_data = data_loader.future_data
    parameters = data_loader.parameters

    LR_experiment = ForecastExperiment("BEST Experiment", experiment_type="BEST", freq=parameters['frequency'], season_length=SEASON_LENGTH)
    forecasts = LR_experiment.run_experiment(parameters=parameters,  
                                             history_data=history_data, 
                                             future_data=future_data, 
                                             exog_var_to_use=["temperature_2m"])

    # Clean up the temporary file
    os.unlink(temp_file_path)

    return {"forecast": forecasts['LinearRegression']}

# To run the server:
# uvicorn api:app --reload

import json
import pandas as pd

from fastapi import FastAPI, UploadFile
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

from app.dependencies import binary_to_dataframe, group_dataframe
from app.models import HoltWinters

app = FastAPI()

@app.get("/")
async def root():
    return {"message":"Upload files for prediction."}

@app.post("/uploadfile/predict_{aggregation_freq}")
async def create_upload_file(file: UploadFile, aggregation_freq: str, num_of_predictions: int):

    # Reading JSON file as a binary string.
    binary_string = await file.read()

    # Transforming binary to dataframe with date | data columns.
    time_series = binary_to_dataframe(binary_string)

    time_series_grouped = group_dataframe(time_series, aggregation_freq)
    
    model = HoltWinters(time_series_grouped, aggregation_freq)
    pred = model.get_predictions(num_of_predictions)
    output = pred.to_dict()
    output['metrics'] = model.metrics

    return output
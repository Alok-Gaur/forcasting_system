from fastapi import FastAPI, UploadFile, File, HTTPException, status
from schema import PredictionInput
import pandas as pd
from model_prediction import best_model_prediction
from model_utils import load_latest_data, load_latest_encoding, get_best_model
from datetime import timedelta, datetime
from train.processing import evaluate_best_model, train_on_entire_data
import os
from pathlib import Path
from train.data_processing_utils import training_data_processing

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    FOLDERS = [
        "version_data",
        "encodings",
        "models/ARIMA",
        "models/XGBoost",
        "models/Prophet"
    ]
    for folder in FOLDERS:
        Path(folder).mkdir(parents=True, exist_ok=True)



@app.post("/forecast", tags=["User's API"])
def forecast(request:PredictionInput):
    state = request.state
    weeks = int(request.weeks)

    # Load the latest data of selected state
    # df = load_latest_data()

    # load label encoder
    data_versions_list = pd.read_json("Data_Version_History.json")
    l_date = pd.to_datetime(data_versions_list['l_date'].iloc[-1], format='mixed', dayfirst=True) + pd.Timedelta(weeks=1)
    le = load_latest_encoding()
    encoded_state = le.transform([state])[0]

    output = best_model_prediction(encoded_state, weeks, l_date)


    
    response = {'state':state, 'forecast_len': weeks, 'forecast':{
        str(l_date + timedelta(weeks=week)): float(total) for week, total in enumerate(output)
    }}
    return response



@app.post("/upload-data", tags=["Admin API's"])
async def upload_data(file: UploadFile = File(...)):
    allowed_extensions = [".csv", ".xlsx", ".xls"]

    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV and Excel files are allowed"
        )

    data = await file.read()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_filename = f"{file.filename}_{timestamp}{file_ext}"

    save_path = os.path.join('version_data', saved_filename)

    try:
        with open(save_path, "wb") as f:
            f.write(data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
    
    df = training_data_processing(save_path, data_version=True)
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Not enough data or empty file",
        )
    return "Data processed and saved successfully."



@app.get("/evaluate-models", tags=["Admin API's"])
def evaluate_models():
    evaluation = evaluate_best_model()
    return evaluation.to_dict(orient='records')



@app.get("/save-best-model", tags=["Admin API's"])
def save_best_model():
    best_model_name = get_best_model()
    train_status = train_on_entire_data(best_model_name)
    if train_status:
        return f"{best_model_name} trained successfully"
    return "Something went wrong!"










from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pickle
import os
from typing import List, Dict, Tuple
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox
import logging

from src.model_prediction import ModelPredictor
from src.data_processing import DataProcessor
from src.config import (
    RAW_DATA_PATH,
    MODELS_DIR,
    FORECASTS_DIR,
    SARIMA_MODEL_CONFIGS,
    TEST_SIZE_WEEKS
)

logger = logging.getLogger(__name__)
load_dotenv()

# Define the input data scheme using Pydantic
class InputData(BaseModel):
    historical_sales: List[float] # Pd.Series
    forecast_steps: int = 52 # Default to 52 weeks if not provided

# --- Authentication Configuration ---
# For a real project, store this in environment variable, NOT HARDCODED!
API_KEY = "password" # <-- REPLACE WITH A STRONG KEY IN PRODUCTION
API_KEY_NAME = "X-API-Key" # This is the name of the HTTP header where the client will send the key
# Define the API Key Header scheme
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Dependency to validate the API key
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=401, detail="Unauthorized: Invalid API Key"
    )

# --- Global storage for loaded models and lambda values ---
loaded_models: Dict[str, Tuple[any, float]] = {}
global_predictor: ModelPredictor = None # Assigning after loading ML models

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_predictor # Declare intent to modify global variable

    print("Loading models on application startup...")

    api_data_processor = DataProcessor(file_path=RAW_DATA_PATH)
    global_predictor = ModelPredictor(
        data_processor=api_data_processor,
        models_dir=MODELS_DIR,
        forecasts_dir=FORECASTS_DIR,
        model_configs=SARIMA_MODEL_CONFIGS,
        test_size_weeks=TEST_SIZE_WEEKS
    )

    categories = ["Furniture", "Office Supplies", "Technology"]

    for category in categories:
        model, lambda_val = global_predictor.load_model_and_lambda(category)
        if model and lambda_val is not None:
            loaded_models[category] = (model, lambda_val)
            print(f"Successfully loaded {category} model")
        else:
            print(f"Failed to load model for {category}. This category will not be available for predictions.")

    yield

    print("App shutting down...")
    # Any clean-up can go here

app = FastAPI(title="Sales Forecast Prediction API", lifespan=lifespan)

# --- Root endpoint for health check ---
@app.get("/")
async def root():
    """Basic health check endpoint."""
    return {"message": "Welcome to the Retail Demand Forecasting API! Use /predict/{category_name} for forecasts."}

# --- Prediction endpoint for a specific category
@app.post("/predict/{category}")
def predict_sales(category: str, data: InputData, api_key: str = Depends(get_api_key)):
    """
    Generates a sales forecast for the specified category.
    The 'historical_sales' should be the most recent weekly sales figures
    leading up to the forecast period.
    Requires an API key in the 'X-API-Key' header
    """
    if category not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model for category '{category}' not found or not loaded.")

    model_results, lambda_value = loaded_models[category]
    # Double-check if model_results is actually loaded (redundant but safe)
    if model_results is None:
        raise HTTPException(status_code=500, detail=f"Model for category '{category}' is not available due to a loading error.")

    if global_predictor is None:
        raise HTTPException(status_code=500, detail="Forecasting engine not initialised. Server startup failed.")

    try:
        forecast_values = global_predictor.generate_single_forecast(
            model_results=model_results,
            lambda_value=lambda_value,
            historical_sales=data.historical_sales,
            forecast_steps=data.forecast_steps
        )
        return {"category": category, "forecast": forecast_values}

    except Exception as e:
        print(f"Error during prediction for {category}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed for {category}: {str(e)}. Please check your input data and server logs.")

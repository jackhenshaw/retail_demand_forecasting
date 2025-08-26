from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends, Request
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

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_ipaddr
from slowapi.errors import RateLimitExceeded

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
    # Example values for quick testing
    historical_sales: List[float] = [400, 500, 600, 300]
    forecast_steps: int = 4

# --- Authentication Configuration ---
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

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

# --- Rate Limiter Configuration ---
limiter = Limiter(key_func=get_ipaddr, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Root endpoint for health check ---
@app.get("/")
@limiter.limit("5/minute")
async def root(request: Request):
    """Basic health check endpoint."""
    return {"message": "Welcome to the Retail Demand Forecasting API! Use /predict/{category_name} for forecasts."}

# --- Prediction endpoint for a specific category
@app.post("/predict/{category}")
@limiter.limit("5/minute")
def predict_sales(category: str, request: Request, data: InputData, api_key: str = Depends(get_api_key)):
    """
    Generates a sales forecast for the specified category.
    The 'historical_sales' should be the most recent weekly sales figures
    leading up to the forecast period.

    Requires an API key in the 'X-API-Key' header.

    Category: ["Furniture", "Office Supplies", "Technology"]
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

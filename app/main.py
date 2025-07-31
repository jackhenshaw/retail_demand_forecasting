from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
from typing import List, Dict, Tuple
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox

from src.model_prediction import ModelPredictor
from src.data_processing import DataProcessor
from src.config import RAW_DATA_PATH, MODELS_DIR, FORECASTS_DIR, SARIMA_MODEL_CONFIGS, TEST_SIZE_WEEKS

# Define the input data scheme using Pydantic
class InputData(BaseModel):
    historical_sales: List[float] # Pd.Series
    forecast_steps: int = 52 # Default to 52 weeks if not provided

app = FastAPI(title="Sales Forecast Prediction API")

# --- Global storage for loaded models and lambda values ---
loaded_models: Dict[str, Tuple[any, float]] = {}
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
global_predictor: ModelPredictor = None # Assing after loading ML models

# --- Helpher function to load a single model and its lambda
def load_category_model(category: str) -> Tuple[any, float]:
    """Loads SARIMA modeal and lambda for a given category"""
    model_filename = os.path.join(MODELS_DIR, f'{category.lower().replace(" ", "_")}_sarima_model.pkl')
    lambda_filename = os.path.join(MODELS_DIR, f'{category.lower().replace(" ", "_")}_lambda.pkl')

    try:
        with open(model_filename, "rb") as f:
            model_results = pickle.load(f)
        with open(lambda_filename, "rb") as f:
            lambda_value = pickle.load(f)
        print(f"Loaded model and lambda for {category}")
        return model_results, lambda_value
    except FileNotFoundError:
        print(f"Error: Model or lambda file not found for {category} at {model_filename} / {lambda_filename}. Ensure models are trained and saved.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading model for {category}: {e}")
        return None, None

@app.on_event("startup")
async def startup_event():
    global global_predictor # Declare intent to modify global variable

    print("Loading models on application startup...")
    categories = ["Furniture", "Office Supplies", "Technology"]

    for category in categories:
        model, lambda_val = load_category_model(category)
        if model and lambda_val is not None:
            loaded_models[category] = (model, lambda_val)
        else:
            print(f"Failed to load model for {category}. This category will not be available for predictions.")

    # Initialise ModelPredictor here after models are loaded.
    api_data_processor = DataProcessor(file_path=RAW_DATA_PATH)
    global_predictor = ModelPredictor(
        data_processor=api_data_processor,
        models_dir=MODELS_DIR,
        forecasts_dir=FORECASTS_DIR,
        model_configs=SARIMA_MODEL_CONFIGS,
        test_size_weeks=TEST_SIZE_WEEKS
    )

# --- Root endpoint for health check ---
@app.get("/")
async def root():
    """Basic health check endpoint."""
    return {"message": "Welcome to the Retail Demand Forecasting API! Use /predict/{category_name} for forecasts."}

# --- Prediction endpoint for a specific category
@app.post("/predict/{category}")
def predict_sales(category: str, data: InputData):
    """
    Generates a sales forecast for the specified category.
    The 'historical_sales' should be the most recent weekly sales figures
    leading up to the forecast period.
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
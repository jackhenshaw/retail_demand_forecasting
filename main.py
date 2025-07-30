import os

from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.model_prediction import ModelPredictor
from src.config import (
    RAW_DATA_PATH,
    SARIMA_MODEL_CONFIGS,
    MODELS_DIR,
    FORECASTS_DIR,
    TEST_SIZE_WEEKS
)

def main():
    """
    Main function to run the retail demand forecasting pipeline.
    This includes data processing, model training, and forecast generation.
    """
    print("---Starting Retail Demand Forecasting Pipeline")

    data_processor = DataProcessor(file_path=RAW_DATA_PATH)

    trainer = ModelTrainer(
        data_processor=data_processor,
        model_configs=SARIMA_MODEL_CONFIGS,
        models_dir=MODELS_DIR,
        test_size_weeks=TEST_SIZE_WEEKS
    )
    trainer.run_training()

    predictor = ModelPredictor(
        data_processor=data_processor,
        models_dir=MODELS_DIR,
        forecasts_dir=FORECASTS_DIR,
        model_configs=SARIMA_MODEL_CONFIGS,
        test_size_weeks=TEST_SIZE_WEEKS
    )
    predictor.run_prediction()

    print("\n--Retail Demand Forecasting Pipeline Completed ---")
    print(f"Trained models saved in: {MODELS_DIR}/")
    print(f"Forecasts saved in: {FORECASTS_DIR}/combined_sales_forecasts.csv")

if __name__ == "__main__":
    main()
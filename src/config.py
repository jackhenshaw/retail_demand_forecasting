SARIMA_MODEL_CONFIGS = {
    "Furniture": {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 0, 52),
        "iqr_multiplier": 4.0
    },
    "Office Supplies": {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 0, 52),
        "iqr_multiplier": 4.0
    },
    "Technology": {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 0, 52),
        "iqr_multiplier": 4.0
    }
}

MODELS_DIR = "models"
FORECASTS_DIR = "forecasts"
RAW_DATA_PATH = "data/raw/Sample-Superstore.csv"
TEST_SIZE_WEEKS = 52
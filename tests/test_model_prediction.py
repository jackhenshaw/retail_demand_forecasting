import pytest
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from src.model_prediction import ModelPredictor
from src.data_processing import DataProcessor
from src.config import (
    RAW_DATA_PATH,
    MODELS_DIR,
    FORECASTS_DIR,
    SARIMA_MODEL_CONFIGS,
    TEST_SIZE_WEEKS
)

# --- Simple mock SARIMAXResultsWrapper for testing ---
class MockSARIMAXResults:
    def __init__(self, fitted_values_len: int, mock_predictions: list):
        # fitted_values_len: The legnth of the 'history' the model 'knows'
        self.fittedvalues = np.zeros(fitted_values_len)
        self._mock_predictions = mock_predictions # Pre-defined predictions for testing

    def append(self, new_data: pd.Series) -> 'MockSARIMAXResults':
        # Mimic appending data. In a real scenario, this updates the models internal state.
        # For this mock we just increase the length of 'fittedvalues' to simulate.
        new_len = len(self.fittedvalues) + len(new_data)
        return MockSARIMAXResults(new_len, self._mock_predictions)

    def predict(self, start: int, end: int) -> pd.Series:
        # Mimic prediction. Returns pre-defined mock predictions.
        # Ensure the mock predictions cover the requested range (end - start + 1)
        num_steps = end - start + 1
        if len(self._mock_predictions) < num_steps:
            raise ValueError(f"Mock predictions not sufficient for {num_steps} steps.")

        return pd.Series(self._mock_predictions[:num_steps], index=pd.RangeIndex(start, end + 1))

@pytest.fixture
def mock_predictor():
    """
    Fixture to provide a ModelPredictor instance for testing.
    Uses a dummy DataProcessor as it's not directly used by generate_single_forecast.
    """
    dummy_data_processor = DataProcessor(file_path="dummy_path.csv")

    # Instantiate ModelPredictor with dummy values for other args,
    # as generate_single_forecast only uses the model_results and lambda_value directly.
    predictor = ModelPredictor(
        data_processor=dummy_data_processor,
        models_dir=MODELS_DIR,
        forecasts_dir=FORECASTS_DIR,
        model_configs=SARIMA_MODEL_CONFIGS,
        test_size_weeks=TEST_SIZE_WEEKS
    )
    return predictor

def test_generate_single_forecast_basic(mock_predictor):
    initial_fitted_len = 100
    mock_predictions_transformed = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_model_results = MockSARIMAXResults(initial_fitted_len, mock_predictions_transformed)

    # Note: historical_sales are untransformed, but only used to determine append_data length.
    # The actual values don't affect the predict() output of our MockSARIMAXResults
    # because it returns pre-defined mock_predictions_transformed
    historical_sales = [50.0, 55.0, 60.0, 65.0, 70.0]
    forecast_steps = 5

    test_lambda = 0.5
    # inv_boxcox(y, 0.5) - 1 for y = [0.1, 0.2, 0.3, 0.4, 0.5]
    # x = exp( ln(lambda*y + 1) / lambda) - 1 <-- added 1 in forward transformation
    expected_forecast = [0.1025, 0.2100, 0.3225, 0.4400, 0.5625]

    actual_forecast = mock_predictor.generate_single_forecast(
        model_results=mock_model_results,
        lambda_value=test_lambda,
        historical_sales=historical_sales,
        forecast_steps=forecast_steps
    )

    assert isinstance(actual_forecast, list)
    assert len(actual_forecast) == forecast_steps
    assert np.allclose(actual_forecast, expected_forecast, atol=1e-6)

def test_generate_single_forecast_no_historical_sales(mock_predictor):
    """
    Tests forecasting when no new historical sales are provided.
    This simulates calling forecast on the originally fitted model.
    """
    initial_fitted_len = 100
    mock_predictions_transformed = [10.0, 11.0, 12.0]
    mock_model_results = MockSARIMAXResults(initial_fitted_len, mock_predictions_transformed)

    historical_sales = [] # No new data
    forecast_steps = 3
    test_lambda = 0.5

    # Expected: (x^2) - 1
    # 10.0^2 - 1 = 99.0
    # 11.0^2 - 1 = 120.0
    # 12.0^2 - 1 = 143.0
    expected_forecast = [35.00, 41.25, 48.00]

    actual_forecast = mock_predictor.generate_single_forecast(
        model_results=mock_model_results,
        lambda_value=test_lambda,
        historical_sales=historical_sales,
        forecast_steps=forecast_steps
    )

    assert isinstance(actual_forecast, list)
    assert len(actual_forecast) == forecast_steps
    assert np.allclose(actual_forecast, expected_forecast, atol=1e-6)

def test_generate_single_forecast_error_handling(mock_predictor, mocker):
    """
    Tests that errors during prediction are handled (e.g. re-raised)
    """
    mock_model_results = MockSARIMAXResults(100, [0.1])
    test_lambda = 0.5
    historical_sales = [50.0]
    forecast_steps = 1

    mock_model_results_error = MockSARIMAXResults(100, []) # Not enough predictions

    with pytest.raises(ValueError, match="Mock predictions not sufficient"):
        mock_predictor.generate_single_forecast(
            model_results=mock_model_results_error, # Use the error-simulating mock
            lambda_value=test_lambda,
            historical_sales=historical_sales,
            forecast_steps=forecast_steps
        )
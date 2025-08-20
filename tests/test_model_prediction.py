import pytest
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from unittest.mock import MagicMock, patch, mock_open

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

@pytest.fixture
def mock_data_processor():
    return MagicMock(spec=DataProcessor)

def test_load_model_and_lambda_success(mock_predictor, mocker):
    mock_model_results = MagicMock()
    mock_lambda = 0.5

    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        with patch('pickle.load', side_effect=[mock_model_results, mock_lambda]) as mock_pickle_load:
            category = "Furniture"

            model_results, lambda_value = mock_predictor.load_model_and_lambda(category)

            mock_file.assert_any_call(os.path.join(mock_predictor.models_dir, "furniture_sarima_model.pkl"), "rb")
            mock_file.assert_any_call(os.path.join(mock_predictor.models_dir, "furniture_lambda.pkl"), "rb")
            assert mock_pickle_load.call_count == 2
            assert model_results is mock_model_results
            assert lambda_value == mock_lambda

def test_load_model_and_lambda_file_not_found(mock_predictor, mocker):
    mock_file = mock_open()
    mock_file.side_effect = FileNotFoundError

    with patch('builtins.open', mock_file):
        with patch('logging.error') as mock_logger_error:
            category = "Furniture"

            model_results, lambda_value = mock_predictor.load_model_and_lambda(category)
            assert model_results is None
            assert lambda_value is None

            mock_logger_error.assert_called_once()
            assert "file not found" in mock_logger_error.call_args[0][0]

def test_generate_forecasts_success(mock_predictor, mocker):
    # Mock the SARIMA model and its predict method
    mock_model_results = MagicMock(spec=SARIMAXResultsWrapper)
    mock_model_results.fittedvalues = pd.Series([1] * (52 - mock_predictor.test_size_weeks))

    # Define mock predictions for the test and future periods
    mock_predictions_transformed = np.linspace(50, 60, mock_predictor.test_size_weeks + mock_predictor.forecast_steps)
    mocker.patch.object(mock_predictor, 'generate_single_forecast',
                        return_value=mock_predictions_transformed.tolist())

    # Create mock weekly data
    weekly_data = pd.DataFrame({
        'Sales': np.linspace(100, 200, 52)
    }, index=pd.date_range('2023-01-01', periods=52, freq='W-SUN'))

    # Call the method under test
    forecasts_df = mock_predictor.generate_forecasts(
        category='Test',
        weekly_data=weekly_data,
        model_results=mock_model_results,
        lambda_value=0.5
    )

    # Assertions
    assert not forecasts_df.empty
    assert 'Category' in forecasts_df.columns
    assert 'Actuals' in forecasts_df.columns
    assert 'Predicted' in forecasts_df.columns
    assert forecasts_df['Category'].nunique() == 1

    # Assert correct number of rows and columns
    expected_rows = len(weekly_data) + mock_predictor.forecast_steps
    assert len(forecasts_df) == expected_rows
    assert len(forecasts_df.columns) == 3

    # Check for NaNs and non-NaNs in the correct places
    assert forecasts_df['Actuals'].isnull().sum() == mock_predictor.forecast_steps
    assert forecasts_df['Predicted'].isnull().sum() == len(weekly_data) - mock_predictor.test_size_weeks

    # Assert that the predicted values match the mock values
    actual_predictions = forecasts_df['Predicted'].dropna().values
    assert np.allclose(actual_predictions, mock_predictions_transformed, atol=1e-6)

def test_generate_forecasts_error_handling(mock_predictor, mocker):
    with patch('src.model_prediction.logger.error') as mock_logger_error:
        forecasts_df = mock_predictor.generate_forecasts(
            category='Test',
            weekly_data=pd.DataFrame(),
            model_results=None,  # Simulate missing model
            lambda_value=0.5
        )
        assert forecasts_df.empty
        mock_logger_error.assert_called_once()
        assert "Cannot generate forecasts for Test: Model or lambda not loaded." in mock_logger_error.call_args[0][0]

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

    # Expected: (lambda * y + 1)^2 - 1
    # For y = 10.0: (0.5 * 10.0 + 1)^2 - 1 = 35.00
    # For y = 11.0: (0.5 * 11.0 + 1)^2 - 1 = 41.25
    # For y = 12.0: (0.5 * 12.0 + 1)^2 - 1 = 48.00
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

def test_run_prediction_success(mock_data_processor, mocker):
    # Create the ModelPredictor instance with the mock DataProcessor
    predictor = ModelPredictor(data_processor=mock_data_processor)

    # Mock the behavior of the data processor's methods
    mock_data_processor.load_data.return_value = None
    mock_data_processor.df = pd.DataFrame({'a':[1]})

    def mock_aggregrate(category=None):
        return pd.DataFrame({'Sales': [1, 2, 3, 4, 5]}, index=pd.date_range('2023-01-01', periods=5, freq='W'))
    mock_data_processor.aggregrate_to_weekly.side_effect = mock_aggregrate

    # Mock the dependencies of the predictor
    mocker.patch('os.path.join', return_value='dummy_path.csv')
    mocker.patch('pandas.DataFrame.to_csv')
    mocker.patch('builtins.open', mock_open(read_data=b'')) # Mock the pickle file opening

    # Mock the internal methods of the predictor instance
    mocker.patch.object(predictor, 'load_model_and_lambda', return_value=(MagicMock(), 0.5))
    mocker.patch.object(predictor, 'generate_forecasts', return_value=pd.DataFrame({'a':[1], 'b':[2]}))

    # Run the method under test
    predictor.run_prediction()

    # Assert that the key methods were called
    mock_data_processor.load_data.assert_called_once()
    assert mock_data_processor.aggregrate_to_weekly.call_count == len(SARIMA_MODEL_CONFIGS)
    predictor.load_model_and_lambda.assert_called()
    predictor.generate_forecasts.assert_called()
    pd.DataFrame.to_csv.assert_called_once_with('dummy_path.csv')


def test_run_prediction_load_data_error(mock_data_processor, mocker):
    # Create the ModelPredictor instance with the mock DataProcessor
    predictor = ModelPredictor(data_processor=mock_data_processor)

    # Mock `load_data` to return None, simulating an error
    mock_data_processor.load_data.return_value = None
    mock_data_processor.df = None

    with patch('logging.error') as mock_logger_error:
        with patch.object(predictor, 'generate_forecasts') as mock_generate_forecasts:

            predictor.run_prediction()

            # Assert that `load_data` was called, but `generate_forecasts` was not
            mock_data_processor.load_data.assert_called_once()
            mock_generate_forecasts.assert_not_called()

            # Assert that an error was logged
            mock_logger_error.assert_called_once()
            assert "Cannot run predictions: Raw data not loaded." in mock_logger_error.call_args[0][0]

def test_run_prediction_aggregation_error(mock_predictor, mock_data_processor, mocker):
    # Create the ModelPredictor instance with the mock DataProcessor
    predictor = ModelPredictor(data_processor=mock_data_processor)

    # Mock `load_data` to return a valid DataFrame
    mock_data_processor.load_data.return_value = MagicMock()
    mock_data_processor.df = MagicMock()

    # Mock `aggregrate_to_weekly` to fail for one category (returns None)
    def mock_aggregrate_side_effect(category=None):
        if category == 'Technology':
            return None
        return pd.DataFrame({'Sales': [1, 2, 3]})

    mock_data_processor.aggregrate_to_weekly.side_effect = mock_aggregrate_side_effect

    # Mock `generate_forecasts` to return a non-empty DataFrame to simulate success
    with patch.object(predictor, 'generate_forecasts', return_value=pd.DataFrame({'a': [1]})) as mock_generate_forecasts:
        with patch('logging.error') as mock_logger_error:
            predictor.run_prediction()

            # Assert that `aggregrate_to_weekly` was called for each category
            assert mock_data_processor.aggregrate_to_weekly.call_count == len(SARIMA_MODEL_CONFIGS)

            # Assert that `generate_forecasts` was called twice (for Furniture and Office Supplies)
            assert mock_generate_forecasts.call_count == 2

            # Assert that generate_forecasts was called with the correct arguments for Furniture
            mock_generate_forecasts.assert_any_call(
                category='Furniture',
                weekly_data=mocker.ANY,
                model_results=mocker.ANY,
                lambda_value=mocker.ANY
            )

            # Assert that generate_forecasts was also called with the correct arguments for Office Supplies
            mock_generate_forecasts.assert_any_call(
                category='Office Supplies',
                weekly_data=mocker.ANY,
                model_results=mocker.ANY,
                lambda_value=mocker.ANY
            )

            # Assert that an error was logged only once for the failed aggregation
            mock_logger_error.assert_called_once()
            assert "Skipping prediction for Technology due to data aggregation error." in mock_logger_error.call_args[0][0]
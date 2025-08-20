import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
from src.model_training import ModelTrainer
from src.data_processing import DataProcessor
import os
import pickle
import logging

# Set up a basic logger for the test file
logging.basicConfig(level=logging.INFO)

# A simple mock SARIMA model to use in the tests
class MockSARIMAResults:
    def __init__(self):
        pass
    def summary(self):
        return "Mock SARIMA Summary"

def test_constructor():
    # Create a dummy DataProcessor and a dummy directory name
    mock_data_processor = MagicMock(spec=DataProcessor)
    dummy_models_dir = 'dummy_models_dir'

    # Use patch to mock os.makedirs and prevent actual directory creation
    with patch('os.makedirs') as mock_makedirs:
        mt = ModelTrainer(data_processor=mock_data_processor, models_dir=dummy_models_dir)

        # Assert the attributes were set correctly
        assert mt.data_processor == mock_data_processor
        assert mt.models_dir == dummy_models_dir

        # Assert that os.makedirs was called with the correct directory and `exist_ok=True`
        mock_makedirs.assert_called_with(dummy_models_dir, exist_ok=True)


def test_train_model():
    # Create a mock DataProcessor and its expected return values
    mock_data_processor = MagicMock(spec=DataProcessor)
    # Mock return values for `detect_and_treat_outliers`
    # The cleaned series will be the input series itself, but this simulates the method being called.
    mock_cleaned_series = pd.Series([10, 11, 12])
    mock_data_processor.detect_and_treat_outliers.return_value = mock_cleaned_series
    # Mock return values for `apply_boxcox_transformation`
    mock_transformed_series = pd.Series([1, 2, 3])
    mock_lambda = 0.5
    mock_data_processor.apply_boxcox_transformation.return_value = (mock_transformed_series, mock_lambda)

    # Mock a mock SARIMA model and results
    mock_sarima_model = MagicMock()
    mock_sarima_model.fit.return_value = MockSARIMAResults()

    # Patch the SARIMAX class to return our mock model
    with patch('src.model_training.SARIMAX', return_value=mock_sarima_model) as mock_SARIMAX:
        # Patch `os.path.join` and `pickle.dump` to prevent file I/O
        with patch('os.path.join', side_effect=os.path.join) as mock_join:
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('pickle.dump') as mock_pickle_dump:
                    mt = ModelTrainer(data_processor=mock_data_processor, models_dir='mock_models')

                    # Call the method to be tested
                    category = 'Furniture'
                    weekly_data = pd.DataFrame({'Sales': pd.Series([10, 11, 12])}, index=pd.to_datetime(['2020-01-01', '2020-01-08', '2020-01-15']))
                    order = (1, 1, 1)
                    seasonal_order = (1, 1, 1, 52)
                    iqr_multiplier = 1.5

                    mt.test_size_weeks = 1 # Set a small test size for the mock data

                    mt.train_model(category, weekly_data, order, seasonal_order, iqr_multiplier)

                    # Assert that DataProcessor methods were called correctly
                    mock_data_processor.detect_and_treat_outliers.assert_called_once()
                    mock_data_processor.apply_boxcox_transformation.assert_called_once()

                    mock_SARIMAX.assert_called_once()
                    call_args, call_kwargs = mock_SARIMAX.call_args
                    pd.testing.assert_series_equal(call_args[0], mock_transformed_series.iloc[:-1])
                    assert call_kwargs['order'] == order
                    assert call_kwargs['seasonal_order'] == seasonal_order
                    assert call_kwargs['freq'] == 'W'
                    mock_sarima_model.fit.assert_called_once()

                    # Assert that the model and lambda value were saved
                    assert mock_pickle_dump.call_count == 2

                    # Get the calls to pickle.dump
                    model_call, lambda_call = mock_pickle_dump.call_args_list

                    # Check if the correct objects were pickled
                    assert isinstance(model_call.args[0], MockSARIMAResults)
                    assert model_call.args[1] is not None # Check file handler

                    assert lambda_call.args[0] == mock_lambda
                    assert lambda_call.args[1] is not None # Check file handler

def test_run_training():
    # Mock dependencies and configuration
    mock_data_processor = MagicMock(spec=DataProcessor)
    mock_data_processor.load_data.return_value = pd.DataFrame()
    mock_data_processor.df = pd.DataFrame()
    mock_data_processor.aggregrate_to_weekly.return_value = pd.DataFrame(
        {'Sales': [1, 2, 3]},
        index=pd.to_datetime(['2020-01-01', '2020-01-08', '2020-01-15'])
    )
    mock_configs = {
        "Furniture": {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 52), "iqr_multiplier": 1.5},
        "Technology": {"order": (0, 1, 1), "seasonal_order": (0, 1, 1, 52), "iqr_multiplier": 1.5}
    }

    # Patch the `train_model` method of the ModelTrainer instance itself
    # This prevents the test from executing the entire training process
    with patch.object(ModelTrainer, 'train_model', return_value=None) as mock_train_model:
        mt = ModelTrainer(
            data_processor=mock_data_processor,
            model_configs=mock_configs,
            models_dir='mock_models'
        )
        mt.run_training()

        mock_data_processor.load_data.assert_called_once()

        # Assert that `aggregrate_to_weekly` was called for each category
        assert mock_data_processor.aggregrate_to_weekly.call_count == 2
        mock_data_processor.aggregrate_to_weekly.assert_any_call(category="Furniture")
        mock_data_processor.aggregrate_to_weekly.assert_any_call(category="Technology")

        # Assert that `train_model` was called for each category with the correct arguments
        assert mock_train_model.call_count == 2

        # Check the calls for the 'Furniture' category
        mock_train_model.assert_any_call(
            category='Furniture',
            weekly_data=mock_data_processor.aggregrate_to_weekly.return_value,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 52),
            iqr_multiplier=1.5
        )

        # Check the calls for the 'Technology' category
        mock_train_model.assert_any_call(
            category='Technology',
            weekly_data=mock_data_processor.aggregrate_to_weekly.return_value,
            order=(0, 1, 1),
            seasonal_order=(0, 1, 1, 52),
            iqr_multiplier=1.5
        )

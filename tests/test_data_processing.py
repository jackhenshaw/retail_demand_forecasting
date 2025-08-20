import pytest
import pandas as pd
from scipy.stats import boxcox
import numpy as np
from unittest.mock import patch, mock_open
from src.data_processing import DataProcessor

mock_data = """
Order Date,Ship Date,Category,Sales
01/01/1999,02/02/1999,Furniture,5
06/06/2000,09/09/2000,Technology,8
"""


def test_constructor():
    dp = DataProcessor("dummy_string")
    assert dp.file_path == "dummy_string"
    assert dp.df is None

@patch('builtins.open', mock_open(read_data=mock_data))
def test_read_data():
    dp = DataProcessor("dummy_string")
    df = dp.load_data()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2,4)
    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])
    assert pd.api.types.is_datetime64_any_dtype(df["Ship Date"])
    # Assert a value to ensure the data is what we expect
    assert df.iloc[0]["Sales"] == 5

def test_aggregate_to_weekly_no_data():
    dp = DataProcessor("dummy_string")
    # Test when df is empty
    assert dp.aggregrate_to_weekly() is None

def test_aggregate_to_weekly_all_categories():
    dp = DataProcessor("dummy_string")
    mock_df = pd.DataFrame({
        "Order Date": ["01/01/2020", "01/01/2020", "01/12/2020", "01/12/2020"], # American dates
        "Ship Date":  ["01/01/2020", "01/01/2020", "01/12/2020", "01/12/2020"], # American dates
        "Category": ["Furniture", "Furniture", "Furniture", "Technology"],
        "Sales": [1.0, 2.0, 3.0, 4.0]
    })
    mock_df["Order Date"] = pd.to_datetime(mock_df["Order Date"])
    dp.df = mock_df
    aggregated_sales = dp.aggregrate_to_weekly()

    expected_df = pd.DataFrame(
        data=[3.0, 7.0],
        index=pd.Index(pd.to_datetime(["2020-01-05", "2020-01-12"]), name="Order Date"),
        columns=["Sales"]
    )
    pd.testing.assert_frame_equal(aggregated_sales, expected_df, check_freq=False)

def test_aggregate_to_weekly_with_category():
    dp = DataProcessor("dummy_string")
    mock_df = pd.DataFrame({
        "Order Date": ["01/01/2020", "01/01/2020", "01/12/2020", "01/12/2020"], # American dates
        "Ship Date":  ["01/01/2020", "01/01/2020", "01/12/2020", "01/12/2020"], # American dates
        "Category": ["Furniture", "Furniture", "Furniture", "Technology"],
        "Sales": [1.0, 2.0, 3.0, 4.0]
    })
    mock_df["Order Date"] = pd.to_datetime(mock_df["Order Date"])
    dp.df = mock_df
    aggregated_sales = dp.aggregrate_to_weekly("Furniture")

    expected_df = pd.DataFrame(
        data=[3.0, 3.0],
        index=pd.Index(pd.to_datetime(["2020-01-05", "2020-01-12"]), name="Order Date"),
        columns=["Sales"]
    )
    pd.testing.assert_frame_equal(aggregated_sales, expected_df, check_freq=False)

def test_apply_boxcox_transformation():
    mock_series = pd.Series([1, 2, 3, 4, 5])

    expected_series_tuple, expected_lambda = boxcox(mock_series + 1)
    expected_series = pd.Series(expected_series_tuple)

    dp = DataProcessor("dummy_string")
    transformed_series, lambda_val = dp.apply_boxcox_transformation(mock_series)

    np.testing.assert_allclose(transformed_series.values, expected_series.values)
    assert lambda_val == pytest.approx(expected_lambda)

def test_inverse_boxcox_transformation():
    mock_series_original = pd.Series([10, 20, 30, 40, 50])
    transformed_series, lambda_val = boxcox(mock_series_original + 1)

    dp = DataProcessor("dummy_string")
    inverse_transformed_series = dp.inverse_boxcox_transformation(pd.Series(transformed_series), lambda_val)
    np.testing.assert_allclose(inverse_transformed_series.values, mock_series_original.values)

def test_detect_and_treat_outliers():
    # Create a mock series with a known outlier
    data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 100.0]
    dates = pd.to_datetime(pd.date_range('2020-01-01', periods=len(data), freq='D'))
    mock_series = pd.Series(data, index=dates)

    dp = DataProcessor("dummy_string")
    cleaned_series = dp.detect_and_treat_outliers(mock_series.copy(), iqr_multiplier=1.5, window_size=2)

    # The outlier (100) should be replaced with the median of its neighbors (18 and 19), which is 18.5
    expected_data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 18.5]
    expected_series = pd.Series(expected_data, index=dates)

    # Assert that the cleaned series matches the expected series
    pd.testing.assert_series_equal(cleaned_series, expected_series)
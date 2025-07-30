import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class to handle all data loading, cleaning, aggregation, and transformation
    steps for the retail sales data.
    """
    def __init__(self, file_path: str):
        """
        Args:
            file_path (str): The path to the raw 'Sample-Superstore.csv' file
        """
        self.file_path = file_path
        self.df = None # To store the loaded DataFrame

    def load_data(self) -> pd.DataFrame:
        """
        Loads the raw sales data from the specified csv file.

        Returns:
            pd.DataFrame: The loaded DataFrame with 'Order Date' parsed.
        """
        try:
            self.df = pd.read_csv(
                self.file_path,
                encoding='latin1',
                parse_dates=["Order Date", "Ship Date"]
            )
            logger.info(f"Data loaded successfully from {self.file_path}")
            return self.df
        except FileNotFoundError:
            logger.error(f"Error: File not found at {self.file_path}. Please check the path.", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An error occurred while loading data: {e}", exc_info=True)
            return None

    def aggregrate_to_weekly(self, category: str = None) -> pd.DataFrame:
        """
        Aggregrates sales data to a weekly frequency.
        Can filter by category if specified.

        Args:
            category (str, optional): The category to filter by (e.g 'Furniture',
                                      'Office Supplies', 'Technology'). If None,
                                      aggregrates total sales. Defaults to None.

        Returns:
            pd.DataFrame: Weekly aggregated sales DataFrame with 'Order Date' as index.
        """

        if self.df is None:
            logger.error("Error: Data not loaded. Call load_data() first.", exc_info=True)
            return None

        df_filtered = self.df.copy()
        if category:
            df_filtered = df_filtered[df_filtered["Category"] == category]
            logger.info(f"Aggregating weekly sales for category: {category}")
        else:
            logger.info("Aggregating weekly sales for all categories.")

        # Set 'Order Date' as DataFrame index for resampling
        df_filtered.set_index("Order Date", inplace=True)

        # Resample to weekly totals, ending on Sundays
        weekly_data = df_filtered.resample("W").agg(Sales=("Sales", "sum"))
        # Fill any weeks with no sales with 0
        weekly_data["Sales"] = weekly_data["Sales"].fillna(0)

        logger.info(f"Weekly data prepared. Length: {len(weekly_data)} weeks.")
        return weekly_data

    def apply_boxcox_transformation(self, data: pd.Series) -> tuple[pd.Series, float]:
        """
        Applies the Box-Cox transformation to a sales series.
        Adds 1 to handle zero values before transformation.

        Args:
            data (pd.Series): The sales data series (e.g weekly_data["Sales"])

        Returns:
            tuple[pd.Series, float]: Tuple containing the transformed series
                                     and the lambda value used.
        """
        # Add a small constant to handle zero or negative values before Box-Cox
        # We've seen sales can be 0, so +1 is appropriate
        transformed_data, lambda_value = boxcox(data + 1)
        logger.info(f"Box-Cox transformation applied. Lambda: {lambda_value:.4f}")
        return pd.Series(transformed_data, index=data.index), lambda_value

    def inverse_boxcox_transformation(self, transformed_data: pd.Series, lambda_value: float) -> pd.Series:
        """
        Applies the inverse Box-Cox transformation to convert data back to the
        original scale. Subtracts 1 to reverse the initial offset.

        Args:
            transformed_data (pd.Series): The Box-Cox transformed data series.
            lambda_value (float): The lambda value used during the forward transformation.

        Returns:
            pd.Series: The data in its original scale.
        """
        original_scale_data = inv_boxcox(transformed_data, lambda_value) - 1
        # Ensure no negative values after inverse transform (due to slight numerical errors or model predicting negative)
        original_scale_data[original_scale_data < 0] = 0
        logger.info("Inverse Box-Cox transformation applied.")
        return original_scale_data

    def detect_and_treat_outliers(self, data_series: pd.Series, window_size: int = 2, iqr_multiplier: float = 4) -> pd.Series:
        """
        Generically detects and treats outliers in a given time series using the IQR method.
        Outliers are replaced with the median of their surrounding values.

        Note: Using fixed IQR is not neccessarily the best idea with trending data.

        Args:
            data_series (pd.Series): The time series data to process.
            window_size (int): Number of data points before and after the outlier
                               to consider for calculating the median replacement.
                               Defaults to 2.
            iqr_multiplier (float): The multiplier for the IQR to define outlier bounds.
                                    Commonly 1.5 (for mild outliers) or 3.0 (for extreme).
                                    After inspection in jupyter notebook using 4.0

        Returns:
            pd.Series: The series with detected outliers treated.
        """
        cleaned_series = data_series.copy()

        Q1 = cleaned_series.quantile(0.25)
        Q3 = cleaned_series.quantile(0.75)
        IQR = Q3-Q1

        lower_bound = Q1 - (iqr_multiplier * IQR)
        upper_bound = Q3 + (iqr_multiplier * IQR)
        if lower_bound < 0:
            lower_bound = 0

        outlier_indices = cleaned_series[(cleaned_series < lower_bound) | (cleaned_series > upper_bound)].index
        if not outlier_indices.empty:
            logger.info(f"Detected {len(outlier_indices)} outliers using IQR method (multiplier={iqr_multiplier}).")
            for outlier_date in outlier_indices:
                outlier_idx = cleaned_series.index.get_loc(outlier_date)

                neighbour_values = []
                for i in range(1, window_size + 1):
                    if outlier_idx - i >= 0:
                        neighbour_values.append(cleaned_series.iloc[outlier_idx - i])
                    if outlier_idx + i < len(cleaned_series):
                        neighbour_values.append(cleaned_series.iloc[outlier_idx + i])

                if neighbour_values:
                    median_replacement = pd.Series(neighbour_values).median()
                    original_value = cleaned_series.loc[outlier_date]
                    cleaned_series.loc[outlier_date] = median_replacement
                    logger.info(f"Treated outlier at {outlier_date}: {original_value:.2f} replaced with {median_replacement:.2f}")
                else:
                    logger.info(f"Could not treat outlier at {outlier_date}: Not enough neighbours.")
        else:
            logger.info("No outliers detected using the IQR method")

        return cleaned_series



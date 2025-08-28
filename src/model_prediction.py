from dotenv import load_dotenv
import pandas as pd
import pickle
import os
import logging
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from typing import List
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from src.data_processing import DataProcessor
from src.config import MODELS_DIR, FORECASTS_DIR, TEST_SIZE_WEEKS, SARIMA_MODEL_CONFIGS, FORECAST_STEPS

load_dotenv()
AZURE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME")

logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    A class to handle loading trained SARIMA models and generating sales forecasts.
    It combines actuals and predictions for easy computation by a dashboard.
    """
    def __init__(self, data_processor: DataProcessor,
                 models_dir: str = MODELS_DIR,
                 forecasts_dir: str = FORECASTS_DIR,
                 model_configs: dict = SARIMA_MODEL_CONFIGS,
                 test_size_weeks: int = TEST_SIZE_WEEKS,
                 forecast_steps: int = FORECAST_STEPS):
        """
        Initialises the ModelPredictor.

        Args:
            data_processor (DataProcessor): An instance of the DataProcessor class
                                            to handle data preparation and inverse
                                            transformation.
            models_dir (str): Directory where trained models are stored.
            forecasts_dir (str): Directory where forecast outputs will be saved.
            model_configs (dict): Dictionary of SARIMA model configurations per category.
            forecast_steps (int): The number of future periods to forecast.
        """
        self.data_processor = data_processor
        self.models_dir = models_dir
        self.forecasts_dir = forecasts_dir
        self.model_configs = model_configs
        self.test_size_weeks = test_size_weeks
        self.forecast_steps = forecast_steps

        os.makedirs(self.forecasts_dir, exist_ok=True)
        logger.info(f"ModelPredictor Initialised. Forecasts will be saved to: {self.forecasts_dir}")

    def download_models_from_azure(self, blob_name: str, local_file_path: str):
        """
        Downloads a file from Azure Blob Storage

        Args:
            blob_name (str): The name of the blob in Azure Storage.
            local_file_path (str): The local path where the file will be saved.
        """
        if not AZURE_CONNECTION_STRING:
            logger.warning("AZURE_STORAGE_CONNNECTION_STRING is not set. Skipping model download.")
            return

        try:
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
            blob_client = blob_service_client.get_blob_client(
                container=AZURE_CONTAINER_NAME,
                blob=blob_name
            )

            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            logger.info(f"Successfully downloaded {blob_name} to {local_file_path}")
        except Exception as e:
            logger.error(f"Failed to download {blob_name} from Azure: {e}")

    def load_model_and_lambda(self, category: str) -> tuple:
        """
        Loads the trained SARIMA model and its corresponding lambda value for a given category.

        Args:
            category (str): The name of the sales category.

        Returns:
            tuple: A tuple containing (model_results, lambda_value) if successful,
                   otherwise (None, None)
        """
        model_filename = f'{category.lower().replace(" ", "_")}_sarima_model.pkl'
        lambda_filename = f'{category.lower().replace(" ", "_")}_lambda.pkl'
        model_path = os.path.join(self.models_dir, model_filename)
        lambda_path = os.path.join(self.models_dir, lambda_filename)

        if os.path.exists(model_path) and os.path.exists(lambda_path):
            try:
                with open(model_path, "rb") as model_file, open(lambda_path, "rb") as lambda_file:
                    model = pickle.load(model_file)
                    lambda_file = pickle.load(lambda_file)
                logger.info(f"Successfully loaded model and lambda for {category} from local storage.")
                return model, lambda_file
            except Exception as e:
                logger.error(f"Error loading local model files for {category}: {e}")

        # If local files are missing, attempt to download them
        self.download_models_from_azure(blob_name=model_filename, local_file_path=model_path)
        self.download_models_from_azure(blob_name=lambda_filename, local_file_path=lambda_path)

        # After downloading, try to load them again
        if os.path.exists(model_path) and os.path.exists(lambda_path):
            try:
                with open(model_path, "rb") as model_file, open(lambda_path, "rb") as lambda_file:
                    model = pickle.load(model_file)
                    lambda_file = pickle.load(lambda_file)
                logger.info(f"Successfully loaded model and lambda for {category} after downloading from Azure.")
                return model, lambda_file
            except Exception as e:
                logger.error(f"Error loading downloaded model files for {category}: {e}")
                return None, None
        else:
            logger.error(f"Model files for {category} were not found locally or failed to download from Azure.")
            return None, None

    def generate_forecasts(self, category: str, weekly_data: pd.DataFrame,
                          model_results: SARIMAXResultsWrapper, lambda_value: float) -> pd.DataFrame:
        """
        Generates a combined DataFrame of historical data, test set actuals, and predictions.

        Args:
            category (str): The category name.
            weekly_data (pd.DataFrame): The full weekly aggregated data for the category.
            model_results (SARIMAXResultsWrapper): The loaded model results.
            lambda_value (float): The loaded Box-Cox lambda value.

        Returns:
            pd.DataFrame: A DataFrame with 'Category', 'Actuals', and 'Predicted' columns,
                          or an empty DataFrame if an error occurs.
        """
        if model_results is None or lambda_value is None:
            logger.error(f"Cannot generate forecasts for {category}: Model or lambda not loaded.")
            return pd.DataFrame()

        # Split the data into training and test sets
        train_data = weekly_data[:-self.test_size_weeks]
        test_data = weekly_data[-self.test_size_weeks:]

        # Create a list of all historical sales (training + test)
        historical_sales = test_data['Sales'].tolist()

        # The total number of steps to forecast is the test size plus the future steps
        total_forecast_steps = self.test_size_weeks + self.forecast_steps

        # Generate the forecast
        forecasted_values = self.generate_single_forecast(
            model_results=model_results,
            lambda_value=lambda_value,
            historical_sales=historical_sales,
            forecast_steps=total_forecast_steps
        )

        future_dates = pd.date_range(start=weekly_data.index[-1] + pd.DateOffset(weeks=1),
                                     periods=self.forecast_steps,
                                     freq='W-SUN')
        full_index = weekly_data.index.append(future_dates)

        # Create the combined DataFrame
        combined_df = pd.DataFrame(index=full_index)
        combined_df['Category'] = category
        combined_df['Actuals'] = weekly_data['Sales']

        # Align predictions with the correct dates
        # The predictions start at the first date of the test set
        prediction_index = full_index[-total_forecast_steps:]
        combined_df['Predicted'] = pd.Series(forecasted_values, index=prediction_index)

        return combined_df

    def generate_single_forecast(self,
                                 model_results,
                                 lambda_value: float,
                                 historical_sales: List[float],
                                 forecast_steps: int) -> List[float]:
        """
        Generates a single forecast for a given model, lambda, and historical sales data.
        This method is designed for live API predictions where new historical data
        is provided to update the model state before forecasting.

        Args:
            model_results: The loaded SARIMAXResults object for the category
            lambda_value (float): The Box-Cox lambda value for the category.
            historical_sales (List[float]): The most recent actual sales data (untransformed).
            forecast_steps (int): The number of steps (weeks) to forecast.

        Returns:
            List[float]: The forecasted sales values, inverse-transformed and non-negative.
        """
        try:
            if historical_sales:
                prepared_historical_series = pd.Series(historical_sales) + 1
                transformed_historical_sales = boxcox(prepared_historical_series, lmbda=lambda_value)
                updated_model_results = model_results.append(transformed_historical_sales)
            else:
                updated_model_results = model_results

            start_forecast_index = len(updated_model_results.fittedvalues)
            end_forecast_index = start_forecast_index + forecast_steps - 1
            predictions_boxcox = updated_model_results.predict(
                start=start_forecast_index,
                end=end_forecast_index
            )
            forecast_values = inv_boxcox(predictions_boxcox, lambda_value) - 1
            forecast_values[forecast_values < 0] = 0
            return forecast_values.tolist()

        except Exception as e:
            logger.error(f"Error during single forecast generation: {e}", exc_info=True)
            # Re-raise the exception or return an empty list/error indicitor
            raise # Re-raise to be caught by the API endpoint's try-except

    def run_prediction(self):
        """
        Orchestrates the prediction process for all categories and saves the results.
        This method will also save a combined plot data file for each category.
        """
        self.data_processor.load_data()
        if self.data_processor.df is None:
            logger.error("Cannot run prediction: Raw data not loaded. Exiting.")
            return

        for category, config in self.model_configs.items():
            logger.info(f"\n--- Running prediction for {category} ---")
            weekly_data = self.data_processor.aggregrate_to_weekly(category=category)
            if weekly_data is None:
                logger.error(f"Skipping {category} due to data aggregation error.")
                continue

            # Split the data to get the test set for plotting
            train_size = len(weekly_data) - self.test_size_weeks
            test_data = weekly_data[train_size:]

            # Generate and save the combined data for the dashboard plot
            dashboard_df = self.generate_dashboard_data(
                category=category,
                weekly_data=weekly_data,
                test_data=test_data
            )

            if dashboard_df is not None:
                filename = os.path.join(self.forecasts_dir, f"{category}_dashboard_data.csv")
                dashboard_df.to_csv(filename, index=False)
                logger.info(f"Dashboard plot data for {category} saved to {filename}")

    def generate_dashboard_data(self, category: str, weekly_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a combined DataFrame for dashboard plotting, including historical,
        test, and future forecast data.

        Args:
            category (str): The sales category.
            weekly_data (pd.DataFrame): The full weekly sales data
            test_data (pd.DataFrame): The actual sales data for the test period.

        Returns:
            pd.DataFrame: A combined DataFrame ready for plotting.
        """
        model_results, lambda_value = self.load_model_and_lambda(category)
        if model_results is None:
            return None

        # Split data into train and test sets
        train_size = len(weekly_data) - self.test_size_weeks
        train_data = weekly_data[:train_size]

        # Generate predicitons for the test set
        predictions_transformed = model_results.predict(
            start=len(train_data),
            end=len(train_data) + len(test_data) - 1,
            dynamic=True
        )
        predictions = self.data_processor.inverse_boxcox_transformation(predictions_transformed.values, lambda_value)
        predictions_series = pd.Series(predictions, index=test_data.index, name='Predicted Sales')

        # Generate future forecast
        future_forecast_transformed = model_results.forecast(steps=self.forecast_steps)
        future_forecast = self.data_processor.inverse_boxcox_transformation(future_forecast_transformed, lambda_value)

        # Create a new data range for the future forecast
        last_date = test_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=self.forecast_steps, freq='W')
        future_forecast_series = pd.Series(future_forecast, index=future_dates, name='Predicted Sales')

        # Combine all data into a single DataFrame
        combined_df = pd.DataFrame(index=weekly_data.index.append(future_dates))
        combined_df['Actual Sales'] = weekly_data['Sales']
        combined_df['Predicted Sales'] = pd.concat([predictions_series, future_forecast_series])
        combined_df['Category'] = category

        return combined_df.reset_index().rename(columns={'index': 'Date'})

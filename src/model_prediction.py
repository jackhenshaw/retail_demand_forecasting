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
        model_filename = os.path.join(self.models_dir, f'{category.lower().replace(" ", "_")}_sarima_model.pkl')
        lambda_filename = os.path.join(self.models_dir, f'{category.lower().replace(" ", "_")}_lambda.pkl')

        os.makedirs(self.models_dir, exist_ok=True)

        # Attempt to download the models from Azure first
        self.download_models_from_azure(f"{category.lower().replace(' ', '_')}_sarima_model.pkl", model_filename)
        self.download_models_from_azure(f"{category.lower().replace(' ', '_')}_lambda.pkl", lambda_filename)

        if not os.path.exists(model_filename) or not os.path.exists(lambda_filename):
            logger.error(f"Model or lambda file not found locally for {category}.")
            return None, None

        try:
            with open(model_filename, 'rb') as f:
                model_results = pickle.load(f)
            with open(lambda_filename, 'rb') as f:
                lambda_value = pickle.load(f)
            logger.info(f"Model and lambda value loaded for {category}")
            return model_results, lambda_value
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Error loading model or lambda for {category}: {e}", exc_info=True)
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
        Orchestrates the prediction process for all categories.
        Combines all category forecasts into a single DataFrame for PowerBI.
        """
        logging.info("\n--- Running full prediction pipeline for all categories ---")
        all_forecasts = []

        self.data_processor.load_data()
        if self.data_processor.df is None:
            logging.error("Cannot run predictions: Raw data not loaded.", exc_info=True)
            return

        for category, config in self.model_configs.items():
            weekly_data = self.data_processor.aggregrate_to_weekly(category=category)
            if weekly_data is None:
                logging.error(f"Skipping prediction for {category} due to data aggregation error.")
                continue

            model_results, lambda_value = self.load_model_and_lambda(category)

            category_forecasts_df = self.generate_forecasts(
                category=category,
                weekly_data=weekly_data,
                model_results=model_results,
                lambda_value=lambda_value
            )
            if not category_forecasts_df.empty:
                all_forecasts.append(category_forecasts_df)

        if all_forecasts:
            final_forecast_df = pd.concat(all_forecasts).sort_index()
            forecast_filename = os.path.join(self.forecasts_dir, 'combined_sales_forecasts.csv')
            final_forecast_df.to_csv(forecast_filename)
            logging.info(f"All category forecasts combined and saved to {forecast_filename}")
        else:
            logging.error("No forecasts were generated for any category", exc_info=True)

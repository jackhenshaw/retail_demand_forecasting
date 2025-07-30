import pandas as pd
import pickle
import os
import logging

from src.data_processing import DataProcessor
from src.config import MODELS_DIR, FORECASTS_DIR, TEST_SIZE_WEEKS, SARIMA_MODEL_CONFIGS

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
                 test_size_weeks: int = TEST_SIZE_WEEKS):
        """
        Initialises the ModelPredictor.

        Args:
            data_processor (DataProcessor): An instance of the DataProcessor class
                                            to handle data preparation and inverse
                                            transformation.
            models_dir (str): Directory where trained models are stored.
            forecasts_dir (str): Directory where forecast outputs will be saved.
            model_configs (dict): Dictionary of SARIMA model configurations per category.
        """
        self.data_processor = data_processor
        self.models_dir = models_dir
        self.forecasts_dir = forecasts_dir
        self.model_configs = model_configs
        self.test_size_weeks = test_size_weeks

        os.makedirs(self.forecasts_dir, exist_ok=True)
        logger.info(f"ModelPredictor Initialised. Forecasts will be saved to: {self.forecasts_dir}")

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

        try:
            with open(model_filename, 'rb') as f:
                model_results = pickle.load(f)
            with open(lambda_filename, 'rb') as f:
                lambda_value = pickle.load(f)
            logger.info(f"Model and lambda value loaded for {category}")
            return model_results, lambda_value
        except FileNotFoundError:
            logging.error(f"Error: Model or lambda file not found for {category}. Please ensure models are trained first.", exc_info=True)
            return None, None
        except Exception as e:
            logging.error(f"An error occured while loading model for {category}: {e}", exc_info=True)
            return None, None

    def generate_forecasts(self,
                           category: str,
                           weekly_data: pd.DataFrame,
                           model_results,
                           lambda_value: float) -> pd.DataFrame:
        """
        Generates forecasts for a given category using its trained model.
        The forecasts are made for the test_size_weeks period.

        Args:
            category (str): The name of the sales category.
            weekly_data (pd.DataFrame): The full weekly aggregated sales data for
                                        the category (including both training and
                                        testing periods).
            model_results: The fitted SARIMAXResults object
            lambda_value (float): The lambda value used for Box-Cox transformation.

        Returns:
            pd.DataFrame: A DataFrame containing 'Order Date', 'Category', 'Actual Sales',
                          and 'Predicted Sales' for the forecast period.
        """
        if model_results is None or lambda_value is None:
            logging.error(f"Cannot generate forecasts for {category}: Model or lambda not loaded.", exc_info=True)
            return pd.DataFrame()

        # The predict method needs the start and end indices relative to the *original*
        # series length used for training, plus the forecast horizon.
        # Here, we are predicting for the held-out test set
        train_size = len(weekly_data) - self.test_size_weeks
        start_index = train_size
        end_index = len(weekly_data) - 1

        logging.info(f"Generating forecasts for {category} from index {start_index} to {end_index}")

        try:
            predictions_boxcox = model_results.predict(start=start_index, end=end_index)
            predicted_sales = self.data_processor.inverse_boxcox_transformation(
                predictions_boxcox, lambda_value
            )

            actual_sales_series = weekly_data["Sales"].iloc[start_index:end_index+1]

            # Create a DataFrame for the results
            forecast_df = pd.DataFrame({
                "Order Date": actual_sales_series.index,
                "Category": category,
                "Actual Sales": actual_sales_series,
                "Predicted Sales": predicted_sales
            })
            forecast_df.set_index("Order Date", inplace=True)
            logging.info(f"Forecasts generated for {category}.")
            return forecast_df

        except Exception as e:
            logging.error(f"Error generating forecasts for {category}: {e}", exc_info=True)
            return pd.DataFrame()

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
import pandas as pd
import pickle
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.special import inv_boxcox
import numpy as np
from datetime import datetime

from src.data_processing import DataProcessor
from src.config import SARIMA_MODEL_CONFIGS, MODELS_DIR, TEST_SIZE_WEEKS

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A class to handle the training of SARIMA models for different sales categories.
    It uses processed data, applies train/test split, fits models, and saves them.
    """
    def __init__(self, data_processor: DataProcessor,
                 model_configs: dict = SARIMA_MODEL_CONFIGS,
                 models_dir: str = MODELS_DIR,
                 test_size_weeks:int = TEST_SIZE_WEEKS
                 ):
        """
        Initialises the ModelTrainer.

        Args:
            data_processor (DataProcessor): An instance of the DataProcessor class
                                            to handle data preparation.
            model_configs (dict): A dictionary where keys are category names (str)
                                  and values are dictionaries containing SARIMA
                                  'order', 'seasonal_order' and 'iqr_multiplier'.
            models_dir (str): The directory where trained models will be saved.
            test_size_weeks (int): Number of weeks to reserve for the test set during training
        """
        self.data_processor = data_processor
        self.model_configs = model_configs
        self.models_dir = models_dir
        self.test_size_weeks = test_size_weeks

        os.makedirs(self.models_dir, exist_ok=True)
        logger.info(f"ModelTrainer initialised. Models will be saved to {self.models_dir}")

    def train_model(self, category: str, weekly_data: pd.DataFrame,
                    order: tuple, seasonal_order: tuple, iqr_multiplier: float):
        """
        Trains a SARIMA model for a specific category

        Args:
            category (str): The name of the sales category (e.g Furniture).
            weekly_data (pd.DataFrame): The weekly aggregated sales data for the category.
            order (tuple): Non-seasonal (p, d, q) order of the SARIMA model.
            seasonal_order (tuple): Seasonal (P, D, Q, s) order of the SARIMA model.
            iqr_multiplier (float): Multiplier for the IQR to detect and treat outliers.

        Returns:
            dict: A dictionary containing calculated performance metrics.
                  MAE, RMSE, MAPE
        """
        logger.info(f"\n--- Starting training for {category} category ---")

        # 1. Treat outliers
        logger.info(f"Applying outlier treatment with IQR multiplier: {iqr_multiplier}")
        weekly_data["Sales"] = self.data_processor.detect_and_treat_outliers(
            weekly_data["Sales"],
            iqr_multiplier=iqr_multiplier
        )

        # 2. Apply Box-Cox Transformation
        transformed_sales, lambda_value = self.data_processor.apply_boxcox_transformation(
            weekly_data["Sales"]
        )

        # 3. Prepare Train/Test Data (on transformed series)
        train_size = len(transformed_sales) - self.test_size_weeks
        train_data = transformed_sales.iloc[:train_size]
        test_data = transformed_sales.iloc[train_size:] # This will be used for evaluation later

        logger.info(f"Train data length: {len(train_data)} weeks")
        logger.info(f"Test data length: {len(test_data)} weeks")
        logger.info(f"SARIMA order: {order}, Seasonal order: {seasonal_order}")

        # 4. Fit SARIMA Model
        try:
            sarima_model = SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
                freq="W"
            )
            results = sarima_model.fit(disp=False) # suppres convergence messages during fitting
            logger.info(f"SARIMA model for {category} fitted successfully.")
            # Optionally print summary for debugging/review
            #print(results.summary())

            # 5. Save the trained model and lambda value
            model_filename = os.path.join(self.models_dir, f'{category.lower().replace(" ", "_")}_sarima_model.pkl')
            lambda_filename = os.path.join(self.models_dir, f'{category.lower().replace(" ", "_")}_lambda.pkl')

            with open(model_filename, 'wb') as f:
                pickle.dump(results, f)
            with open(lambda_filename, 'wb') as f:
                pickle.dump(lambda_value, f)
            logger.info(f"Model and lambda value for {category} save to {self.models_dir}.")

            # 6. Calculate performance metrics
            test_predictions = results.forecast(self.test_size_weeks)
            forecast_values = inv_boxcox(test_predictions, lambda_value) - 1
            test_values = inv_boxcox(test_data, lambda_value) - 1
            perf_metrics = self.calculate_metrics(test_values, forecast_values)
            self.save_metrics_to_csv(perf_metrics, category)

        except Exception as e:
            logger.error(f"Error training model for {category}: {e}", exc_info=True)

    def run_training(self):
        """
        Orchestrates the training process for all categories defined in model_configs.
        """
        self.data_processor.load_data()
        if self.data_processor.df is None:
            logger.error("Cannot run training: Raw data not loaded. Exiting.", exc_info=True)
            return

        logger.info("\n--- Running full training pipeline for all categories ---")
        for category, config in self.model_configs.items():
            # Aggregate data for the current category
            weekly_data = self.data_processor.aggregrate_to_weekly(category=category)
            if weekly_data is None:
                logger.error(f"Skipping {category} due to data aggregation error.", exc_info=True)
                continue

            self.train_model(
                category=category,
                weekly_data=weekly_data,
                order=config["order"],
                seasonal_order=config["seasonal_order"],
                iqr_multiplier=config['iqr_multiplier']
            )
        logger.info("\n--- All category models training complete ---")

    def calculate_metrics(self, actual, prediction):
        """
        Calculates performance metrics: MAE, RMSE, MAPE

        Args:
            actual (np.ndarray or pd.Series): The actual, true, sales values.
            prediction (np.ndarray or pd.Series): The model's forecasted values.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        # Ensure inputs are numpy arrays for consistent operations
        actual = np.array(actual)
        prediction = np.array(prediction)

        # Calculate MAE
        mae = mean_absolute_error(actual, prediction)

        # Calculate RMSE
        rmse = root_mean_squared_error(actual, prediction)

        # Calculate MAPE
        mape_actual = np.where(actual == 0, 1e-8, actual)
        mape = np.mean(np.abs((actual - prediction) / mape_actual)) * 100

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        }

    def save_metrics_to_csv(self, metrics: dict, category: str):
        """
        Saves the performance metrics to a CSV file.
        The file is created if it does not exist, and new data is appended.

        Args:
            metrics (dict): A dictionary of performance metrics (MAE, RMSE, MAPE)
            category (str): The category namer for which the metrics were calculated.
        """
        metrics_file_path = os.path.join(self.models_dir, 'model_performances.csv')

        new_entry = pd.DataFrame([{
            'timestamp': datetime.now(),
            'category': category,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape']
        }])

        if os.path.exists(metrics_file_path):
            new_entry.to_csv(metrics_file_path, mode='a', header=False, index=False)
            logger.info(f"Appended metrics for {category} to {metrics_file_path}")
        else:
            new_entry.to_csv(metrics_file_path, mode='w', header=True, index=False)
            logger.info(f"Created new metrics file at {metrics_file_path}")

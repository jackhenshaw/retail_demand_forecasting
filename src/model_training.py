import pandas as pd
import pickle
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging

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
# Superstore Sales Forecasting and Performance Analysis

## 1. Overview and Problem Statement
This project aims to build a sales forecasting model for a global Superstore dataset and provide actionable insights into sales perfomance and trends. 
Accurate sales forecasts are critical for optimising inventory management, improving supply chain efficiency, and making informed business decisions, while performance analysis helps identify key drivers of success and areas for improvement.

### Question: 
Can we accurately forecast future sales at a granular level (e.g by product category or region) to improve inventory management and resource allocation? Additionally, how can we leverage historical data to analyse sales performance and profitability?

## 2. Dataset
Name: Superstore Dataset \
Source: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final/data \
Description: Contains detailed sales data for a global superstore including customer information, product details, sales figures, profit, order dates, and geographical locations.

## 3. Project Structure & Workflow (High-Level Plan)
- [x] Phase-1: Exploratory data analysis & data preprocessing
  - Initial data exploration to understand distributions, trends, and relationships.
  - Handling missing values, outliers, and feature engineering
  - Developing key analytical insights and initial dashboards in PowerBI (including learning PowerBI)
- [x] Phase-2: Sales Forecasting Model Development
  - Selection and training of a suitable time-series forecasting model (e.g ARIMA, Prophet, or a simpler regression-based approach)
  - Model evaluation and selection
- [x] Phase-2.5: Refactoring to a Python Pipeline
  - Conversion of exploratory notebooks into reusable, scalable Python files, establishing a robust forecasting pipeline. This includes modularising data processing, model training, and prediction logic, and integrating formal logging.
    - **Data Processing** (`data_processing.py`): Responsible for loading data, aggregating it to a weekly frequency, and applying transformations like outlier treatment and Box-Cox normalisation.
    - **Model Training** (`model_training.py`): Handles fitting SARIMA models for each category, and saving the trained models along with their transformation parameters.
    - **Model Prediction** (`model_prediction.py`): Loads the trained models, generates future forecasts, and applies the inverse transformation to produce final, readable sales predictions.
- [ ] Phase-3: Model Deployment (IN PROGRESS)
  - [x] Containerisation of the trained model (e.g using Docker)
  - [x] Exposing the model via a REST API (e.g using Flask/FastAPI)
    - Including super basic authentication
  - [ ] Deployment to a cloud platform (Azure)
- [x] Phase-4: Automated Testing
  - [x] Build out unit tests using `pytest`  for the `DataProcessor`, `ModelTrainer`, and `ModelPredictor` logic. 
  - [x] Build out integration tests using `pytest` for the API endpoints, including authentication checks.
- [ ] Phase-5: Performance Monitoring & Visualisation
  - Setting up basic monitoring for the deployed model's performance
  - Creating interactive Power BI dashboards to visualise forecasted vs actual sales, model accuracy and key performance indicators for business users.
     
### Future Work
- [ ] Phase-6: Model improvements:
  - [ ] Improve outlier detection model. Current implementation is fixed IQR range, should move towards rolling window IQR or z-score. 
  - [ ] ML models version control (MLFlow?)
 
## 4. Technologies Used
- Languages:
  - Python
- Libraries:
  - pandas, NumPy (for data manipulation)
  - Matplotlib, Seaborn (for initial python visualisations)
  - statsmodels (for ARIMA/SARIMA modeling)
  - FastAPI, uvicorn (for deployment)
  - Pytest, unittest.mock (for automated testing)
- Tools & Platforms:
  - Jupyter Notebooks (for development)
  - Power BI Desktop (for interactive dashboards)
  - Git, GitHub (for version control)
  - Azure (for cloud deployment)

## 5. Getting Started
To run the retail demand forecasting pipeline, follow these steps:
1. **Clone the repository:**
   ```
   git clone https://github.com/jackhenshaw/retail_demand_forecasting.git
   cd retail_demand_forecasting
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Place your data:** Ensure your `Sample-Superstore.csv` file is located in the `data/raw/` directory.
4. **Run the main script:**
   ```
   python main.py
   ```
   This will execute the full pipeline, training models and saving forecasts to the `forecasts/` directory.

### Running tests
To execute the automated testing suite, navigate to the project root directory and run:
```
pytest
```
The testing suite verifies the core business logic of the forecasting pipeline and ensures the API endpoints are functioning correctly. The tests use mocking to isolate functions, ensuring resillience against failures and guaranteering predictable results.

## 6. Screenshots & Dashboards
### Basic EDA dashboards
<img width="1632" height="830" alt="image" src="https://github.com/user-attachments/assets/de25b391-800e-403c-ade7-b9bcb1827352" />
<img width="1630" height="832" alt="image" src="https://github.com/user-attachments/assets/6cf2e358-66f5-45ef-909f-6dd490e23ded" />
<img width="1632" height="832" alt="image" src="https://github.com/user-attachments/assets/5f5eea41-73b9-4e1c-9299-c6b565308855" />
<img width="1632" height="834" alt="image" src="https://github.com/user-attachments/assets/5a254d08-c14d-477c-ac5f-2ea16c17e343" />



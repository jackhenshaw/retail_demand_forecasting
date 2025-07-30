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
- [ ] Phase-3: Model Deployment
  - Containerisation of the trained model (e.g using Docker)
  - Exposing the model via a REST API (e.g using Flask/FastAPI)
  - Deployment to a cloud platform (Azure)
- [ ] Phase-4: Performance Monitoring & Visualisation
  - Setting up basic monitoring for the deployed model's performance
  - Creating interactive Power BI dashboards to visualise forecasted vs actual sales, model accuracy and key performance indicators for business users.
 
## 4. Technologies Used
- Languages:
  - Python
- Libraries:
  - pandas, NumPy (for data manipulation)
  - Matplotlib, Seaborn (for initial python visualisations)
  - statsmodels (for ARIMA/SARIMA modeling)
  - Flask/FastAPI, Docker (for deployment)
- Tools & Platforms:
  - Jupyter Notebooks (for development)
  - Power BI Desktop (for interactive dashboards)
  - Git, GitHub (for version control)
  - Azure (for cloud deployment)

## 5. Getting Started (Future Section)

## 6. Screenshots & Dashboards
### Basic EDA dashboards
<img width="1632" height="830" alt="image" src="https://github.com/user-attachments/assets/de25b391-800e-403c-ade7-b9bcb1827352" />
<img width="1630" height="832" alt="image" src="https://github.com/user-attachments/assets/6cf2e358-66f5-45ef-909f-6dd490e23ded" />
<img width="1632" height="832" alt="image" src="https://github.com/user-attachments/assets/5f5eea41-73b9-4e1c-9299-c6b565308855" />
<img width="1632" height="834" alt="image" src="https://github.com/user-attachments/assets/5a254d08-c14d-477c-ac5f-2ea16c17e343" />



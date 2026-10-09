# Task 3: Predictive Analytics — Monthly Sales Prediction

## 1. Project Overview
This project demonstrates predictive analytics by analyzing historical monthly sales data and forecasting sales for the next three months using Machine Learning.

## 2. Objective
- Analyze historical sales trends.
- Train a Linear Regression model.
- Predict future monthly sales.
- Evaluate model performance using standard regression metrics.
- Visualize historical and predicted sales.

## 3. Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- CSV dataset

## 4. Dataset
The dataset contains 21 monthly sales records from January 2025 to September 2026. It is synthetic sample data created for educational purposes, not actual company sales data.

## 5. MMethodology
1. Load and inspect the dataset using Pandas.
2. Visualize historical sales with Matplotlib.
3. Convert months into sequential numeric features.
4. Train a Linear Regression model.
5. Forecast sales for October, November, and December 2026.
6. Evaluate predictions using MAE, MSE, RMSE, and R².
7. Test model performance on the final five months using a chronological train/test split.
8. Save the forecasts, evaluation results, and chart.

## 6. Project Outputs
- `sales_forecast_next_3_months.csv` — forecast results.
- `model_evaluation.csv` — regression evaluation metrics.
- `sales_forecast.png` — visualization of historical sales and forecasts.

## 7. Limitations
The dataset is small and synthetic. The simple Linear Regression model captures a general time trend but may not capture seasonality or external factors. Results are for demonstration and should not be used for real business decisions.

## 8. Conclusion
This project demonstrates a basic predictive analytics workflow, from data loading and visualization to model training, evaluation, forecasting, and saving results.

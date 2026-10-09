
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("monthly_sales.csv")

print("First five records:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSales summary:")
print(df["Sales"].describe())

df["Month"]=pd.to_datetime(df["Month"])

plt.figure(figsize=(10,5))

plt.plot(df["Month"],df["Sales"],marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()


df["MonthNumber"] = range(1, len(df) + 1)

X = df[["MonthNumber"]]
y = df["Sales"]

print("\nInput data (X):")
print(X.head())

print("\nTarget data (y):")
print(y.head())


model = LinearRegression()

model.fit(X, y)

print("\nModel trained successfully!")

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)



future_months = pd.DataFrame({
    "MonthNumber": [22, 23, 24]
})

predictions = model.predict(future_months)

future_months["PredictedSales"] = predictions.round(2)

future_months["Month"] = [
    "2026-10",
    "2026-11",
    "2026-12"
]

print("\nSales Forecast for Next 3 Months:")
print(future_months[["Month", "PredictedSales"]])


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

predicted_sales = model.predict(X)

mae = mean_absolute_error(y, predicted_sales)
mse = mean_squared_error(y, predicted_sales)
rmse = mse ** 0.5
r2 = r2_score(y, predicted_sales)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R-squared (R2):", round(r2, 4))



future_months.to_csv(
    "sales_forecast_next_3_months.csv",
    index=False
)

evaluation = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Value": [mae, mse, rmse, r2]
})

evaluation.to_csv("model_evaluation.csv", index=False)

print("\nFiles saved successfully!")
print("1. sales_forecast_next_3_months.csv")
print("2. model_evaluation.csv")




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


X_train = X.iloc[:-5]
X_test = X.iloc[-5:]

y_train = y.iloc[:-5]
y_test = y.iloc[-5:]


test_model = LinearRegression()
test_model.fit(X_train, y_train)

test_predictions = test_model.predict(X_test)


test_mae = mean_absolute_error(y_test, test_predictions)

print("\nUnseen Data Evaluation:")
print("Actual sales:", list(y_test))
print("Predicted sales:", test_predictions.round(2).tolist())
print("Test MAE:", round(test_mae, 2))


test_evaluation = pd.DataFrame({
    "Metric": ["Test MAE"],
    "Value": [round(test_mae, 2)]
})

test_evaluation.to_csv(
    "test_evaluation.csv",
    index=False
)

print("Test evaluation saved successfully!")


plt.figure(figsize=(10, 5))

plt.plot(
    df["Month"],
    df["Sales"],
    marker="o",
    label="Historical Sales"
)

future_dates = pd.to_datetime(future_months["Month"])

plt.plot(
    future_dates,
    future_months["PredictedSales"],
    marker="o",
    linestyle="--",
    label="Predicted Sales"
)

plt.title("Monthly Sales Prediction")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("sales_forecast.png")
plt.show()

print("\nForecast graph saved successfully!")


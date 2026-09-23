
# ============================================================
# CocaCola Sales Forecasting using Decomposition & Holt-Winters
# ============================================================

# Importing necessary libraries
import pandas as pd                     # For data handling
import numpy as np                      # For numerical operations
import matplotlib.pyplot as plt         # For plotting graphs
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing


# ------------------------------------------------------------
# Load the dataset
# ------------------------------------------------------------
cocacola = pd.read_excel("C:/360DG/Datasets/CocaCola_Sales_Rawdata.xlsx")

# Plot original sales data to visualize trend/seasonality
cocacola.Sales.plot(title="CocaCola Sales Data")
plt.show()


# ============================================================
# Trend & Seasonality Interpretation
# ============================================================

# Upward Trend
# The overall direction of the line is moving upwards.
# This means CocaCola sales are increasing over time.
# For example, early quarters show ~1500,
# while later quarters show ~5000+.

# Seasonality / Fluctuations
# Even though trend is upward, there are regular ups and downs.
# Indicates quarterly seasonal pattern.

# Volatility
# Fluctuations become larger as sales grow.
# Later quarters show bigger peaks & dips than early quarters.
# Suggests multiplicative seasonality may fit better.


# ============================================================
# Decomposition (Additive Model)
# Sales = Trend + Seasonality + Residual
# ============================================================

decompose_ts_add = seasonal_decompose(
    cocacola.Sales,
    model="additive",
    period=4
)

print(decompose_ts_add.trend)      # Trend component
print(decompose_ts_add.seasonal)   # Seasonal component
print(decompose_ts_add.resid)      # Residual component
print(decompose_ts_add.observed)   # Original data
decompose_ts_add.plot()
plt.show()


# ============================================================
# Decomposition (Multiplicative Model)
# Sales = Trend × Seasonality × Residual
# ============================================================

decompose_ts_mul = seasonal_decompose(
    cocacola.Sales,
    model="multiplicative",
    period=4
)

print(decompose_ts_mul.trend)
print(decompose_ts_mul.seasonal)
print(decompose_ts_mul.resid)
print(decompose_ts_mul.observed)
decompose_ts_mul.plot()
plt.show()


# ============================================================
# Split Dataset
# ------------------------------------------------------------
# Data is quarterly → 4 quarters per year
# Training: first 38 quarters
# Testing: last 4 quarters
# ============================================================

Train = cocacola.head(38)
Test = cocacola.tail(4)


# ============================================================
# Define MAPE (Mean Absolute Percentage Error)
# ============================================================

def MAPE(pred, actual):
    temp = np.abs((pred - actual) / actual) * 100
    return np.mean(temp)


# ============================================================
# Exploratory Moving Average (EDA)
# ============================================================

# Calculate 4-quarter moving average
mv_pred = cocacola["Sales"].rolling(4).mean()

# Compare last 4 predicted vs test
print("Moving Average MAPE:", MAPE(mv_pred.tail(4), Test.Sales))

# Plot different rolling window sizes
cocacola.Sales.plot(label="Original")
for i in range(2, 9, 2):
    cocacola["Sales"].rolling(i).mean().plot(label=str(i))

plt.legend(loc=3)
plt.title("Rolling Mean Comparison")
plt.show()


# ============================================================
# Data-Driven Forecasting Models
# ============================================================

# 1️⃣ Simple Exponential Smoothing (Level only)
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start=Test.index[0], end=Test.index[-1])
print("SES MAPE:", MAPE(pred_ses, Test.Sales))


# 2️⃣ Holt’s Method (Level + Trend)
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start=Test.index[0], end=Test.index[-1])
print("Holt MAPE:", MAPE(pred_hw, Test.Sales))


# 3️⃣ Holt-Winters (Additive Trend + Additive Seasonality)
hwe_model_add_add = ExponentialSmoothing(
    Train["Sales"],
    trend="add",
    seasonal="add",
    seasonal_periods=4
).fit()

pred_hwe_model_add_add = hwe_model_add_add.predict(
    start=Test.index[0],
    end=Test.index[-1]
)

print("Holt-Winters (Add-Add) MAPE:",
      MAPE(pred_hwe_model_add_add, Test.Sales))


# 4️⃣ Holt-Winters (Additive Trend + Multiplicative Seasonality)
hwe_model_mul_add = ExponentialSmoothing(
    Train["Sales"],
    trend="add",
    seasonal="mul",
    seasonal_periods=4
).fit()

pred_hwe_model_mul_add = hwe_model_mul_add.predict(
    start=Test.index[0],
    end=Test.index[-1]
)

print("Holt-Winters (Add-Mul) MAPE:",
      MAPE(pred_hwe_model_mul_add, Test.Sales))


# ============================================================
# Final Forecasting
# ------------------------------------------------------------
# Since Additive Trend + Additive Seasonality gave lowest MAPE,
# use it for future forecasting.
# ============================================================

hwe_model_add_add = ExponentialSmoothing(
    cocacola["Sales"],
    trend="add",
    seasonal="add",
    seasonal_periods=4
).fit()

# Load new dataset for future forecasting
new_data = pd.read_excel("C:/360DG/Datasets/Newdata_CocaCola_Sales.xlsx")

# Predict for new data horizon
newdata_pred = hwe_model_add_add.forecast(steps=len(new_data))

# Final Predictions
print("\nFinal Forecast for New Data:")
print(newdata_pred)
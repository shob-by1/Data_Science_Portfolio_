# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 21:58:26 2026

@author: syin
"""

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")  # to avoid harmless warnings

# ------------------------------
# 1. Load dataset and preprocess
# ------------------------------
# Read CSV, parse DATE column as datetime, and set as index
df = pd.read_csv(r"c:/24-Time Series/uspopulation.csv",
                 index_col="DATE",
                 parse_dates=True)

# Tell pandas that data is monthly (MS = Month Start)
df.index.freq = "MS"

print(df.head())

# Plot the data
df["PopEst"].plot(figsize=(12,5),
                  title="U.S. Monthly Population Estimate")
plt.ylabel("Population Estimate")
plt.show()

# ------------------------------
# 2. Train-Test Split
# ------------------------------
# First 84 months → training, last 12 months → testing
train = df.iloc[:84]
test  = df.iloc[84:]

print(len(train), len(test))   # Expected: 84, 12

# ------------------------------
# 3. Fit AR models with different lags
# ------------------------------

# AR(1)
model = AutoReg(train["PopEst"], lags=1).fit()
pred1 = model.predict(start=len(train),
                      end=len(train)+len(test)-1)

# AR(2)
model2 = AutoReg(train["PopEst"], lags=2).fit()
pred2 = model2.predict(start=len(train),
                       end=len(train)+len(test)-1)

# AR(11)
model_auto = AutoReg(train["PopEst"], lags=11).fit()
pred_auto = model_auto.predict(start=len(train),
                               end=len(train)+len(test)-1)

# ------------------------------
# 4. Compare predictions with actual values
# ------------------------------
plt.figure(figsize=(12,6))

# Plot training data
plt.plot(train.index, train["PopEst"], label="Train")

# Plot actual test values
plt.plot(test.index, test["PopEst"],
         label="Test", color="black")

# Plot predictions
plt.plot(test.index, pred1, label="AR(1) Prediction")
plt.plot(test.index, pred2, label="AR(2) Prediction")
plt.plot(test.index, pred_auto, label="AR(11) Prediction")

plt.legend()
plt.title("AR Model Forecasts")
plt.show()

# ------------------------------
# 5. Calculate MSE
# ------------------------------
print("AR(1) MSE:",
      mean_squared_error(test["PopEst"], pred1))
print("AR(2) MSE:",
      mean_squared_error(test["PopEst"], pred2))
print("AR(11) MSE:",
      mean_squared_error(test["PopEst"], pred_auto))

# ------------------------------
# 6. Forecast future population
# ------------------------------
# Retrain best model (AR(11)) on full dataset
final_model = AutoReg(df["PopEst"], lags=11).fit()

# Forecast next 12 months
forecast = final_model.predict(start=len(df),
                               end=len(df)+12)

print(forecast.head())

# Plot forecast vs historical data
plt.figure(figsize=(12,6))
plt.plot(df.index, df["PopEst"],
         label="Historical Data")
plt.plot(forecast.index, forecast,
         label="12-Month Forecast", color="red")

plt.legend()
plt.title("US Population Forecast using AR(11)")
plt.show()
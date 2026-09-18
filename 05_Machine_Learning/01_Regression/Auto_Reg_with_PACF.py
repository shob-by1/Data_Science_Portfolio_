# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 22:06:59 2026

@author: syin
"""

# ==========================================================
# AutoRegressive (AR) Model Example on US Population Dataset
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")   # ignore harmless warnings


# -------------------------------
# 1. Load dataset and preprocess
# -------------------------------
df = pd.read_csv("c:/24-Time Series/uspopulation.csv",
                 index_col="DATE",
                 parse_dates=True)

# Set monthly frequency
df.index.freq = "MS"   # Month Start frequency

print(df.head())

# Plot historical data
df["PopEst"].plot(figsize=(12,5),
                  title="U.S. Monthly Population Estimates")
plt.ylabel("Population Estimate")
plt.show()


# -------------------------------
# 2. Train-Test Split
# -------------------------------
# First 84 months for training, last 12 months for testing
train = df.iloc[:84]
test  = df.iloc[84:]

print("Train size:", len(train))
print("Test size :", len(test))


# --------------------------------
# 3. Use PACF to select AR order
# --------------------------------
plt.figure(figsize=(8,4))
plot_pacf(train["PopEst"], lags=20, method="ywm")
plt.title("PACF Plot for AR Order Selection")
plt.show()


# --------------------------------
# 4. Fit AR models with different lags
# --------------------------------

# AR(1)
model1 = AutoReg(train["PopEst"], lags=1).fit()
pred1  = model1.predict(start=len(train),
                        end=len(train)+len(test)-1,
                        dynamic=False)

# AR(2)
model2 = AutoReg(train["PopEst"], lags=2).fit()
pred2  = model2.predict(start=len(train),
                        end=len(train)+len(test)-1,
                        dynamic=False)


# --------------------------------
# 5. Compare predictions visually
# --------------------------------
plt.figure(figsize=(12,6))
plt.plot(train.index, train["PopEst"], label="Train")
plt.plot(test.index, test["PopEst"], label="Test", color="black")
plt.plot(test.index, pred1, label="AR(1) Prediction")
plt.plot(test.index, pred2, label="AR(2) Prediction")
plt.legend()
plt.title("AR Model Forecasts")
plt.show()


# --------------------------------
# 6. Evaluation (MSE)
# --------------------------------
for label, pred in zip(["AR(1)", "AR(2)"], [pred1, pred2]):
    error = mean_squared_error(test["PopEst"], pred)
    print(f"{label} MSE: {error:.2f}")


# --------------------------------
# 7. Final Forecast (Using AR(2))
# --------------------------------
final_model = AutoReg(df["PopEst"], lags=2).fit()

# Forecast next 12 months
forecast = final_model.predict(start=len(df),
                               end=len(df)+12,
                               dynamic=False)

print("\nFuture Forecast:")
print(forecast.head())


# --------------------------------
# 8. Plot Final Forecast
# --------------------------------
plt.figure(figsize=(12,6))
plt.plot(df.index, df["PopEst"], label="Historical Data")
plt.plot(forecast.index, forecast,
         label="12-Month Forecast", color="red")
plt.legend()
plt.title("US Population Forecast using AR(2)")
plt.show()
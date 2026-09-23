# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 22:55:58 2026

@author: syin
"""

# ==========================================================
# ARIMA Modeling – Walmart Footfalls Example (Complete Notes + Code)
# ==========================================================


# ----------------------------------------------------------
# What is d in ARIMA(p,d,q)?
# ----------------------------------------------------------
# d = degree of differencing.
# It tells the model how many times the time series has been differenced
# to make it stationary.
#
# Differencing means: subtract the previous value from the current value.
#
# Example with d = 1:
# If series is [100, 105, 111, 120],
# then first difference = [105-100, 111-105, 120-111] = [5, 6, 9].


# ----------------------------------------------------------
# Why is d important?
# ----------------------------------------------------------
# Stationarity requirement:
# ARIMA models assume the time series is stationary
# (constant mean, variance, autocorrelation over time).
#
# If data has a trend (upward/downward), differencing removes it.
# If variance grows with time, log + differencing stabilizes it.
#
# Controls model complexity:
# d = 0 : model raw stationary data (like stock returns)
# d = 1 : remove linear trend
# d = 2 : remove quadratic trends (rare)
#
# Too much differencing can add noise → poor forecasts.
#
# After differencing, AR(p) and MA(q) terms work on the
# “stationarized” data.


# ----------------------------------------------------------
# How to decide d?
# ----------------------------------------------------------
# Visual check:
# If data shows clear trend → likely d = 1.
#
# Statistical test:
# Augmented Dickey-Fuller (ADF) or KPSS test.
#
# If ADF p-value > 0.05 → series is non-stationary → need differencing.
# Keep differencing until p-value < 0.05.
#
# ACF shape:
# If ACF decays very slowly → need differencing.
#
# Practical guideline:
# Usually d = 0, 1, or 2.
# Start with d = 1, check residuals.
# If still non-stationary → increase d.
# If over-differenced (series oscillates around 0 with no structure) → reduce d.


# ==========================================================
# FULL ARIMA IMPLEMENTATION – WALMART FOOTFALLS
# ==========================================================

import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot


# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
Walmart = pd.read_csv("C:/360DG/Datasets/Walmart Footfalls Raw.csv")

# Data Partition (Train = first 147 observations, Test = last 12)
Train = Walmart.head(147)
Test  = Walmart.tail(12)


# ----------------------------------------------------------
# Step 1: Check Stationarity with ADF Test
# ----------------------------------------------------------
print("===== Stationarity Check (ADF Test) =====")
result = adfuller(Train.Footfalls)

print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Rule of thumb:
# If p-value > 0.05 → series is NOT stationary → apply differencing
# If p-value < 0.05 → series IS stationary → d = 0


# Try 1st order differencing
diff1 = Train.Footfalls.diff().dropna()
result1 = adfuller(diff1)
print("\nAfter 1st difference → p-value:", result1[1])

# Try 2nd order differencing if needed
diff2 = diff1.diff().dropna()
result2 = adfuller(diff2)
print("After 2nd difference → p-value:", result2[1])

# Based on the results, decide whether d = 0, 1, or 2.
# In many practical cases, d = 1 is commonly used.


# ----------------------------------------------------------
# Step 2: Plot ACF and PACF (to decide p and q)
# ----------------------------------------------------------
# ACF → suggests MA(q)
# PACF → suggests AR(p)

tsa_plots.plot_acf(Walmart.Footfalls, lags=12)
tsa_plots.plot_pacf(Walmart.Footfalls, lags=12)
pyplot.show()


# ----------------------------------------------------------
# Step 3: Fit ARIMA Model
# Example: ARIMA with AR=4, d=1, MA=6
# ----------------------------------------------------------
model1 = ARIMA(Train.Footfalls, order=(4,1,6))
res1 = model1.fit()

print("\n===== ARIMA Model Summary =====")
print(res1.summary())


# ----------------------------------------------------------
# Model Selection Criteria
# ----------------------------------------------------------
# These are penalized fit measures:
# AIC (Akaike Information Criterion)
# BIC (Bayesian Information Criterion)
# HQIC (Hannan-Quinn Information Criterion)
#
# Lower values indicate better balance between fit and complexity.

print("\nModel Selection Criteria")
print("AIC :", res1.aic)
print("BIC :", res1.bic)
print("HQIC:", res1.hqic)


# ----------------------------------------------------------
# Interpreting Coefficients
# ----------------------------------------------------------
# P>|z| = p-value
# If p-value < 0.05 → statistically significant
# If p-value > 0.05 → may not be meaningful
#
# If many coefficients are not significant,
# your model may be over-parameterized.
# Try smaller models like ARIMA(2,1,1) or ARIMA(2,1,2)
# and compare AIC/BIC.


# ----------------------------------------------------------
# Step 4: Forecast for Test Data
# ----------------------------------------------------------
start_index = len(Train)
end_index   = start_index + len(Test) - 1

forecast_test = res1.predict(start=start_index, end=end_index)

print("\nForecasted Values:")
print(forecast_test)


# ----------------------------------------------------------
# Step 5: Evaluate Forecast Accuracy
# ----------------------------------------------------------
rmse_test = sqrt(mean_squared_error(Test.Footfalls, forecast_test))
print("\nTest RMSE: %.3f" % rmse_test)


# ----------------------------------------------------------
# Step 6: Plot Actual vs Predicted
# ----------------------------------------------------------
pyplot.plot(Test.Footfalls, label="Actual")
pyplot.plot(forecast_test, color='red', label="Forecast")
pyplot.legend()
pyplot.show()
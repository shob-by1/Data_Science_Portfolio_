# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 00:40:00 2026

@author: syin
"""

# Import Libraries
# ============================================
import pandas as pd  # For handling data (CSV, DataFrame)
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.preprocessing import MinMaxScaler  # For scaling data between 0 and 1
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator  # To create time series data batches
from tensorflow.keras.models import Sequential  # Sequential model API
from tensorflow.keras.layers import Dense, LSTM  # Neural network layers
from sklearn.metrics import mean_squared_error  # To evaluate prediction performance
from math import sqrt  # For RMSE calculation

# Load Data
# ============================================
# Load the CSV file, set "Month" as index, and parse dates properly
df = pd.read_csv("c:/24-Time Series/monthly-milk-production.csv",
                 index_col="Month", parse_dates=True)

# Ensure the frequency of the index is Monthly Start ("MS")
df.index.freq = "MS"

# Rename the production column to a shorter name for convenience
df.rename(columns={"Monthly milk production (pounds per cow)": "Production"},
          inplace=True)

print(df.head())  # Display first 5 rows

# Plot the original milk production time series
df.plot(figsize=(12, 6), title="Monthly Milk Production")
plt.show()

# Train-Test Split
# ============================================
# Use first 156 months (about 13 years) as training set
train = df.iloc[:156]
# Remaining months are used as testing set
test = df.iloc[156:]

# Scale the data between 0 and 1 (LSTMs work better with scaled input)
scaler = MinMaxScaler()
scaler.fit(train)  # Fit only on training data
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Create Time Series Generator
# ============================================
# We will use the past 12 months to predict the next month
n_input = 12  # lookback window (12 months)
n_features = 1  # univariate time series (only Production column)

# TimeseriesGenerator automatically creates input-output pairs for LSTM
generator = TimeseriesGenerator(scaled_train, scaled_train,
                                 length=n_input, batch_size=1)

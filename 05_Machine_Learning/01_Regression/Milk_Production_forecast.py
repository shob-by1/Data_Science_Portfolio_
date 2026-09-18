

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from sklearn.metrics import mean_squared_error
from math import sqrt


# --------------------------------------------------
#  Load Dataset
# --------------------------------------------------
# Load CSV
# Set "Month" as index
# Parse dates properly

df = pd.read_csv(
    "c:/24-Time Series/monthly-milk-production.csv",
    index_col="Month",
    parse_dates=True
)

# Ensure frequency is Monthly Start
df.index.freq = "MS"

# Rename column for simplicity
df.rename(
    columns={"Monthly milk production (pounds per cow)": "Production"},
    inplace=True
)

print(df.head())


# --------------------------------------------------
#  Plot Original Data
# --------------------------------------------------
# Visualize seasonal pattern

df.plot(figsize=(12, 6), title="Monthly Milk Production")
plt.show()


# --------------------------------------------------
#  Train-Test Split
# --------------------------------------------------
# First 156 months (~13 years) → Training
# Remaining months → Testing

train = df.iloc[:156]
test = df.iloc[156:]


# --------------------------------------------------
#  Scale Data
# --------------------------------------------------
# LSTM performs better on scaled data
# Fit only on training data to avoid data leakage

scaler = MinMaxScaler()
scaler.fit(train)

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# --------------------------------------------------
#  Create Time Series Generator
# --------------------------------------------------
# length = n_input → Lookback window
# batch_size = 1 → one sequence processed at a time
#
# Suppose scaled_train = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# and n_input = 3
#
# Generator creates:
# [1,2,3]  → 4
# [2,3,4]  → 5
# [3,4,5]  → 6
# ...
# [12,13,14] → 15
#
# In our milk dataset:
# Given last 12 months, predict the next month.

n_input = 12
n_features = 1

generator = TimeseriesGenerator(
    scaled_train,
    scaled_train,
    length=n_input,
    batch_size=1
)


# --------------------------------------------------
#  Build LSTM Model
# --------------------------------------------------
# input_shape = (timesteps, features)
# timesteps = 12
# features = 1

model = Sequential()

model.add(LSTM(
    100,
    activation="relu",
    input_shape=(n_input, n_features)
))

# Output layer → next month's production
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")


# --------------------------------------------------
#  Train Model
# --------------------------------------------------
# Model learns:
# Given last 12 months → predict next month

model.fit(generator, epochs=50, verbose=1)


# --------------------------------------------------
#  Forecasting
# --------------------------------------------------
# scaled_train[-n_input:] → Last 12 months from training set
# Used as starting point for prediction

test_predictions = []

first_eval_batch = scaled_train[-n_input:]

# LSTM requires 3D input:
# (batch_size, timesteps, features)
# Here:
# batch_size = 1
# timesteps = 12
# features = 1

current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):

    # Predict next value
    current_pred = model.predict(current_batch, verbose=0)[0]

    # Store prediction
    test_predictions.append(current_pred)

    # Update batch:
    # Remove first value and append prediction
    current_batch = np.append(
        current_batch[:, 1:, :],
        [[current_pred]],
        axis=1
    )


# --------------------------------------------------
#  Convert Predictions Back to Original Scale
# --------------------------------------------------

true_predictions = scaler.inverse_transform(test_predictions)
test["Predictions"] = true_predictions


# --------------------------------------------------
#  Plot Results
# --------------------------------------------------

test.plot(figsize=(12, 6))
plt.show()


# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------
# RMSE gives prediction error

rmse = sqrt(mean_squared_error(
    test["Production"],
    test["Predictions"]
))

print("RMSE:", rmse)




# ------------------------------------------------------------
# ARMA, ARIMA, and SARIMA Models on Walmart Footfalls Dataset
# ------------------------------------------------------------

import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
Walmart = pd.read_csv("C:/360DG/Datasets/Walmart Footfalls Raw.csv")

# Data Partition
Train = Walmart.head(147)   # First 147 for training
Test = Walmart.tail(12)     # Last 12 for testing


# ------------------------------------------------------------
# Step 1: Stationarity Test (ADF)
# ------------------------------------------------------------
print("===== Stationarity Check (ADF Test) =====")
result = adfuller(Train.Footfalls)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Try differencing if needed
diff1 = Train.Footfalls.diff().dropna()
print("\nADF p-value after 1st difference:", adfuller(diff1)[1])


# ------------------------------------------------------------
# Step 2: ACF and PACF plots
# ------------------------------------------------------------
tsa_plots.plot_acf(Train.Footfalls, lags=20)
plt.title("ACF Plot (Suggests MA order)")
plt.show()

tsa_plots.plot_pacf(Train.Footfalls, lags=20)
plt.title("PACF Plot (Suggests AR order)")
plt.show()


# ------------------------------------------------------------
# Step 3a: Fit ARMA Model (when d=0, stationary)
# Example: ARMA(2,1)
# ------------------------------------------------------------
arma_model = ARIMA(Train.Footfalls, order=(2,0,1)).fit()
print("\n===== ARMA(2,1) Summary =====")
print(arma_model.summary())

print("AIC =", arma_model.aic)
print("BIC =", arma_model.bic)


# ------------------------------------------------------------
# Step 3b: Fit ARIMA Model (when d>0)
# Example: ARIMA(2,1,2)
# ------------------------------------------------------------
arima_model = ARIMA(Train.Footfalls, order=(2,1,2)).fit()
print("\n===== ARIMA(2,1,2) Summary =====")
print(arima_model.summary())

print("AIC =", arima_model.aic)
print("BIC =", arima_model.bic)


# ------------------------------------------------------------
# Step 3c: Fit SARIMA Model (with seasonality)
# Assume monthly seasonality (s=12)
# ------------------------------------------------------------
sarima_model = SARIMAX(
    Train.Footfalls,
    order=(2,1,2),              # Non-seasonal part
    seasonal_order=(1,1,1,12)   # (P,D,Q,s)
).fit()

print("\n===== SARIMA(2,1,2)(1,1,1,12) Summary =====")
print(sarima_model.summary())

print("Log Likelihood =", sarima_model.llf)
print("AIC =", sarima_model.aic)
print("BIC =", sarima_model.bic)


# ------------------------------------------------------------
# Step 4: Forecast on Test Data
# ------------------------------------------------------------
start = len(Train)
end = start + len(Test) - 1

forecast_arma = arma_model.predict(start=start, end=end)
forecast_arima = arima_model.predict(start=start, end=end)
forecast_sarima = sarima_model.predict(start=start, end=end)


# ------------------------------------------------------------
# Step 5: Evaluate RMSE
# ------------------------------------------------------------
print("\nForecast RMSEs:")
print("ARMA(2,1): %.3f" % sqrt(mean_squared_error(Test.Footfalls, forecast_arma)))
print("ARIMA(2,1,2): %.3f" % sqrt(mean_squared_error(Test.Footfalls, forecast_arima)))
print("SARIMA(2,1,2)(1,1,1,12): %.3f" % sqrt(mean_squared_error(Test.Footfalls, forecast_sarima)))


# ------------------------------------------------------------
# Step 6: Plot Comparison
# ------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(Test.Footfalls.values, label="Actual", color="black")
plt.plot(forecast_arma.values, label="ARMA Forecast")
plt.plot(forecast_arima.values, label="ARIMA Forecast")
plt.plot(forecast_sarima.values, label="SARIMA Forecast")
plt.legend()
plt.title("Forecast Comparison")
plt.show()
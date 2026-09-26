#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Multi-Linear Regression - House Prices (Advanced Regression, Kaggle)
# Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
# (train.csv)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[ ]:


df = pd.read_csv("../Datasets/house_prices_train.csv")

# ---- 1. First look ----
print(df.shape)
print(df.head())

# this dataset has 80 columns - a mix of numeric and categorical, lots of missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
print(missing_pct[missing_pct > 0].sort_values(ascending=False).head(20))

# ---- 2. Drop columns with too much missing data ----
# PoolQC, MiscFeature, Alley, Fence are usually >80% missing - not worth keeping
high_missing_cols = missing_pct[missing_pct > 40].index.tolist()
print("Dropping columns with >40% missing:", high_missing_cols)
df = df.drop(columns=high_missing_cols)



# In[ ]:


# ---- 3. Keep it manageable - pick a focused set of strong numeric predictors ----
# rather than encode all 70+ remaining columns for a "multi-linear regression" example,
# picking the numeric features most people find matter most for price
selected_features = [
    "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
    "FullBath", "YearBuilt", "YearRemodAdd", "Fireplaces",
    "LotArea", "1stFlrSF"
]
df_model = df[selected_features + ["SalePrice"]].copy()

print(df_model.isnull().sum())   # these numeric columns are mostly complete already

# fill any remaining numeric gaps with median
for col in selected_features:
    df_model[col] = df_model[col].fillna(df_model[col].median())

# ---- 4. Check correlation with target ----
corr = df_model.corr()["SalePrice"].sort_values(ascending=False)
print(corr)

plt.figure(figsize=(8, 8))
sns.heatmap(df_model.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap - Selected Features")
plt.show()


# In[ ]:


# ---- 5. Check target distribution ----
plt.hist(df_model["SalePrice"], bins=40)
plt.title("SalePrice Distribution")
plt.show()

# SalePrice is right-skewed - log transform often helps linear regression here
df_model["SalePrice_log"] = np.log1p(df_model["SalePrice"])

plt.hist(df_model["SalePrice_log"], bins=40)
plt.title("Log-Transformed SalePrice Distribution")
plt.show()

# ---- 6. Split features/target ----
X = df_model[selected_features]
y = df_model["SalePrice_log"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- 7. Scale features ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


# ---- 8. Fit multi-linear regression ----
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ---- 9. Predict and evaluate (in log space first) ----
y_pred_log = model.predict(X_test_scaled)

print("R2 (log space):", r2_score(y_test, y_pred_log))
print("RMSE (log space):", np.sqrt(mean_squared_error(y_test, y_pred_log)))

# ---- 10. Convert back to actual dollar scale to evaluate in real terms ----
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_log)

print("MAE (actual $):", mean_absolute_error(y_test_actual, y_pred_actual))
print("RMSE (actual $):", np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)))
print("R2 (actual $):", r2_score(y_test_actual, y_pred_actual))

# ---- 11. Coefficients - which features push price up/down most ----
coef_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print(coef_df)


# In[ ]:


# OverallQual and GrLivArea are usually the strongest positive predictors

# ---- 12. Actual vs predicted plot ----
plt.figure(figsize=(7, 6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.4)
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()], "r--")
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted - House Prices")
plt.show()


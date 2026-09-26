#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Lasso & Ridge Regression (Regularization)
# Dataset: House Prices - Advanced Regression (Kaggle, same as multi-linear regression)
# ../Datasets/house_prices_train.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#df = pd.read_csv("../Datasets/house_prices_train.csv")

from sklearn.datasets import fetch_openml


# In[2]:


housing = fetch_openml(name="house_prices", as_frame=True, parser="auto")
df = housing.frame

# ---- 1. Same setup as the multi-linear regression script ----
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
high_missing_cols = missing_pct[missing_pct > 40].index.tolist()
df = df.drop(columns=high_missing_cols)

# this time, use a WIDER set of numeric features on purpose - this is exactly
# the situation where Lasso/Ridge earn their keep over plain linear regression
# (more features -> more risk of overfitting / multicollinearity)
numeric_features = [
    "OverallQual", "OverallCond", "GrLivArea", "GarageCars", "GarageArea",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "FullBath", "HalfBath",
    "BedroomAbvGr", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd",
    "Fireplaces", "LotArea", "LotFrontage", "MasVnrArea", "WoodDeckSF",
    "OpenPorchSF", "BsmtFinSF1"
]
numeric_features = [c for c in numeric_features if c in df.columns]

df_model = df[numeric_features + ["SalePrice"]].copy()

for col in numeric_features:
    df_model[col] = df_model[col].fillna(df_model[col].median())

df_model["SalePrice_log"] = np.log1p(df_model["SalePrice"])

X = df_model[numeric_features]
y = df_model["SalePrice_log"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- 2. Scale - REQUIRED for Lasso/Ridge, since regularization penalizes
#          coefficient size, and that's meaningless if features aren't on the same scale ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[3]:


# ---- 3. Baseline: plain Linear Regression ----
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("Linear Regression coefficients (min/max):", lr.coef_.min(), lr.coef_.max())

# ---- 4. Ridge Regression (L2 penalty - shrinks coefficients, keeps all features) ----
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
print("\nRidge R2:", r2_score(y_test, ridge_pred))
print("Ridge coefficients (min/max):", ridge.coef_.min(), ridge.coef_.max())

# ---- 5. Lasso Regression (L1 penalty - can shrink some coefficients to exactly 0) ----
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)
print("\nLasso R2:", r2_score(y_test, lasso_pred))

n_zeroed = (lasso.coef_ == 0).sum()
print(f"Lasso zeroed out {n_zeroed} of {len(numeric_features)} features")
# this is the key practical difference from Ridge - Lasso can do feature
# selection automatically by dropping features entirely


# In[4]:


# ---- 6. Compare coefficients side by side ----
coef_compare = pd.DataFrame({
    "Feature": numeric_features,
    "Linear": lr.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_
})
print(coef_compare.sort_values("Linear", key=abs, ascending=False))

# ---- 7. Try a range of alpha values to see the effect of regularization strength ----
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_scores = []
lasso_scores = []

for a in alphas:
    r = Ridge(alpha=a).fit(X_train_scaled, y_train)
    l = Lasso(alpha=a, max_iter=5000).fit(X_train_scaled, y_train)
    ridge_scores.append(r2_score(y_test, r.predict(X_test_scaled)))
    lasso_scores.append(r2_score(y_test, l.predict(X_test_scaled)))

plt.plot(alphas, ridge_scores, marker="o", label="Ridge")
plt.plot(alphas, lasso_scores, marker="o", label="Lasso")
plt.xscale("log")
plt.xlabel("alpha (regularization strength)")
plt.ylabel("R2 Score")
plt.title("Effect of Regularization Strength")
plt.legend()
plt.show()


# In[5]:


# too small an alpha barely regularizes at all (looks like plain linear regression)
# too large an alpha over-penalizes and underfits - there's a sweet spot in between

# ---- 8. Model comparison summary ----
results = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge", "Lasso"],
    "R2": [r2_score(y_test, lr_pred), r2_score(y_test, ridge_pred), r2_score(y_test, lasso_pred)],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, lr_pred)),
        np.sqrt(mean_squared_error(y_test, ridge_pred)),
        np.sqrt(mean_squared_error(y_test, lasso_pred)),
    ]
})
print(results)


# In[ ]:





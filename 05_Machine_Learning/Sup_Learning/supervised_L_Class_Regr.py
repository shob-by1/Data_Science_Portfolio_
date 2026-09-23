#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Supervised Learning - General Overview (Classification + Regression)
# Datasets:
#   - Titanic: ../Datasets/Titanic-Dataset.csv  (classification example)
#   - California Housing: built into sklearn (regression example)
#
# Goal here isn't to build the best model - just to see the full supervised
# learning workflow end to end: train/test split -> fit -> predict -> evaluate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.datasets import fetch_california_housing


# In[3]:


# ============================================================
# PART 1 - Classification (Titanic - predict Survived)
# ============================================================

df = pd.read_csv(r"C:\My-Data-Science-Portfolio\Datasets\Titanic-Dataset.csv")
#df = pd.read_csv("../Datasets/Titanic-Dataset.csv")
# ---- 1. Minimal cleaning (reusing logic from the preprocessing script) ----
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
            "Embarked_Q", "Embarked_S"]
X = df[features]
y = df["Survived"]

# ---- 2. Train/test split ----
# this is the core idea of supervised learning - train on one part,
# test on unseen data to see how well it actually generalizes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# ---- 3. Scale features (logistic regression is distance/gradient based) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # transform only, never fit on test data

# ---- 4. Fit the model ----
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# ---- 5. Predict on test set ----
y_pred = clf.predict(X_test_scaled)

# ---- 6. Evaluate ----
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---- 7. Look at which features mattered most ----
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": clf.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False)
print(coef_df)

# Sex and Pclass usually end up as the strongest predictors of survival


# In[4]:


# ============================================================
# PART 2 - Regression (California Housing - predict median house value)
# ============================================================

housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame

print(housing_df.shape)
print(housing_df.head())
print(housing_df.isnull().sum())   # this dataset is already clean

X_house = housing_df.drop(columns=["MedHouseVal"])
y_house = housing_df["MedHouseVal"]

# ---- 8. Train/test split ----
Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# ---- 9. Scale features ----
scaler2 = StandardScaler()
Xh_train_scaled = scaler2.fit_transform(Xh_train)
Xh_test_scaled = scaler2.transform(Xh_test)

# ---- 10. Fit a regression model ----
reg = LinearRegression()
reg.fit(Xh_train_scaled, yh_train)

# ---- 11. Predict and evaluate ----
yh_pred = reg.predict(Xh_test_scaled)

print("MAE:", mean_absolute_error(yh_test, yh_pred))
print("RMSE:", np.sqrt(mean_squared_error(yh_test, yh_pred)))
print("R2 Score:", r2_score(yh_test, yh_pred))

# ---- 12. Actual vs predicted plot ----
plt.figure(figsize=(7, 6))
plt.scatter(yh_test, yh_pred, alpha=0.3)
plt.plot([yh_test.min(), yh_test.max()], [yh_test.min(), yh_test.max()], "r--")
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted - California Housing")
plt.show()

# ---- 13. Feature importance (coefficients) ----
coef_df2 = pd.DataFrame({
    "Feature": X_house.columns,
    "Coefficient": reg.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print(coef_df2)


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Simple Linear Regression - one feature predicting one target
# Datasets:
#   - Salary vs Experience (Kaggle): ../Datasets/Salary_Data.csv
#   - California Housing (single feature): built into sklearn
#
# Note: the classic "Boston Housing" dataset was removed from sklearn (from
# version 1.2 onward) due to ethical concerns about one of its features.
# California Housing is the standard modern replacement, so that's used here
# instead, picking just one feature to keep this "simple" (not multi-) regression.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# ============================================================
# PART 1 - Salary vs Years of Experience
# ============================================================

df = pd.read_csv("../Datasets/Salary_Data.csv")

print(df.shape)
print(df.head())
print(df.isnull().sum())

# ---- 1. Quick look at the relationship ----
plt.scatter(df["YearsExperience"], df["Salary"])
plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# looks like a pretty clean linear relationship - good candidate for simple linear regression

# ---- 2. Split into X (single feature) and y ----
X = df[["YearsExperience"]]   # double brackets - sklearn wants a 2D array
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- 3. Fit the model ----
# no scaling needed for a single feature - coefficient is directly interpretable
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
# interpretation: each extra year of experience is associated with this much
# more salary, on average, holding everything else in the (simple) model equal

# ---- 4. Predict and evaluate ----
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# ---- 5. Plot the regression line against actual data ----
plt.scatter(X_train, y_train, color="blue", label="Train data")
plt.scatter(X_test, y_test, color="green", label="Test data")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.title("Simple Linear Regression - Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

# ---- 6. Try predicting a new value ----
new_experience = pd.DataFrame({"YearsExperience": [5]})
predicted_salary = model.predict(new_experience)
print(f"Predicted salary for 5 years experience: {predicted_salary[0]:.2f}")


# ============================================================
# PART 2 - California Housing (single feature: median income)
# ============================================================

housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame

print(housing_df.head())

# MedInc (median income) is usually the strongest single predictor here
X_house = housing_df[["MedInc"]]
y_house = housing_df["MedHouseVal"]

plt.scatter(X_house, y_house, alpha=0.2)
plt.title("Median House Value vs Median Income")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.show()

# ---- 7. Train/test split and fit ----
Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

model2 = LinearRegression()
model2.fit(Xh_train, yh_train)

print("Intercept:", model2.intercept_)
print("Coefficient:", model2.coef_[0])

# ---- 8. Predict and evaluate ----
yh_pred = model2.predict(Xh_test)

print("MAE:", mean_absolute_error(yh_test, yh_pred))
print("RMSE:", np.sqrt(mean_squared_error(yh_test, yh_pred)))
print("R2 Score:", r2_score(yh_test, yh_pred))

# ---- 9. Plot regression line ----
plt.scatter(Xh_test, yh_test, alpha=0.2, label="Actual")
plt.plot(Xh_test.sort_values("MedInc"), model2.predict(Xh_test.sort_values("MedInc")),
         color="red", label="Regression line")
plt.title("Simple Linear Regression - California Housing (MedInc only)")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.legend()
plt.show()

# R2 here will be noticeably lower than the Salary example - real-world,
# noisier data with only one feature rarely fits this cleanly


# In[1]:


# Simple Linear Regression - one feature predicting one target
# Datasets:
#   - Salary vs Experience (Kaggle): ../Datasets/Salary_Data.csv
#   - California Housing (single feature): built into sklearn
#
# Note: the classic "Boston Housing" dataset was removed from sklearn (from
# version 1.2 onward) due to ethical concerns about one of its features.
# California Housing is the standard modern replacement, so that's used here
# instead, picking just one feature to keep this "simple" (not multi-) regression.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing


# In[2]:


# ============================================================
# PART 1 - Salary vs Years of Experience
# ============================================================

df = pd.read_csv(r"C:\Users\syin\Downloads\archive (38)\Salary_dataset.csv")

print(df.shape)
print(df.head())
print(df.isnull().sum())

# ---- 1. Quick look at the relationship ----
plt.scatter(df["YearsExperience"], df["Salary"])
plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# looks like a pretty clean linear relationship - good candidate for simple linear regression

# ---- 2. Split into X (single feature) and y ----
X = df[["YearsExperience"]]   # double brackets - sklearn wants a 2D array
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- 3. Fit the model ----
# no scaling needed for a single feature - coefficient is directly interpretable
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
# interpretation: each extra year of experience is associated with this much
# more salary, on average, holding everything else in the (simple) model equal

# ---- 4. Predict and evaluate ----
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# ---- 5. Plot the regression line against actual data ----
plt.scatter(X_train, y_train, color="blue", label="Train data")
plt.scatter(X_test, y_test, color="green", label="Test data")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.title("Simple Linear Regression - Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

# ---- 6. Try predicting a new value ----
new_experience = pd.DataFrame({"YearsExperience": [5]})
predicted_salary = model.predict(new_experience)
print(f"Predicted salary for 5 years experience: {predicted_salary[0]:.2f}")


# In[3]:


# ============================================================
# PART 2 - California Housing (single feature: median income)
# ============================================================

housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame

print(housing_df.head())

# MedInc (median income) is usually the strongest single predictor here
X_house = housing_df[["MedInc"]]
y_house = housing_df["MedHouseVal"]

plt.scatter(X_house, y_house, alpha=0.2)
plt.title("Median House Value vs Median Income")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.show()

# ---- 7. Train/test split and fit ----
Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

model2 = LinearRegression()
model2.fit(Xh_train, yh_train)

print("Intercept:", model2.intercept_)
print("Coefficient:", model2.coef_[0])

# ---- 8. Predict and evaluate ----
yh_pred = model2.predict(Xh_test)

print("MAE:", mean_absolute_error(yh_test, yh_pred))
print("RMSE:", np.sqrt(mean_squared_error(yh_test, yh_pred)))
print("R2 Score:", r2_score(yh_test, yh_pred))

# ---- 9. Plot regression line ----
plt.scatter(Xh_test, yh_test, alpha=0.2, label="Actual")
plt.plot(Xh_test.sort_values("MedInc"), model2.predict(Xh_test.sort_values("MedInc")),
         color="red", label="Regression line")
plt.title("Simple Linear Regression - California Housing (MedInc only)")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.legend()
plt.show()

# R2 here will be noticeably lower than the Salary example - real-world,
# noisier data with only one feature rarely fits this cleanly


# In[ ]:





# In[ ]:





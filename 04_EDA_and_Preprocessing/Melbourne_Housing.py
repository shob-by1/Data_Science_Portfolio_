#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Melbourne Housing - Data Preprocessing & Feature Engineering
# Dataset: https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv(r"C:\Users\syin\Downloads\archive (33)\Melbourne_housing_FULL.csv")



# In[2]:


# ---- 1. First look ----
print(df.shape)
print(df.info())


# In[3]:


# ---- 2. Check missing values ----
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# Melbourne housing usually has a lot of missing values in:
# BuildingArea, YearBuilt, Car, CouncilArea, Landsize


# In[4]:


# ---- 3. Drop columns with too much missing data ----
# BuildingArea and YearBuilt are often >40-50% missing - not worth imputing heavily
df = df.drop(columns=["BuildingArea", "YearBuilt"])



# In[5]:


# ---- 4. Handle remaining missing values ----

# Car (number of car spots) - numeric, fill with median
df["Car"] = df["Car"].fillna(df["Car"].median())

# Landsize - numeric, fill with median (has outliers/zeros too, median is safer than mean)
df["Landsize"] = df["Landsize"].fillna(df["Landsize"].median())

# CouncilArea - categorical, fill with mode
df["CouncilArea"] = df["CouncilArea"].fillna(df["CouncilArea"].mode()[0])

# Price is the target - can't predict/train on rows with missing target, so drop those rows
df = df.dropna(subset=["Price"])

print(df.isnull().sum().sum(), "missing values remaining")


# In[6]:


# ---- 5. Drop columns that aren't useful for modelling ----
# Address is free text/unique per row, Date isn't needed in raw form yet
df = df.drop(columns=["Address"])

# ---- 6. Feature engineering from Date ----
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["SaleYear"] = df["Date"].dt.year
df["SaleMonth"] = df["Date"].dt.month
df = df.drop(columns=["Date"])

# ---- 7. Property age at time of sale would've been nice, but we dropped YearBuilt
#          so skipping that here - just noting this is a common feature to add
#          when YearBuilt isn't heavily missing


# In[7]:


# ---- 8. Categorical encoding ----

# Type (h = house, u = unit, t = townhouse) - few categories, no order -> one-hot
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

# Method (sale method: S, SP, PI, VB, etc.) - one-hot as well
df = pd.get_dummies(df, columns=["Method"], drop_first=True)

# Regionname - one-hot
df = pd.get_dummies(df, columns=["Regionname"], drop_first=True)

# Suburb and CouncilArea have too many unique categories for one-hot
# (would create hundreds of columns) - use label/frequency encoding instead
suburb_counts = df["Suburb"].value_counts()
df["Suburb_freq"] = df["Suburb"].map(suburb_counts)
df = df.drop(columns=["Suburb"])

council_counts = df["CouncilArea"].value_counts()
df["CouncilArea_freq"] = df["CouncilArea"].map(council_counts)
df = df.drop(columns=["CouncilArea"])

# SellerG has many unique agents too - same approach
sellerg_counts = df["SellerG"].value_counts()
df["SellerG_freq"] = df["SellerG"].map(sellerg_counts)
df = df.drop(columns=["SellerG"])

print(df.head())
print(df.shape)


# In[8]:


# ---- 9. Outlier check on Price (target) ----
print(df["Price"].describe())

# cap extreme outliers using IQR method instead of dropping (keeps row count)
Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df["Price"] = df["Price"].clip(lower=lower, upper=upper)



# In[9]:


# ---- 10. Scale numeric features ----
numeric_cols = ["Rooms", "Distance", "Landsize", "Car", "Lattitude", "Longtitude"]
numeric_cols = [c for c in numeric_cols if c in df.columns]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df.head())




# In[10]:


# ---- 11. Final check ----
print(df.shape)
print(df.isnull().sum().sum(), "missing values remaining")

# ---- 12. Save cleaned dataset ----
df.to_csv("melbourne_housing_processed.csv", index=False)
print("Saved -> melbourne_housing_processed.csv")


# In[ ]:





# In[ ]:





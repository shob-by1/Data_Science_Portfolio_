#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Titanic - Data Preprocessing & Feature Engineering
# Dataset: https://www.kaggle.com/c/titanic/data (train.csv)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv(r"C:\Users\syin\Downloads\Titanic_dataset\Titanic-Dataset.csv")


# In[3]:


# ---- 1. Check missing values first ----
print(df.isnull().sum())

# Age       -> ~20% missing, numeric -> fill with median
# Cabin     -> ~77% missing, too many to fill -> drop column
# Embarked  -> only 2 missing, categorical -> fill 


# In[4]:


# ---- 2. Handle missing values ----

# Age: fill with median (not mean, since age can be skewed by outliers)
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked: fill with the most common port
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin: too many missing, drop the column
df = df.drop(columns=["Cabin"])

print(df.isnull().sum())   # should be all 0 now


# In[5]:


# ---- 3. Drop columns that aren't useful as-is ----
# Name and Ticket are free text, not usable directly without more work
# PassengerId is just a row index
df = df.drop(columns=["Name", "Ticket", "PassengerId"])


# In[6]:


# ---- 4. Categorical encoding ----

# Sex: only 2 categories -> simple binary mapping
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Embarked: 3 categories, no natural order -> one-hot encoding
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

print(df.head())
print(df.dtypes)


# In[7]:


# ---- 5. Feature engineering ----

# Family size from SibSp (siblings/spouses) + Parch (parents/children)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# IsAlone flag - travelling solo often affected survival chances
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Bin Age into groups - sometimes ranges work better than raw age
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"]
)
df = pd.get_dummies(df, columns=["AgeGroup"], drop_first=True)

# Bin Fare into quartiles - fare is heavily skewed, raw values aren't great for some models
df["FareBin"] = pd.qcut(df["Fare"], 4, labels=["Low", "Mid", "High", "VeryHigh"])
df = pd.get_dummies(df, columns=["FareBin"], drop_first=True)

print(df.head())


# In[8]:


# ---- 6. Scaling numeric features ----
# Needed for models like KNN, SVM, Logistic Regression - not needed for trees
scaler = StandardScaler()
numeric_cols = ["Age", "Fare", "FamilySize"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df.head())


# In[9]:


# ---- 7. Final check ----
print(df.shape)
print(df.isnull().sum().sum(), "missing values remaining")


# In[10]:


# ---- 8. Save cleaned dataset ----
df.to_csv("titanic_processed.csv", index=False)
print("Saved -> titanic_processed.csv")



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





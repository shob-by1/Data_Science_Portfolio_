#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Titanic EDA - basic exploration
# Dataset: https://www.kaggle.com/c/titanic/data (train.csv)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


df = pd.read_csv(r"C:\Users\syin\Downloads\Titanic_dataset\Titanic-Dataset.csv")


# In[14]:


# ---- 1. First look at the data ----
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())


# In[15]:


# ---- 2. Check missing values ----
print(df.isnull().sum())


# In[16]:


# Age, Cabin, Embarked have missing values
# Cabin has too many missing values, might drop it later

# ---- 3. Target variable ----
print(df["Survived"].value_counts())

sns.countplot(x="Survived", data=df)
plt.title("Survived vs Not Survived")
plt.show()


# In[17]:


# ---- 4. Survival by sex ----
print(df.groupby("Sex")["Survived"].mean())

sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Sex")
plt.show()

# females survived way more than males


# In[18]:


# ---- 5. Survival by passenger class ----
print(df.groupby("Pclass")["Survived"].mean())

sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

# 1st class survived more than 3rd class, makes sense


# In[19]:


# ---- 6. Age distribution ----
plt.hist(df["Age"].dropna(), bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()


# In[20]:


# ---- 7. Age vs survival ----
sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True)
plt.title("Age Distribution by Survival")
plt.show()


# In[21]:


# ---- 8. Fare distribution ----
plt.hist(df["Fare"], bins=30)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.show()

# fare is very skewed, few people paid a lot


# In[22]:


# ---- 9. Fare vs survival ----
sns.boxplot(x="Survived", y="Fare", data=df)
plt.title("Fare vs Survival")
plt.show()


# In[23]:


# ---- 10. Family size (SibSp + Parch) ----
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

sns.countplot(x="FamilySize", hue="Survived", data=df)
plt.title("Survival by Family Size")
plt.show()


# In[24]:


# ---- 11. Correlation heatmap ----
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pclass and Fare have decent correlation with Survived


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





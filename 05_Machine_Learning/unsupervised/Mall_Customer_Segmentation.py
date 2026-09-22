#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Mall Customer Segmentation - Unsupervised Learning Overview
# Dataset: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
# (Mall_Customers.csv)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[2]:


#"C:\Users\syin\Downloads\archive (34)\Mall_Customers.csv"
df = pd.read_csv(r"C:\Users\syin\Downloads\archive (34)\Mall_Customers.csv")
#df = pd.read_csv("../Datasets/Mall_Customers.csv")

# ---- 1. First look ----
print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())   # this dataset is usually clean, no missing values

# ---- 2. Rename columns for convenience ----
df = df.rename(columns={
    "Annual Income (k$)": "AnnualIncome",
    "Spending Score (1-100)": "SpendingScore"
})


# In[3]:


# ---- 3. Quick EDA before clustering ----
sns.pairplot(df[["Age", "AnnualIncome", "SpendingScore"]])
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.show()

# ---- 4. Select features for unsupervised learning ----
# no target/label here - that's the whole point of unsupervised learning
X = df[["AnnualIncome", "SpendingScore"]]


# In[4]:


# ---- 5. Scale features ----
# distance-based algorithms (KMeans, hierarchical, DBSCAN) need scaled inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- 6. Find a good number of clusters (Elbow Method) ----
inertia = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(k_range, inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# elbow usually shows up around k=5 for this dataset


# In[5]:


# ---- 7. Fit KMeans with chosen k ----
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(df["Cluster"].value_counts())

# ---- 8. Visualize the clusters ----
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x="AnnualIncome", y="SpendingScore",
    hue="Cluster", palette="tab10", s=80
)
plt.title(f"Customer Segments (k={k})")
plt.show()


# In[ ]:


# ---- 9. Look at cluster profiles ----
# this is the actual business value of unsupervised learning here -
# understanding WHAT each cluster represents
cluster_summary = df.groupby("Cluster")[["Age", "AnnualIncome", "SpendingScore"]].mean()
print(cluster_summary)

# e.g. one cluster = high income, low spending (careful spenders)
# another = low income, high spending (potential risk / impulsive buyers)
# useful for targeted marketing

# ---- 10. Save segmented dataset ----
df.to_csv("mall_customers_segmented.csv", index=False)
print("Saved -> mall_customers_segmented.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





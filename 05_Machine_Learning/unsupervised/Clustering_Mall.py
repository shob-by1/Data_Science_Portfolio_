#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"C:\Users\syin\Downloads\archive (35)\Mall_Customers.csv"


# In[ ]:


# Clustering Techniques - KMeans, Hierarchical, DBSCAN
# Datasets:
#   - Mall Customers: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
#   - Iris: built into sklearn, no download needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score



# In[ ]:


# ============================================================
# PART 1 - Mall Customers (KMeans, Hierarchical, DBSCAN compared)
# ============================================================
df = pd.read_csv(r"C:\Users\syin\Downloads\archive (35)\Mall_Customers.csv")
#df = pd.read_csv("../Datasets/Mall_Customers.csv")
df = df.rename(columns={
    "Annual Income (k$)": "AnnualIncome",
    "Spending Score (1-100)": "SpendingScore"
})

X = df[["AnnualIncome", "SpendingScore"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- 1. KMeans ----
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="AnnualIncome", y="SpendingScore", hue="KMeans_Cluster", palette="tab10")
plt.title("KMeans Clustering")
plt.show()

# ---- 2. Hierarchical Clustering ----
# dendrogram helps visually decide how many clusters make sense
plt.figure(figsize=(10, 5))
linked = linkage(X_scaled, method="ward")
dendrogram(linked)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=5)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="AnnualIncome", y="SpendingScore", hue="Hierarchical_Cluster", palette="tab10")
plt.title("Hierarchical Clustering")
plt.show()

# ---- 3. DBSCAN ----
# density-based - doesn't need you to specify number of clusters upfront
# but IS sensitive to eps and min_samples, usually needs some tuning
dbscan = DBSCAN(eps=0.4, min_samples=5)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

print(df["DBSCAN_Cluster"].value_counts())
# -1 means "noise" / outlier points that don't belong to any cluster

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="AnnualIncome", y="SpendingScore", hue="DBSCAN_Cluster", palette="tab10")
plt.title("DBSCAN Clustering")
plt.show()

# ---- 4. Compare with silhouette score ----
# higher is better (closer to 1 = well separated clusters)
print("KMeans silhouette:", silhouette_score(X_scaled, df["KMeans_Cluster"]))
print("Hierarchical silhouette:", silhouette_score(X_scaled, df["Hierarchical_Cluster"]))

# skip DBSCAN silhouette if it only found 1 cluster or all noise
if df["DBSCAN_Cluster"].nunique() > 1:
    print("DBSCAN silhouette:", silhouette_score(X_scaled, df["DBSCAN_Cluster"]))



# In[ ]:


# ============================================================
# PART 2 - Iris dataset (built into sklearn)
# ============================================================

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["species"] = iris.target   # 0, 1, 2 - true labels, only used here to CHECK clustering quality

print(iris_df.head())

X_iris = iris_df.drop(columns=["species"])
X_iris_scaled = StandardScaler().fit_transform(X_iris)

# ---- 5. KMeans on Iris ----
kmeans_iris = KMeans(n_clusters=3, random_state=42, n_init=10)
iris_df["Cluster"] = kmeans_iris.fit_predict(X_iris_scaled)

# compare clusters found vs actual species (just for learning purposes -
# in real unsupervised problems you won't have true labels to check against)
print(pd.crosstab(iris_df["species"], iris_df["Cluster"]))

# ---- 6. Visualize using two features ----
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=iris_df, x="petal length (cm)", y="petal width (cm)",
    hue="Cluster", style="species", palette="tab10"
)
plt.title("KMeans Clusters vs True Species (Iris)")
plt.show()

print("Silhouette score (Iris KMeans):", silhouette_score(X_iris_scaled, iris_df["Cluster"]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





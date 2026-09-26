#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sklearn
print(sklearn.__version__)


# In[ ]:


# Multinomial (Multi-class) Logistic Regression
# Datasets (both built into sklearn, no download needed):
#   - Iris: 3 classes of flower species
#   - Wine: 3 classes of wine cultivar
#
# Note: written for scikit-learn 1.9.1 - the multi_class="multinomial" parameter
# was deprecated in 1.5 and removed entirely in 1.7+. Modern LogisticRegression
# automatically handles multi-class problems correctly without it.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris, load_wine

# ============================================================
# PART 1 - Iris (3-class classification)
# ============================================================

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["species"] = iris.target

print(iris_df["species"].value_counts())   # nicely balanced - 50 of each class

X = iris_df.drop(columns=["species"])
y = iris_df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- 1. Fit multinomial logistic regression ----
# no multi_class param needed anymore - sklearn picks the right multi-class
# strategy automatically based on the solver
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# ---- 2. Predict and evaluate ----
y_pred = clf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# ---- 3. Look at predicted probabilities for a few samples ----
proba = clf.predict_proba(X_test_scaled)
proba_df = pd.DataFrame(proba, columns=iris.target_names)
print(proba_df.head())

# each row sums to 1 - the model's confidence split across all 3 classes

# ---- 4. Confusion matrix as a heatmap ----
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix - Iris")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# PART 2 - Wine (3-class classification)
# ============================================================

wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df["target"] = wine.target

print(wine_df["target"].value_counts())

Xw = wine_df.drop(columns=["target"])
yw = wine_df["target"]

Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    Xw, yw, test_size=0.2, random_state=42, stratify=yw
)

# ---- 5. Scale (wine features are on very different scales - alcohol % vs proline etc.) ----
scaler2 = StandardScaler()
Xw_train_scaled = scaler2.fit_transform(Xw_train)
Xw_test_scaled = scaler2.transform(Xw_test)

# ---- 6. Fit and evaluate ----
clf2 = LogisticRegression(max_iter=1000, random_state=42)
clf2.fit(Xw_train_scaled, yw_train)

yw_pred = clf2.predict(Xw_test_scaled)

print("Accuracy:", accuracy_score(yw_test, yw_pred))
print("\nConfusion Matrix:\n", confusion_matrix(yw_test, yw_pred))
print("\nClassification Report:\n", classification_report(yw_test, yw_pred, target_names=wine.target_names))

# ---- 7. Coefficients per class ----
# each row here corresponds to one class' set of coefficients
coef_df = pd.DataFrame(clf2.coef_, columns=wine.feature_names, index=wine.target_names)
print(coef_df.T)


# In[ ]:


# ============================================================
# PART 1 - Iris (3-class classification)
# ============================================================

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["species"] = iris.target

print(iris_df["species"].value_counts())   # nicely balanced - 50 of each class

X = iris_df.drop(columns=["species"])
y = iris_df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- 1. Fit multinomial logistic regression ----
# no multi_class param needed anymore - sklearn picks the right multi-class
# strategy automatically based on the solver
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# ---- 2. Predict and evaluate ----
y_pred = clf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# ---- 3. Look at predicted probabilities for a few samples ----
proba = clf.predict_proba(X_test_scaled)
proba_df = pd.DataFrame(proba, columns=iris.target_names)
print(proba_df.head())

# each row sums to 1 - the model's confidence split across all 3 classes

# ---- 4. Confusion matrix as a heatmap ----
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix - Iris")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:


# ============================================================
# PART 2 - Wine (3-class classification)
# ============================================================

wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df["target"] = wine.target

print(wine_df["target"].value_counts())

Xw = wine_df.drop(columns=["target"])
yw = wine_df["target"]

Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    Xw, yw, test_size=0.2, random_state=42, stratify=yw
)

# ---- 5. Scale (wine features are on very different scales - alcohol % vs proline etc.) ----
scaler2 = StandardScaler()
Xw_train_scaled = scaler2.fit_transform(Xw_train)
Xw_test_scaled = scaler2.transform(Xw_test)

# ---- 6. Fit and evaluate ----
clf2 = LogisticRegression(max_iter=1000, random_state=42)
clf2.fit(Xw_train_scaled, yw_train)

yw_pred = clf2.predict(Xw_test_scaled)


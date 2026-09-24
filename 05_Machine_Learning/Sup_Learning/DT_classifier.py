#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Decision Tree Classifier
# Datasets:
#   - Titanic: ../Datasets/Titanic-Dataset.csv
#   - Breast Cancer Wisconsin: built into sklearn, no download needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer


# In[3]:


# ============================================================
# PART 1 - Titanic (Decision Tree doesn't need scaling, unlike KNN/Logistic Reg)
# ============================================================

df = pd.read_csv("../Datasets/Titanic-Dataset.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
            "Embarked_Q", "Embarked_S"]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 1. Fit a Decision Tree ----
# no scaling needed here - trees split on raw feature values, not distances
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

# ---- 2. Predict and evaluate ----
y_pred = tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---- 3. Visualize the tree ----
# limited to max_depth=4 above specifically so this plot stays readable
plt.figure(figsize=(18, 8))
plot_tree(tree, feature_names=features, class_names=["Died", "Survived"],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree - Titanic Survival")
plt.show()

# ---- 4. Feature importance ----
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": tree.feature_importances_
}).sort_values("Importance", ascending=False)

print(importance_df)

sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance - Decision Tree")
plt.show()
# ---- 5. Check if max_depth matters (overfitting check) ----
depths = range(1, 15)
train_acc = []
test_acc = []

for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, t.predict(X_train)))
    test_acc.append(accuracy_score(y_test, t.predict(X_test)))

plt.plot(depths, train_acc, marker="o", label="Train Accuracy")
plt.plot(depths, test_acc, marker="o", label="Test Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Check - Train vs Test Accuracy")
plt.legend()
plt.show()

# train accuracy keeps climbing with depth, test accuracy plateaus/drops -
# classic sign of overfitting once the tree gets too deep




# In[4]:


# ============================================================
# PART 2 - Breast Cancer Wisconsin (binary classification, sklearn built-in)
# ============================================================

cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df["target"] = cancer.target   # 0 = malignant, 1 = benign

print(cancer_df.shape)
print(cancer_df["target"].value_counts())

Xc = cancer_df.drop(columns=["target"])
yc = cancer_df["target"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42, stratify=yc
)

# ---- 6. Fit Decision Tree ----
tree_cancer = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_cancer.fit(Xc_train, yc_train)

# ---- 7. Predict and evaluate ----
yc_pred = tree_cancer.predict(Xc_test)
print("Accuracy:", accuracy_score(yc_test, yc_pred))
print("\nConfusion Matrix:\n", confusion_matrix(yc_test, yc_pred))
print("\nClassification Report:\n", classification_report(yc_test, yc_pred))

# ---- 8. Top 10 most important features ----
importance_cancer = pd.DataFrame({
    "Feature": cancer.feature_names,
    "Importance": tree_cancer.feature_importances_
}).sort_values("Importance", ascending=False).head(10)

print(importance_cancer)

sns.barplot(x="Importance", y="Feature", data=importance_cancer)
plt.title("Top 10 Feature Importances - Breast Cancer Wisconsin")
plt.show()

# "worst concave points" / "mean concave points" type features are usually
# the strongest predictors in this dataset


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Association Rules - Market Basket Analysis
# Dataset: Groceries dataset - https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset
# (simpler alternative to Instacart, which is much larger and needs more setup)
#
# Needs: pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("../Datasets/Groceries_dataset.csv")


# In[2]:


# ---- 1. First look ----
print(df.shape)
print(df.head())

# columns are usually: Member_number, Date, itemDescription
# each row = one item bought by one member on one date

print(df.isnull().sum())

# ---- 2. Group items into transactions ----
# a "transaction" here = everything one member bought on one date
df["Transaction"] = df["Member_number"].astype(str) + "_" + df["Date"].astype(str)

transactions = df.groupby("Transaction")["itemDescription"].apply(list)
print(transactions.head())
print("Number of transactions:", len(transactions))


# In[3]:


# ---- 3. Convert to one-hot encoded format ----
# apriori needs a table where each column = an item, each row = a transaction,
# with True/False showing if that item was in that transaction
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket_df = pd.DataFrame(te_array, columns=te.columns_)

print(basket_df.shape)
print(basket_df.head())

# ---- 4. Find frequent itemsets ----
# min_support = how often an itemset needs to appear to be considered "frequent"
# start low since grocery data is usually sparse (many unique items, few repeats)
frequent_itemsets = apriori(basket_df, min_support=0.001, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

print(frequent_itemsets.head(15))


# In[4]:


# ---- 5. Generate association rules ----
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values("lift", ascending=False)

print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(15))

# lift > 1 means the items are bought together MORE than random chance would suggest
# confidence = "if they bought A, how often did they also buy B"

# ---- 6. Look at a specific example ----
# find rules where "whole milk" is the antecedent (since it's usually the most common item)
milk_rules = rules[rules["antecedents"].apply(lambda x: "whole milk" in x)]
print(milk_rules[["antecedents", "consequents", "confidence", "lift"]].head(10))


# In[5]:


# ---- 7. Save rules ----
rules.to_csv("association_rules_output.csv", index=False)
print("Saved -> association_rules_output.csv")

# Note: Instacart Market Basket Analysis follows the exact same apriori/association_rules
# approach - the only real difference is it's a much bigger dataset (millions of orders
# across multiple files that need merging first), so it's worth trying this Groceries
# version first to get the concepts down before scaling up to Instacart.


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Netflix Titles EDA - basic exploration
# Dataset: https://www.kaggle.com/datasets/shivamb/netflix-shows


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\syin\Downloads\Netflix_movies_N_tvshows\netflix_titles.csv")


# In[3]:


# ---- 1. First look at the data ----
print(df.shape)
print(df.head())
print(df.info())


# In[4]:


# ---- 2. Missing values ----
print(df.isnull().sum())

# director, cast, country have a lot of missing values
# rating and date_added have a few


# In[5]:


# ---- 3. Movies vs TV Shows ----
print(df["type"].value_counts())

sns.countplot(x="type", data=df)
plt.title("Movies vs TV Shows")
plt.show()

# way more movies than tv shows on netflix


# In[6]:


# ---- 4. Convert date_added to datetime ----
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["year_added"] = df["date_added"].dt.year


# In[7]:


# ---- 5. Titles added per year ----
print(df["year_added"].value_counts().sort_index())

df["year_added"].value_counts().sort_index().plot(kind="line", marker="o")
plt.title("Titles Added per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

# huge jump around 2018-2020, netflix was adding content fast


# In[8]:


# ---- 6. Top countries ----
country_counts = df["country"].dropna().str.split(", ").explode().value_counts()
print(country_counts.head(10))

country_counts.head(10).plot(kind="barh")
plt.title("Top 10 Countries by Number of Titles")
plt.gca().invert_yaxis()
plt.show()
# US and India dominate



# In[9]:


# ---- 7. Ratings ----
print(df["rating"].value_counts())

sns.countplot(y="rating", data=df, order=df["rating"].value_counts().index)
plt.title("Content Rating Distribution")
plt.show()


# In[10]:


# ---- 8. Top genres ----
genre_counts = df["listed_in"].dropna().str.split(", ").explode().value_counts()
print(genre_counts.head(10))

genre_counts.head(10).plot(kind="barh")
plt.title("Top 10 Genres")
plt.gca().invert_yaxis()
plt.show()


# In[11]:


# ---- 9. Movie duration ----
movies = df[df["type"] == "Movie"].copy()
movies["duration_min"] = movies["duration"].str.replace(" min", "", regex=False)
movies["duration_min"] = pd.to_numeric(movies["duration_min"], errors="coerce")

plt.hist(movies["duration_min"].dropna(), bins=30)
plt.title("Movie Duration Distribution")
plt.xlabel("Minutes")
plt.show()

# most movies are around 90-100 minutes



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





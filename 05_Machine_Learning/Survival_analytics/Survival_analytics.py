#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Survival Analytics - Kaplan-Meier and Cox Proportional Hazards
# Datasets:
#   - Telco Customer Churn (Kaggle): ../Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv
#     https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#   - Rossi recidivism + Lung cancer: built into the lifelines package
#
# Needs: pip install lifelines
#
# Survival analysis vocabulary (worth memorising):
#   duration = how long until the event happened OR until we stopped watching
#   event    = 1 if the event actually happened, 0 if the person was "censored"
#              (still around when the data ended, so we don't know their real end)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.datasets import load_rossi, load_lung


# In[4]:


# ============================================================
# PART 1 - Telco Customer Churn (time-to-churn)
# ============================================================

#df = pd.read_csv("../Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(r"C:\Users\syin\Downloads\archive (42)\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.shape)
print(df.head())

# ---- 1. Clean the data ----
# TotalCharges is stored as text because a few rows have blank spaces
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df["TotalCharges"].isnull().sum(), "blank TotalCharges rows")

# those blank rows are brand-new customers with tenure = 0 - no observation time yet,
# so they can't contribute anything to a survival analysis. Drop them.
df = df[df["tenure"] > 0].copy()

# ---- 2. Define duration and event ----
# duration = tenure (months the customer stayed)
# event    = 1 if they churned, 0 if still a customer (censored)
df["event"] = (df["Churn"] == "Yes").astype(int)

print(df["event"].value_counts())
print("Churn rate:", df["event"].mean().round(3))

# ---- 3. Kaplan-Meier curve for ALL customers ----
kmf = KaplanMeierFitter()
kmf.fit(durations=df["tenure"], event_observed=df["event"], label="All customers")
kmf.plot_survival_function()
plt.title("Kaplan-Meier: Probability a Customer Is Still Subscribed")
plt.xlabel("Tenure (months)")
plt.ylabel("Survival probability")
plt.show()

print("Median survival time (months):", kmf.median_survival_time_)
# if this prints inf, more than half the customers never churned during the data window

# ---- 4. Compare groups - month-to-month vs long contracts ----
plt.figure(figsize=(8, 5))
ax = plt.subplot(111)

for contract in df["Contract"].unique():
    mask = df["Contract"] == contract
    kmf_c = KaplanMeierFitter()
    kmf_c.fit(df.loc[mask, "tenure"], df.loc[mask, "event"], label=contract)
    kmf_c.plot_survival_function(ax=ax)

plt.title("Survival Curves by Contract Type")
plt.xlabel("Tenure (months)")
plt.ylabel("Survival probability")
plt.show()

# month-to-month customers drop off much faster than one/two year contracts

# ---- 5. Log-rank test - is the difference statistically real? ----
m2m = df[df["Contract"] == "Month-to-month"]
two_year = df[df["Contract"] == "Two year"]

result = logrank_test(
    m2m["tenure"], two_year["tenure"],
    event_observed_A=m2m["event"], event_observed_B=two_year["event"]
)
print("Log-rank p-value:", result.p_value)
# p < 0.05 -> the two survival curves are significantly different

# ---- 6. Cox Proportional Hazards - which factors affect churn risk? ----
# Cox needs numeric columns only, so pick a few and encode them
cox_df = df[["tenure", "event", "MonthlyCharges", "SeniorCitizen"]].copy()
cox_df["Contract_OneYear"] = (df["Contract"] == "One year").astype(int)
cox_df["Contract_TwoYear"] = (df["Contract"] == "Two year").astype(int)
cox_df["PaperlessBilling"] = (df["PaperlessBilling"] == "Yes").astype(int)
cox_df["Partner"] = (df["Partner"] == "Yes").astype(int)

cph = CoxPHFitter()
cph.fit(cox_df, duration_col="tenure", event_col="event")

cph.print_summary()

cph.plot()
plt.title("Cox Model - Effect of Each Factor on Churn Risk")
plt.show()

# reading the results:
#   exp(coef) > 1 -> factor INCREASES churn risk (hazard)
#   exp(coef) < 1 -> factor DECREASES churn risk
# e.g. Contract_TwoYear should have exp(coef) far below 1 (much lower churn hazard)

print(cph.summary[["exp(coef)", "p"]].round(3))


# In[5]:


# ============================================================
# PART 2 - Rossi recidivism dataset (lifelines built-in)
# ============================================================
# NOTE: Rossi is NOT a cancer dataset - it tracks 432 men released from prison and
# how many weeks until they were re-arrested. It's the standard teaching
# dataset for Cox regression.

rossi = load_rossi()
print(rossi.head())
# week   = weeks until re-arrest (or until follow-up ended)
# arrest = 1 if re-arrested, 0 if censored (not re-arrested by end of study)
# fin    = 1 if received financial aid after release
# age, race, wexp (work experience), mar (married), paro (paroled), prio (prior convictions)

kmf_r = KaplanMeierFitter()
kmf_r.fit(rossi["week"], rossi["arrest"], label="All released prisoners")
kmf_r.plot_survival_function()
plt.title("Rossi - Time Until Re-arrest")
plt.xlabel("Weeks")
plt.show()

# ---- Cox regression on Rossi ----
cph_r = CoxPHFitter()
cph_r.fit(rossi, duration_col="week", event_col="arrest")
cph_r.print_summary()

cph_r.plot()
plt.title("Rossi Cox Model - Effect on Re-arrest Risk")
plt.show()

# ---- Check the proportional hazards assumption ----
# Cox regression assumes each factor's effect stays constant over time
cph_r.check_assumptions(rossi, p_value_threshold=0.05, show_plots=False)


# In[6]:


# ============================================================
# PART 3 - Lung cancer dataset (lifelines built-in, load_lung)
# ============================================================
# This is the real cancer survival dataset: patients with advanced lung cancer.
# time = survival time in days, status = 1 if died, 0 if censored

lung = load_lung()
print(lung.head())
print(lung.isnull().sum())

# keep a few columns and drop rows with missing values (small amount here)
lung_df = lung[["time", "status", "age", "sex", "ph.ecog", "ph.karno"]].dropna()

kmf_l = KaplanMeierFitter()

plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
# sex is coded 1 = male, 2 = female in this dataset
for sex_code, label in [(1, "Male"), (2, "Female")]:
    mask = lung_df["sex"] == sex_code
    kmf_l.fit(lung_df.loc[mask, "time"], lung_df.loc[mask, "status"], label=label)
    kmf_l.plot_survival_function(ax=ax)

plt.title("Lung Cancer Survival by Sex")
plt.xlabel("Days")
plt.ylabel("Survival probability")
plt.show()

cph_l = CoxPHFitter()
cph_l.fit(lung_df, duration_col="time", event_col="status")
cph_l.print_summary()


# In[ ]:







import pandas as pd
import numpy as np
import seaborn as sns

wcat = pd.read_csv("C:/360DG/Datasets/wc_at.csv")

# EDA
wcat.info()
wcat.describe()

# Average waist is 91.90 and min is 63.50 and max is 121
# Average AT is 101.89 and min is 11.44 and max is 253

import matplotlib.pyplot as plt

sns.distplot(wcat.AT)
# Data is normal but right skewed
plt.boxplot(wcat.AT)
# No outliers but right skewed

sns.distplot(wcat.Waist)
# Data is normal bimodal
plt.boxplot(wcat.Waist)
# No outliers but right skewed

####################
# Bivariant analysis
plt.scatter(x=wcat.Waist, y=wcat.AT)
# Data is linearly scattered, direction positive, strength: poor

# Now let us check the correlation coefficient
np.corrcoef(wcat.Waist, wcat.AT)


# F-statistic: This value measures how well your overall regression model fits the data.
# A higher F-value means your model explains much more variance.

# F-statistic = 217.3 and Prob(F) = 1.62e-27 (= 0)
# Model is statistically significant (better than a model with no predictors)

# **Intercept**   -215.98    | When Waist = 0, AT is -215.98
# **Waist**       +3.4589    | For every 1 unit increase in Waist, AT increases by 3.4589

# P-values for both = 0.000  ✓ Both are statistically significant

# 3. Residual Analysis
# The Durbin-Watson statistic tests whether the residuals are autocorrelated.

# DW Value | What it Means
# ≈ 2      | No autocorrelation (Ideal)
# < 1.5    | Positive autocorrelation (bad)


# In One Sentence:
# "Waist has a strong and significant impact on AT, and this model explains about 67% of the variation in AT.
# The regression assumptions are mostly satisfied, so we can trust the results."


pred1 = model.predict(pd.DataFrame(wcat.Waist))
pred1

###############
# Regression line
plt.scatter(wcat.Waist, wcat.AT)
plt.plot(wcat.Waist, pred1, 'r')
plt.legend(['Actual data', 'Predicted data'])
plt.show()

###############
# error calculations
res1 = wcat.AT - pred1
np.mean(res1)

res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


# Let us try another model
# x = log(Waist)
plt.scatter(x=np.log(wcat.Waist), y=wcat.AT)
# Data is linearly scattered, direction positive, strength: poor

# Now let us check the correlation coefficient
np.corrcoef(np.log(wcat.Waist), wcat.AT)

# The correlation coefficient is 0.8217 < 0.85 hence the correlation is good
r = 0.8217

import statsmodels.formula.api as smf
model2 = smf.ols('AT ~ np.log(Waist)', data=wcat).fit()
model2.summary()

# R-squared = 0.675 < 0.85, there is scope of improvement
# p = 0.00 < 0.05 hence acceptable
# beta-0 = -1328.3420
# beta-1 = 317.1356


pred2 = model2.predict(pd.DataFrame(wcat.Waist))
pred2

###############
# Regression line
plt.scatter(np.log(wcat.Waist), wcat.AT)
plt.plot(np.log(wcat.Waist), pred2, 'r')
plt.legend(['Actual data', 'predicted data_model2'])
plt.show()

###############
# error calculations
res2 = wcat.AT - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

# There are no significant change as r = 0.821, RSquare = 0.675
# Hence let us try another model


plt.scatter(x=wcat.Waist, y=np.log(wcat.AT))

# Now let us check the correlation coefficient
np.corrcoef(wcat.Waist, np.log(wcat.AT))

# The correlation coefficient is 0.8409 < 0.85 hence the correlation is good
r = 0.8409

model3 = smf.ols('np.log(AT) ~ Waist', data=wcat).fit()
model3.summary()

# R-squared = 0.707 < 0.85, there is scope of improvement
# p = 0.002 < 0.05 hence acceptable
# beta-0 = 0.7410
# beta-1 = 0.0403


pred3 = model3.predict(pd.DataFrame(wcat.Waist))
pred3_at = np.exp(pred3)
pred3_at

###############
# Regression line
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist, pred3, 'r')
plt.legend(['Predicted line', 'Observed data_model3'])
plt.show()

###############
# error calculations
res3 = wcat.AT - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


# We have to generalize the best model
from sklearn.model_selection import train_test_split

train, test = train_test_split(wcat, test_size=0.2)

plt.scatter(train.Waist, np.log(train.AT))
plt.scatter(test.Waist, np.log(test.AT))

final_model = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data=wcat).fit()
final_model.summary()

# R-squared = 0.779 < 0.85, there is scope of improvement
# p = 0.000 < 0.05 hence acceptable
# beta-0 = -7.8241
# beta-1 = 0.2289


test_pred = final_model.predict(pd.DataFrame(test))
test_pred_at = np.exp(test_pred)
test_pred_at

################
train_pred = final_model.predict(pd.DataFrame(train))
train_pred_at = np.exp(train_pred)
train_pred_at

################
# Evaluation on test data
test_err = test.AT - test_pred_at
test_sqr = test_err * test_err
test_mse = np.mean(test_sqr)
test_rmse = np.sqrt(test_mse)
test_rmse

################
# Evaluation on train data
train_res = train.AT - train_pred_at
train_sqr = train_res * train_res
train_mse = np.mean(train_sqr)
train_rmse = np.sqrt(train_mse)
train_rmse

# test_rmse > train_rmse
# The model is overfit

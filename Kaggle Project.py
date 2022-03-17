import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

bike = pd.read_csv("/Users/emmanuelacquah/Desktop/Udemy/Udemy DS/Kaggle Project/hour.csv")

# 1- Preliminary Analysis
bike_prep = bike.copy()
bike_prep = bike_prep.drop(['index', 'date', 'casual', 'registered'], axis=1)
# _______________________________________
# 2- Basic Analysis
# ______________________________________
bike_prep.isnull().sum()
bike_prep.hist(rwidth=0.9)
plt.tight_layout()

# _______________________
# 3 - Data Visualization
# _______________________
# Analysing relationship between the Demand vs Other features
z = 0.5
plt.subplot(2, 2, 1)
plt.title('Temperature vs Demand')
plt.scatter(bike_prep['temp'], bike_prep['demand'], s=z, c='b')

plt.subplot(2, 2, 2)
plt.title('aTemperature vs Demand')
plt.scatter(bike_prep['atemp'], bike_prep['demand'], s=z, c='y')

plt.subplot(2, 2, 3)
plt.title('Humidity vs Demand')
plt.scatter(bike_prep['humidity'], bike_prep['demand'], s=z, c='m')

plt.subplot(2, 2, 4)
plt.title('Windspeed vs Demand')
plt.scatter(bike_prep['windspeed'], bike_prep['demand'], s=z, c='k')
plt.tight_layout()
plt.show()

# Plot categorical features vs Demand
colors = ['r', 'b', 'm', 'y']
plt.subplot(3, 3, 1)
# Drop all duplicates to get unique features
cat_sea = bike_prep['season'].unique()

# Find the average demand per each unique feature in season (AverageIF/SumIF)
cat_seaAv = bike_prep.groupby('season').mean()['demand']
plt.title('Demand per season')
plt.bar(cat_sea, cat_seaAv, color=colors)

plt.subplot(3, 3, 2)
cat_yr = bike_prep['year'].unique()
cat_yrAv = bike_prep.groupby('year')['demand'].mean()
plt.title('Demand per year')
plt.bar(cat_yr, cat_yrAv, color=colors)

plt.subplot(3, 3, 3)
cat_mon = bike_prep['month'].unique()
cat_monAv = bike_prep.groupby('month')['demand'].mean()
plt.title('Demand per month')
plt.bar(cat_mon, cat_monAv, color=colors)

plt.subplot(3, 3, 4)
cat_hr = bike_prep['hour'].unique()
cat_hrAv = bike_prep.groupby('hour')['demand'].mean()
plt.title('Demand per Hour of day')
plt.bar(cat_hr, cat_hrAv, color=colors)

plt.subplot(3, 3, 5)
cat_wkday = bike_prep['weekday'].unique()
cat_wkdAv = bike_prep.groupby('weekday')['demand'].mean()
plt.title('Demand per Weekday')
plt.bar(cat_wkday, cat_wkdAv, color=colors)

plt.subplot(3, 3, 6)
cat_hday = bike_prep['holiday'].unique()
cat_hdayAv = bike_prep.groupby('holiday')['demand'].mean()
plt.title('Demand per Holiday')
plt.bar(cat_hday, cat_hdayAv, color=colors)

plt.subplot(3, 3, 7)
cat_wrkday = bike_prep['workingday'].unique()
cat_wrkdayAv = bike_prep.groupby('workingday')['demand'].mean()
plt.title('Demand per workingday')
plt.bar(cat_wrkday, cat_wrkdayAv, color=colors)

plt.subplot(3, 3, 8)
cat_weather = bike_prep['weather'].unique()
cat_weatherAv = bike_prep.groupby('weather')['demand'].mean()
plt.title('Demand per weather')
plt.bar(cat_weather, cat_weatherAv, color=colors)
plt.tight_layout()
plt.show()

# Check for Outliers
bike_prep['demand'].describe()
bike_prep['demand'].quantile(q=[0.05, 0.1, 0.15, 0.9, 0.95, 0.99])

# ________________________________________________
# 4. Check Multiple Linear Regression Assumptions
# _________________________________________________

# 1. Check the Linearity of variables using the correlation coefficient matrix using the corr function

correlation = bike_prep[['temp', 'atemp', 'humidity', 'windspeed', 'demand']].corr()

# We drop the columns that are not needed

bike_prep = bike_prep.drop(['atemp', 'windspeed', 'year', 'weekday', 'workingday'], axis=1)

# Check  for Autocorrelation in the demand feature

df1 = pd.to_numeric(bike_prep['demand'], downcast='float')
plt.acorr(df1, maxlags=12)

# Solving the Normality problem in Demand feature

df1 = bike_prep['demand']
df2 = np.log(df1)

plt.figure()
plt.hist(df1, bins=20, rwidth=0.9)

plt.figure()
plt.hist(df2, bins=20, rwidth=0.9)

bike_prep['demand'] = df2

# Solving the problem of autocorrelation in demand
t_1 = bike_prep['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = bike_prep['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = bike_prep['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']

bike_prep_lag = pd.concat([bike_prep, t_1, t_2, t_3], axis=1)
bike_prep_lag = bike_prep_lag.dropna()

bike_prep_lag.isnull().sum(axis=0)

# ____________________________________________________________
# Create Dummy variables and drop the first to prevent Dummy variable trap.
# We need to change the data type of the categorical data to categories first
# ______________________________________________________________

bike_prep_lag['season'] = bike_prep_lag['season'].astype('category')
bike_prep_lag['holiday'] = bike_prep_lag['holiday'].astype('category')
bike_prep_lag['weather'] = bike_prep_lag['weather'].astype('category')
bike_prep_lag['month'] = bike_prep_lag['month'].astype('category')
bike_prep_lag['hour'] = bike_prep_lag['hour'].astype('category')

bike_prep_lag = pd.get_dummies(bike_prep_lag, drop_first=True)
bike_prep_lag.isnull().sum()
# _________________________________________________________________
# Train and Test Data
# __________________________________________________________________
# We are not using the train_test_split because the "demand" data is autocorrelated and so we will need to split the
# data manually

Y = bike_prep_lag[['demand']]
X = bike_prep_lag.drop(['demand'], axis=1)

tr_size = 0.8 * len(X)
tr_size = int(tr_size)

x_train = X.values[0: tr_size]
x_test = X.values[tr_size: len(X)]

y_train = Y.values[0: tr_size]
y_test = Y.values[tr_size: len(Y)]

pd.isnull(x_train).sum(axis=1)
# __________________________________________________
# Building the ML model
# __________________________________________________
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
import math

mlr_reg = LinearRegression()
mlr_reg.fit(x_train, y_train)

R_score_tst = mlr_reg.score(x_test, y_test)
R_score_train = mlr_reg.score(x_train, y_train)
print(R_score_tst, R_score_train)

mlr_coefficient = mlr_reg.coef_
print(mlr_coefficient)

mlr_intercept = mlr_reg.intercept_

Y_pred = mlr_reg.predict(x_test)
rmse = math.sqrt(mean_squared_error(y_test, Y_pred))

# _________________________________________
# Final - Calculate RMSLE
# ____________________________________________

y_test_e = []
Y_pred_e = []

for i in range(0, len(Y_pred)):
    pv = math.exp(y_test[i])
    y_test_e.append(pv)
    Y_pred_e.append(math.exp(Y_pred[i]))

# Do the sum of the logs and squares
log_sum = 0
for i in range(0, len(Y_pred)):
    log_a = math.log(y_test_e[i] + 1)
    log_p = math.log(Y_pred_e[i] + 1)
    log_diff = (log_a - log_p) ** 2
    log_sum = log_sum + log_diff
rmsle = math.sqrt(log_sum / len(Y_pred))
rmsle2 = math.sqrt(mean_squared_log_error(y_test, Y_pred))
print(rmsle)

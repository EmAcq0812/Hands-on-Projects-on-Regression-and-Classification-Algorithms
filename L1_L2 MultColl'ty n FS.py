import pandas as pd

data = pd.read_csv("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Complete File"
                   "/L1 L2 - Multicollinearity n Feat. Selt/mcl.csv")

Data_S = data.copy()
X = Data_S.iloc[:, :-1]
Y = Data_S.iloc[:, -1]

# correlation matrix of the independent features

Correlation = X.corr()
import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(Correlation, annot=False)
plt.show()

# Perform 3 regressions on the Data

from sklearn.linear_model import Ridge, Lasso, LinearRegression

# Linear Regression
lr = LinearRegression()
lr.fit(X, Y)
lr_int  = lr.intercept_
lr_coef = lr.coef_

# Lasso Regression

lasso = Lasso(alpha=10)
lasso.fit(X, Y)
lasso_coef = lasso.coef_
lasso_int  = lasso.intercept_

# Ridge Regression

ridge = Ridge(alpha=100)
ridge.fit(X, Y)
ridge_coef = ridge.coef_
ridge_int  = ridge.intercept_


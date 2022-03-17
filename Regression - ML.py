import pandas as pd

Stu2_data = pd.read_csv("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Section 3/Multiple Linear/02Students.csv")
df = Stu2_data.copy()

# Split data
X = df.iloc[:, :-1]
#X_1 = X.iloc[:,:-1]
#X_2 = df.iloc[:, 1]
Y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

from sklearn.linear_model import LinearRegression

mlr_reg = LinearRegression()
mlr_reg.fit(x_train, y_train)

mlr_reg.score(x_test, y_test)

mlr_coefficient = mlr_reg.coef_

mlr_intercept = mlr_reg.intercept_

mlr_pred = mlr_reg.predict(x_test)

# Calculating the RMSE

from sklearn.metrics import mean_squared_error
import math

mlr_RMSE = math.sqrt(mean_squared_error(y_test, mlr_pred))

# Equation of the line: y = 1.31 + 4.66x_1 + 5.07x_2


# Plotting
import matplotlib.pyplot as plt
plt.scatter(x_test, y_test)

#trendline
plt.plot(x_test, mlr_pred)
plt.ylim(ymin=0)
plt.show()
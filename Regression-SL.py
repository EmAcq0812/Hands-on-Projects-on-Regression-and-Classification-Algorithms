import pandas as pd

Stu_data = pd.read_csv("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Section 3/Simple Linear/01Students.csv")
df = Stu_data.copy()

# Split data

X = df.iloc[:,:-1]
Y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

from sklearn.linear_model import LinearRegression

Stu_reg = LinearRegression()
Stu_reg.fit(x_train, y_train)

Stu_reg.score(x_test,y_test)

Stu_coefficient = Stu_reg.coef_

Stu_intercept = Stu_reg.intercept_

Stu_pred = Stu_reg.predict(x_test)

# Calculating the RMSE

from sklearn.metrics import mean_squared_error
import math

Stu_RMSE = math.sqrt(mean_squared_error(y_test, Stu_pred))

# Plotting
import matplotlib.pyplot as plt
plt.scatter(x_test, y_test)

#trendline
plt.plot(x_test, Stu_pred)
plt.ylim(ymin=0)
plt.show()
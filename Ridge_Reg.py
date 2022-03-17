import pandas as pd

dataset = pd.read_csv('/Users/emmanuelacquah/Desktop/Udemy/Udemy DS/Complete File/Ridge-Lasso Reg/ridge.csv')
data = dataset.copy()

# Split

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

from sklearn.linear_model import Ridge
alphas = [0, 2, 5, 10, 100]
for l in alphas:
    ridge = Ridge(alpha=l)
    ridge.fit(X, Y)
    coefficient = ridge.coef_
    intercept = ridge.intercept_

    # Ridge: y = 1.667x + 0.833

    from sklearn.linear_model import LinearRegression

    linReg = LinearRegression()
    linReg.fit(X,Y)
    coefficient2 = linReg.coef_
    intercept2 = linReg.intercept_

    # Linear: y = 2x

    # plot the ridge
    x_plt = [0, 1, 2, 3, 4]
    y_plt = ridge.predict(pd.DataFrame(x_plt))

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(x_plt, y_plt)
    plt.ylim(ymin=0, ymax=9)
    plt.xlim(xmin=0, xmax=6)

    # add text to the plot
    plt.text(x_plt[-1], y_plt[-1], ' y = ' + str('%.2f' %coefficient) + ' * x + '
         + str('%.2f' %intercept) + ' for \u03BB or \u03B1 =' + str(l), fontsize=12)

for i, l in enumerate(alphas):
    print(l)

import matplotlib.pyplot as plt

x_days = [1, 2, 3, 4, 5]
y_price1 = [9, 9.5, 10, 10.4, 11]
y_price2 = [8, 9, 8.3, 7, 7.5]

# chart elements
plt.title('Stock Market Movement')
plt.xlabel('Week days')
plt.ylabel('Prices')

plt.plot(x_days, y_price1, label = 'stock1', color= 'green', linewidth= 2, marker= 'o', markersize = 5, linestyle= '--')
plt.plot(x_days, y_price2, label = 'stock2', color= 'b', linewidth= 2, marker= 'o', markersize = 5, linestyle= '--')

plt.legend(loc=2, fontsize = 12)

plt.show()





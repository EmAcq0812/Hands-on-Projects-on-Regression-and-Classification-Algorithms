import matplotlib.pyplot as plt
f = open("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Section 2/salesdata.csv", 'r')
sales_data = f.readlines()
print(sales_data)

sales_list = []

for dt in sales_data:
    sales_list.append(int(dt))

plt.title('Box plot of sales data')
plt.boxplot(sales_list)
plt.show()
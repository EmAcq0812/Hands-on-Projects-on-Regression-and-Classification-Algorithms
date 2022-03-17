import matplotlib.pyplot as plt

sales_data = open("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Section 2/salesdata2.csv", 'r')

sales_data = sales_data.readlines()

s_list = []
c_list = []

for lst in sales_data:
    sales, cost = lst.split(',')
    s_list.append(int(sales))
    c_list.append(int(cost))

plt.title('Cost vs Sales')
plt.xlabel('sales')
plt.ylabel('cost')

plt.scatter(s_list, c_list, linewidths= 1)
plt.show()

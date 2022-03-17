import matplotlib.pyplot as plt

sales_data = open("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Section 2/salesdata2.csv", 'r')

sales_data = sales_data.readlines()

sales_list = []
s_list = []
c_list = []

for lst in sales_data:
    sales, cost = lst.split(',')
    s_list.append(int(sales))
    c_list.append(int(cost))
    sales_list.append(int(sales))

plt.figure('My Scatter plot')
plt.title('Cost vs Sales')
plt.xlabel('sales')
plt.ylabel('cost')
plt.scatter(s_list, c_list, linewidths= 1)
plt.show()

plt.figure('My Boxplot')
plt.title('Box plot of sales data')
plt.ylabel('USD')
plt.boxplot(sales_list)
plt.show()

# Multiplots in one figure
plt.subplot(2, 1, 1)
plt.title('Cost vs Sales')
plt.xlabel('sales')
plt.ylabel('cost')
plt.scatter(s_list, c_list, linewidths=0.02, marker='*', s=100, c='#FF5733')

# Markers:

plt.subplot(2, 1, 2)
plt.title('Box plot of sales data')
plt.ylabel('USD')
plt.boxplot(sales_list, patch_artist=True, boxprops=dict(facecolor='g', color = 'r', linewidth= 2),
            whiskerprops= dict(color = 'r', linewidth= 2),
            medianprops= dict(color = 'w', linewidth= 2),
            capprops= dict (color = 'k', linewidth = 2),
            flierprops= dict(markerfacecolor='r', marker='o', markersize=5))

plt.tight_layout()  # Solves overlapping problem

plt.show()
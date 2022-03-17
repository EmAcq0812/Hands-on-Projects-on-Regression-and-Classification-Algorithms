import matplotlib.pyplot as plt
from collections import Counter

age_data2 = open("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Section 2/agedata2.csv", 'r')

AgeD = age_data2.readlines()
listcity = []
for agd in AgeD:
    age, city = agd.split(',')
    listcity.append(city)
city_count = Counter(listcity)
print(city_count)

city_names = list(city_count.keys())
city_age = list(city_count.values())

Citlist = [city_names, city_age]
print(Citlist)

print(Citlist[1][0])

plt.pie(city_age, labels= city_names, autopct= '%.2f%%')
plt.show()



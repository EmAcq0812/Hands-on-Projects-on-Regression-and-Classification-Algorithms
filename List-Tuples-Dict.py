import numpy as np
print('Hello')
x = 2
y = 3
print(x*y)

z = "Emmanuel"
b = "Acquah"
print(z)

e = np.concatenate(z[3], x, )
print(e)

print(z[2])


for l in z:
    for i in b:
        print(l + i)
import numpy as np

list1 = ['Kofi', 'Ama', 'Yaw']
list2 = ["James", "Anane"]
list3 = list1.append(list2)
print(list1)
print(list1[3])
list1[0] = list2[0]

del list2[0]

# Multidimensional list

twodlist = [['Kofi', 'Ama', 'Yaw'], [2, 3, 4], [10, 12, 17]]
print(twodlist[1][0])
twodlist[0][0] = 6
print(twodlist)

# Slicing a list

sublist = []

for lst in twodlist:
    sublist.append(list1[0:2])
    print(sublist)

# Tuples: The difference between list and Tuple is that elements in a tuple cannot be changed, updated or deleted.

tup1 = ('Ama', 2, 4)

tup1.index(2)

# Dictionary: Dictionary is a type of list with key-value pair enteries.

dic1 = {'country': 'Ghana', 'Region':'Ashanti Region', 'District': 'Kwabre District', 'Town': 'Heman'}
print(dic1['District'])

dic1['Zip'] = 75428
print(dic1)
del dic1['Town']
dic1['country'] = 'Togo'
print(dic1)
# Convert dictionary to string

dic_str = str(dic1)

print(dic_str)
print(dic1)

#Access all keys and values

print(dic1.keys())
print(dic1.values())


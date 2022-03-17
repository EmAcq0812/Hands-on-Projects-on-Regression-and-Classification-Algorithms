from sklearn import datasets
iris = datasets.load_iris()

x_i = iris.data
y_i = iris.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_i,y_i,test_size=0.3, random_state=1234,stratify=y_i)
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# train the SVC
# 1st training
svc = SVC(kernel='rbf', gamma=1.0)
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)
cm_rbf01 = confusion_matrix(y_test,y_predict)

# 2nd training
svc = SVC(kernel='rbf', gamma=10)
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)
cm_rbf10 = confusion_matrix(y_test,y_predict)

# 3rd training - Linear Kernel
svc = SVC(kernel='linear')
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)
cm_rbf_lin = confusion_matrix(y_test,y_predict)

# 4th training - Polynomial
svc = SVC(kernel='poly')
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)
cm_rbf_poly = confusion_matrix(y_test,y_predict)


# 5th training - Sigmoid kernel
svc = SVC(kernel='sigmoid')
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)
cm_rbf_sig = confusion_matrix(y_test,y_predict)
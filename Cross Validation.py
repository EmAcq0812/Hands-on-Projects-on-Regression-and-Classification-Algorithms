# COmpare different Classifeirs for different train and test values

import pandas as pd

data = pd.read_csv("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Complete File/GridSearch/hpt_small.csv")
data_r = data.copy()

data_r.dtypes

data_prep = pd.get_dummies(data_r, drop_first=True)

X = data_prep.iloc[:,:-1]
Y = data_prep.iloc[:,-1]

# import classifiers

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1234)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)


from sklearn.svm import SVC
svc = SVC(kernel='rbf', gamma=0.5)

from sklearn.linear_model import LogisticRegression
logist = LogisticRegression(random_state=1234)

# cross validation

from sklearn.model_selection import cross_validate
cv_result_dtc = cross_validate(dtc, X, Y, cv=10, return_train_score=True, )
cv_result_rfc = cross_validate(rfc, X, Y, cv=10, return_train_score=True)
cv_result_svc = cross_validate(svc, X, Y, cv=10, return_train_score=True)
cv_result_log = cross_validate(logist, X, Y, cv=10, return_train_score=True)


# Get average of train and test score
import numpy as np
cv_test_dtc = np.average(cv_result_dtc['test_score'])
cv_test_rfc = np.average(cv_result_rfc['test_score'])
cv_test_svc = np.average(cv_result_svc['test_score'])
cv_test_log = np.average(cv_result_log['test_score'])


cv_train_dtc = np.average(cv_result_dtc['train_score'])
cv_train_rfc = np.average(cv_result_rfc['train_score'])
cv_train_svc = np.average(cv_result_svc['train_score'])
cv_train_log = np.average(cv_result_log['train_score'])


print()
print()

print('       ', 'Decision Tree ', '  Random Forest  ', '   Support Vector  ', '  Logistic Reg ')
print('       ','______________ ', '  _____________  ', '  _______________ ','    ____________ ')
print(' Test :  ', round(cv_test_dtc, 4), '              ',
      round(cv_test_rfc, 4), '                ',
      round(cv_test_svc, 4), '           ',
      round(cv_test_log, 4))
print(' Train:  ', round(cv_train_dtc, 4), '             ',
      round(cv_train_rfc, 4),'               ', round(cv_train_svc, 4),'          ',
      round(cv_train_log, 4))


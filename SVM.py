import pandas as pd

loan_data = pd.read_csv("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Section 4/Logistic/01Exercise1.csv")
loan_prep = loan_data.copy()
loan_prep.isnull().sum(axis=0)

# we simply drop the missing values

loan_prep = loan_prep.dropna()
loan_prep.isnull().sum(axis=0)

# drop irrelevant columns: "gender"

loan_prep = loan_prep.drop(['gender'],axis=1)

# create dummy variable for categorical features
loan_prep.dtypes
loan_prep['married'].astype('category')
loan_prep = pd.get_dummies(loan_prep,drop_first=True)

# Normalize the numeric data

from sklearn.preprocessing import StandardScaler

scalar_ = StandardScaler()
loan_prep['income']  = scalar_.fit_transform(loan_prep[['income']])
loan_prep['loanamt'] = scalar_.fit_transform(loan_prep[['loanamt']])



# Split the data

Y = loan_prep[['status_Y']]
X = loan_prep.drop(['status_Y'], axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

# Build the Support Vector Classifier
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
y_pred = pd.DataFrame(y_pred)

# Check the accuracy of our model Using confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

score = svc.score(x_test, y_test)

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

# Build the Logistic model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred = pd.DataFrame(y_pred)

# Check the accuracy of our model Using confusion matrix

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
score = lr.score(x_test, y_test)

# Precision, Accuracy, Sensitivity, F1score

cr = classification_report(y_test, y_pred)
score2 = accuracy_score(y_test, y_pred)

# Getting the probabilities of predictions

Y_prob = lr.predict_proba(x_test)

# We want to access the positive column (1) of Y_prob

Y_prob_2 = Y_prob[:, 1]

# We are going to make new predictions based on Y_prob2 and an increased threshold
threshold = 0.8
Y_new_pred = []

for i in range(0, len(Y_prob_2)):
    if Y_prob_2[i] > threshold:
        Y_new_pred.append(1)
    else:
        Y_new_pred.append(0)

cr_1 = classification_report(y_test, Y_new_pred)
score3 = accuracy_score(y_test, Y_new_pred)
cm_2 = confusion_matrix(y_test, Y_new_pred)

# Get the AUC and plot the curve

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, threshold = roc_curve(y_test, Y_prob_2)
AUC = roc_auc_score(y_test, Y_prob_2)

import matplotlib.pyplot as plt

plt.plot(fpr,tpr, linewidth=4)
plt.title('ROC curve for loan prediction')
plt.xlabel('False positve')
plt.ylabel('True positve')
plt.grid()

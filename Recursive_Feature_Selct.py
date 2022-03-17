import pandas as pd

data = pd.read_csv("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Complete File/008 - "
                   "Feature Selection and Dimensionality Reduction/bank.csv")
bnk_data = data.copy()

bnk_data.isnull().sum(axis=1)

# drop the duration column
bnk_data = bnk_data.drop('duration', axis=1)

# Split the data
X = bnk_data.iloc[:, :-1]
Y = bnk_data.iloc[:, -1]

# Create dummies

X = pd.get_dummies(X, drop_first=True)
Y = pd.get_dummies(Y, drop_first=True)

# split the data into training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0, stratify=Y)

# Import Random forest Classifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=1234)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

# Score and Evaluate model

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
score = rfc.score(x_test, y_test)

# Apply Recursive feature elimination

from sklearn.feature_selection import RFE

rfc2 = RandomForestClassifier(random_state=1234)
rfe = RFE(rfc2, n_features_to_select=30, step=1)

rfe.fit(X, Y)

x_train_rfe = rfe.transform(x_train)
x_test_rfe  = rfe.transform(x_test)

# fit our new feature to rfc2
rfc2.fit(x_train_rfe, y_train)
y_pred2 = rfc2.predict(x_test_rfe)

# score and evaluation
cm_rfe = confusion_matrix(y_test, y_pred2)
score_rfe = rfc2.score(x_test_rfe, y_test)

# Check which features were actually selected using the RFE

# Get columns
columns = list(X.columns)

# Get the ranking of the features.
# Ranking 1 were selected as features.
ranking = rfe.ranking_

# Get the feature importance ranked
important_features = rfc.feature_importances_

# create dataframe of the columns, ranking and important features

rfe_selected = pd.DataFrame()        # Empty dataframe
rfe_selected = pd.concat([pd.DataFrame(columns), pd.DataFrame(ranking),
                          pd.DataFrame(important_features)], axis=1)
rfe_selected.columns = ['Features', 'Rank', 'Feature Importance']


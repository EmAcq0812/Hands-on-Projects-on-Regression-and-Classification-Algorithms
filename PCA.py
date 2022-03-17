from sklearn.datasets import load_breast_cancer
import pandas as pd

lbc = load_breast_cancer()
X = pd.DataFrame(lbc['data'], columns=lbc['feature_names'])
Y = pd.DataFrame(lbc['target'], columns=['types'])
lbc2 = pd.concat([X, Y], axis=1)


# Perform analysis without PCA
# We us the Random Forest classifier

# Split Data into Train test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3 , random_state=1234, stratify=Y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

rbc = RandomForestClassifier()
rbc.fit(x_train, y_train)
y_predict = rbc.predict(x_test)
score1 = rbc.score(x_test, y_test)
cm1 = confusion_matrix(y_test, y_predict)

# ___________________________________________
# Perform PCA and compare results
# ___________________________________________

# center the data

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
x_scaled = scalar.fit_transform(X)

from sklearn.decomposition import PCA

pca = PCA(n_components=5)
x_pca = pca.fit_transform(x_scaled)

x_train_pca, x_test_pca = train_test_split(x_pca, test_size=0.3 , random_state=1234, stratify=Y)

rbc2 = RandomForestClassifier()
rbc2.fit(x_train_pca, y_train)
y_predict2 = rbc2.predict(x_test_pca)
score2 = rbc2.score(x_test_pca, y_test)
cm2 = confusion_matrix(y_test, y_predict2)





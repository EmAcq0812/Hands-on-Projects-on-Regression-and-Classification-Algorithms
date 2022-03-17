import pandas as pd

std_dat = pd.read_csv("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Complete File/008 - Feature Selection and Dimensionality Reduction/Students2.csv")

# Split the data

X = std_dat.iloc[:, :-1]
Y = std_dat.iloc[:, -1]

# import various select transforms along with the f_regression mode

from sklearn.feature_selection import SelectKBest, SelectPercentile, GenericUnivariateSelect,\
    f_regression

selectork = SelectKBest(score_func=f_regression, k=3)

x_k = selectork.fit_transform(X, Y)

# Get the f_scores and p-values of our features

f_score = selectork.scores_
p_value = selectork.pvalues_


columns = list(X.columns)

print("   ")
print("   ")
print("   ")

print("    Features   ", "   f_score   ", "  P_value  ")
print("_______________   _____________ _______________")

for i in range(0, len(columns)):
    f1 = '%4.2f' % f_score[i]
    p1 = '%2.6f' % p_value[i]
    print(' ', columns[i].ljust(12), f1.rjust(8), "        ", p1.rjust(6))

# To know which features were exactly selected

cols = selectork.get_support(indices=True)
selected_cols = X.columns[cols].tolist()
print(selected_cols)

# Using Select Percentile
selectorP = SelectPercentile(score_func=f_regression(X, Y), percentile=50)
columns2 = list(X.columns)

f_score2 = selectork.scores_
p_value2 = selectork.pvalues_

print("   ")
print("   ")
print("   ")

print("    Features   ", "   f_score2   ", "  P_value2  ")
print("_______________   _____________ _______________")

for i in range(0, len(columns2)):
    f2 = '%4.2f' % f_score2[i]
    p2 = '%2.6f' % p_value2[i]
    print(' ', columns2[i].ljust(12), f2.rjust(8), "        ", p2.rjust(6))


# Implement GenericUnivariateSelector
selectorg1 = GenericUnivariateSelect(score_func=f_regression, mode="k_best", param=3)

x_g1 = selectorg1.fit_transform(X, Y)
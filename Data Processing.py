import pandas as pd

dataset = pd.read_csv("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Complete File/004 - Data Preprocessing/loan_small.csv")

# Accessing inputs of a data frame using iloc

subset = dataset.iloc[0:3,1:3]

# Accessing data using column names

subsetN = dataset[['ApplicantIncome', 'LoanAmount']]
subsetNr = dataset[['ApplicantIncome', 'LoanAmount']][0:3]

# When data is Tab-Separated
datasetT = pd.read_csv("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Complete File/004 - Data Preprocessing/loan_small_tsv.txt",sep='\t')
print(datasetT)

datasetA = pd.read_csv("C:/Users/AcquahEmmanuel/Desktop/Udemy DS/Section 2/salesdata2.csv", header= None)
print(datasetA)

# Functions
# Get quick view of the dataset
dataset.head()

dataset.head(10)

# Get the shape of the dataset
dataset.shape

# Get the column names of the dataset
dataset.columns

# Find out the columns with missing values
dataset.isnull().sum(axis=0)

# Handling missing values
cleandata = dataset.dropna()   # this deletes all rows with one or more missing values

cleandata2 = dataset.dropna(subset=['Loan_Status']) # deletes rows in the specified column with missing values

# Replacing missing categorical data
dt = dataset.copy()
cols = ['Gender', 'Area', 'Loan_Status']
dt[cols] = dt[cols].fillna(dt.mode().iloc[0])  # Replaces the missing categorical data with the mode of data in the respective columns
dt.isnull().sum(axis=0)

# Replacing missing value for numerical data

cols2 = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
dt[cols2] = dt[cols2].fillna(dt.mean())
dt.isnull().sum(axis=0)

# Label Encoding
dt.dtypes

dt[cols] = dt[cols].astype('category')
for columns in cols:
     dt[columns] = dt[columns].cat.codes

# Hot Encoding or dummy variables for categorical data

df2 = dt.drop(['Loan_ID'], axis=1)
df2 = pd.get_dummies(df2)
df2.dtypes
df2[cols] = dt[cols].astype('category')
df2 = pd.get_dummies(df2)
df2.dtypes

# Normalizing numeric data

data_to_scale = df2.iloc[:, 0:3]
data_to_scale = data_to_scale['CoapplicantIncome'].astype('int')

# Import Standard scalar from sklearn
from sklearn.preprocessing import StandardScaler

# Using z-score transformation
scalar_ = StandardScaler()
ss_scalar_ = scalar_.fit_transform(data_to_scale)

# Using MinMax transformation
from sklearn.preprocessing import minmax_scale
mm_scale = minmax_scale(data_to_scale)


# Splitting data into Train and Testing data
df2 = df2.drop(['Area_0','Gender_0', 'Loan_Status_0'], axis=1)

df2.dtypes

# Splitting the data vertically
X = df2.iloc[:, :-1] # fetch all columns except the last column
Y = df2.iloc[:, -1]  # fetch the last columns

# Split data by rows

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

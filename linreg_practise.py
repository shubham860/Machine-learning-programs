import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('housing.csv')
pd.plotting.scatter_matrix(dataset)

plt.scatter(dataset['households'], dataset['total_rooms'])
dataset.isnull().sum()

X = dataset.iloc[:,[0,1,2,3,4,5,6,7,9]].values
y =dataset.iloc[:,8].values

from sklearn.preprocessing import Imputer
imp = Imputer(strategy = 'median')
X[:,[4]]  = imp.fit_transform(X[:,[4]])

from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
X[:,8] = lbl.fit_transform(X[:,8])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[8])
X = ohe.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)

#split dataset into test cases and train cases
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)

lin_reg.score(X_train,y_train)
lin_reg.score(X_test,y_test)
lin_reg.score(X,y)








import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('titanic.csv')

X = dataset.iloc[:,[3,4,5,6,7]].values
y = dataset.iloc[:,1].values


from sklearn.preprocessing import Imputer
sim = Imputer(missing_values='NaN',strategy='mean')
X[:,[2]] = sim.fit_transform(X[:,[2]]) 

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X[:,0] = le.fit_transform(X[:,0])
X[:,1] = le.fit_transform(X[:,1])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0,1])
X = ohe.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc.score(X_train,y_train)
dtc.score(X_test,y_test)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
log_reg.score(X_train,y_train)
log_reg.predict(X[[500]])
log_reg.score(X_test,y_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X_train,y_train)
knn.score(X_test,y_test)

y_pred = dtc.predict(X)

from sklearn.metrics import confusion_matrix
cnn = confusion_matrix(y,y_pred)

from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y,y_pred)
recall_score(y,y_pred)
f1_score(y,y_pred)

from sklearn.tree import export_graphviz
export_graphviz(dtc,out_file='Titanic dataset/titanic.dot')

import graphviz
with open('Titanic dataset/titanic.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)   

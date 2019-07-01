import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('NLP/sal.csv',names = ['age',
                                                  'workclass',
                                                  'fnlwgt',
                                                  'education',
                                                  'education-num',
                                                  'marital-status',
                                                  'occupation',
                                                  'relationship',
                                                  'race',
                                                  'gender',
                                                  'capital-gain',
                                                  'capital-loss',
                                                  'hours-per-week',
                                                  'native-country',
                                                  'salary'],na_values=' ?')

X = dataset.iloc[:,0:14].values
y = dataset.iloc[:,-1].values

dataset.isnull().sum()

temp = pd.DataFrame(X[:,[1, 6, 13]])
temp[0].value_counts().index[0]
temp[1].value_counts().index[0]
temp[2].value_counts().index[0]

temp[0] = temp[0].fillna(' Private')
temp[1] = temp[1].fillna(' Prof-specialty')
temp[2] = temp[2].fillna(' United-States')

temp.isnull().sum()

X[:,[1,6,13]] = temp
del(temp)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#convert all strings columns into float

#Encoding Workclass
X[:,1] = le.fit_transform(X[:,1])
#Encoding Education
X[:,3] = le.fit_transform(X[:,3])
#Encoding marital-status
X[:,5] = le.fit_transform(X[:,5])
#Encoding occupation
X[:,6] = le.fit_transform(X[:,6])
#Encoding relationship
X[:,7] = le.fit_transform(X[:,7])
#Encoding race
X[:,8] = le.fit_transform(X[:,8])
#Encoding gender
X[:,9] = le.fit_transform(X[:,9])
#Encoding native-country
X[:,13] = le.fit_transform(X[:,13])


from sklearn.preprocessing import OneHotEncoder
Ohe = OneHotEncoder(categorical_features=[1, 3, 5, 6, 7, 8, 9, 13])
X = Ohe.fit_transform(X)

X = X.toarray()

y = le.fit_transform(y)

#split dataset into test cases and train cases
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

log_reg.score(X_train,y_train)
log_reg.score(X_test,y_test)
log_reg.score(X,y)

y_pred = log_reg.predict(X) 

#make confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)


from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y,y_pred)
recall_score(y,y_pred)
f1_score(y,y_pred)















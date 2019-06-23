import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv('NLP/bank-data.csv',na_values='unknown')

dataset.info()
dataset.describe()
dataset.isnull().sum()

X = dataset.iloc[:,0:20].values
y = dataset.iloc[:,-1].values

temp = pd.DataFrame(X[:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 14]])

temp[0].value_counts()
temp[1].value_counts()
temp[2].value_counts()
temp[3].value_counts()
temp[4].value_counts()
temp[5].value_counts()
temp[6].value_counts()
temp[7].value_counts()
temp[8].value_counts()
temp[9].value_counts()


temp[0] = temp[0].fillna('admin.')
temp[1] = temp[1].fillna('married')
temp[2] = temp[2].fillna('university.degree')
temp[3] = temp[3].fillna('no')
temp[4] = temp[4].fillna('yes')
temp[5] = temp[5].fillna('no')
temp[6] = temp[6].fillna('cellular')
temp[7] = temp[7].fillna('may')
temp[8] = temp[8].fillna('thu')
temp[9] = temp[9].fillna('nonexistent')


temp.isnull().sum()

X[:,[1, 2, 3, 4, 5, 6, 7, 8, 9 ,14]] = temp
del(temp)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#Encode job
X[:,1] = le.fit_transform(X[:,1])
#pd.DataFrame(X)

#encode marital
X[:,2] = le.fit_transform(X[:,2])
#encode education
X[:,3] = le.fit_transform(X[:,3])
#encode default
X[:,4] = le.fit_transform(X[:,4])
#encode housing
X[:,5] = le.fit_transform(X[:,5])
#encode loan
X[:,6] = le.fit_transform(X[:,6])
#encode contact
X[:,7] = le.fit_transform(X[:,7])
#encode month
X[:,8] = le.fit_transform(X[:,8])
#encode day_of_week
X[:,9] = le.fit_transform(X[:,9])
#encode poutcome
X[:,14] = le.fit_transform(X[:,14])

from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder(categorical_features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14])
X = OHE.fit_transform(X)
X = X.toarray()
y = le.fit_transform(y)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)








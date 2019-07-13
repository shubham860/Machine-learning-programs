import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('titanic.csv')

X = dataset.iloc[:,[3,4,5,6,7,11]].values
y = dataset.iloc[:,1].values

temp = pd.DataFrame(X[:,[2,5]])
temp[0].value_counts()
temp[1].value_counts()

temp[1].isnull().sum()

temp[0] = temp[0].fillna(28.56)
temp[1] = temp[1].fillna('S')

X[:,[2,5]] = temp
del(temp)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X[:,0] = le.fit_transform(X[:,0])
X[:,1] = le.fit_transform(X[:,1])
X[:,5] = le.fit_transform(X[:,5])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0,1,5])
X = ohe.fit_transform(X)
X = X.toarray()


from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()

from sklearn.svm import SVC
svm = SVC()

from sklearn.ensemble import VotingClassifier
vc = VotingClassifier([('LR',log_reg),
                        ('KNN',knn),
                        ('DTC',dtc),
                        ('NB',n_b),
                        ('SVM',svm)])

vc.fit(X_train,y_train)
vc.score(X_train,y_train)
vc.score(X_test,y_test)

from sklearn.ensemble import BaggingClassifier
bc = BaggingClassifier(n_b, n_estimators=7)
bc.fit(X_train,y_train)
bc.score(X_train,y_train)
bc.score(X_test,y_test)

from sklearn.ensemble import RandomForestClassifier
rc = RandomForestClassifier(n_estimators = 5)
rc.fit(X_train,y_train)
rc.score(X_train,y_train)
rc.score(X_test,y_test)
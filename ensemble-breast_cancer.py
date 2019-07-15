import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

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
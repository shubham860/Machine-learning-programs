import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

#split dataset into test cases and train cases
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=3)
knc.fit(X_train,y_train)
knc.score(X_train,y_train)


knc.score(X_test,y_test)
knc.score(X,y)s
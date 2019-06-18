#0=Sepal Length, 1=Sepal Width, 2=Petal Length,3=Petal Width
#0=satosa,1=versicolor,2=verginica
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

plt.scatter(X[y==0,0],X[y==0,1],c="r", label="satosa")
plt.scatter(X[y==1,0],X[y==1,1],c="g", label="versicolor")
plt.scatter(X[y==2,0],X[y==2,1],c="b", label="verginica")
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.title('Analysis on the iris Dataset')
plt.show()

plt.scatter(X[y==0,2],X[y==0,3],c="r", label="satosa")
plt.scatter(X[y==1,2],X[y==1,3],c="g", label="versicolor")
plt.scatter(X[y==2,2],X[y==2,3],c="b", label="verginica")
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.title('Analysis on the iris Dataset')
plt.show()

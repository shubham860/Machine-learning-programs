import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston
dataset = load_boston()

X = dataset.data
y = dataset.target

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X, y)
lr.score(X, y)
plt.scatter(X, y)
plt.plot(X, lg.predict(X), c="r")
plt.show()

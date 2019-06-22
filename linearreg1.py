import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = np.random.randn(100)
y = 3 * X + 6

plt.scatter(X,y)
plt.show()

X = np.c_[X, np.ones(100)]

theta = np.linalg.inv(X.T @ X) @ (X.T @ y )
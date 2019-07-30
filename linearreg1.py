import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = np.random.randn(100)
y = 3 * X + 6 + np.random.randn(100)


plt.scatter(X,y)
plt.show()

X = np.c_[X, np.ones(100)]

theta = np.linalg.inv(X.T @ X) @ (X.T @ y )



dataset = pd.read_excel('blood.xlsx')

X = dataset.iloc[2:,1].values
y = dataset.iloc[2:,-1].values
X = X.reshape(-1,1)

dataset.info()

plt.scatter(X,y)
plt.title('Linear Regresssion of Blood Group')
plt.xlabel('Age')
plt.ylabel('Systolic Blood Pressure')
plt.show()


from sklearn.linear_model import LinearRegression
lg = LinearRegression()

lg.fit(X, y)
lg.score(X, y)

plt.scatter(X, y)
plt.plot(X, lg.predict(X), c="r")
plt.show()

lg.predict([[20]])

lg.coef_
lg.intercept_

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#GENERATING FAKE DATA
m = 100
X = 6 * np.random.randn(m ,1) - 3  #-3 FOR MAKING CURVE 
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m,1)

#PLOTING FAKE DATA
plt.scatter(X,y)
plt.axis([-3, 3, 0, 9])   #for zoom B/w -3 and 3 on x-axis and b/w 0 to 9 on y-axis
plt.show()

#FOR SQUARING X**2
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2,include_bias = False)  #if include_bias is true then extra 1's column is addes in matrix
X_poly = poly.fit_transform(X)

#APPLY LINEAR REGRESSION ON X_poly
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_poly , y)

#fake data X_new
X_new = np.linspace(-3,3,100).reshape(-1,1)  #to generate 100 random values b/w 3 and -3 and then reshape to make it a vector
X_new_poly = poly.fit_transform(X_new)  #SQUARE 
y_new = lr.predict(X_new_poly)  

plt.scatter(X,y)
plt.plot(X_new,y_new, c="r")
plt.axis([-3,3,0,9])
plt.show()

lr.coef_
lr.intercept_

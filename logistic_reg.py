import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x =  np.arange(-10,10,0.01)
sig = 1/(1 + np.power(np.e, -x))
line = 4 * x + 7
sig_1 = (np.power(np.e, -x))/(1 + np.power(np.e, -x)) 

plt.plot(x,sig)
plt.show()


plt.plot(x,sig_1)
plt.show()


plt.plot(x,line)
plt.show()

sig_line = 1/(1 + np.power(np.e, -line))
plt.plot(x,sig_line)
plt.show()



#Logistic Regression 

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target


#split dataset into test cases and train cases
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)  #split in 75 and 25 without parameter test_size but according to this parameter it spplits in 80 and 20


#import logistic regression and fit the dataset
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

log_reg.score(X_train,y_train) #score of train dataset
log_reg.score(X_test,y_test) #score of test dataset
log_reg.score(X,y) #score of whole dataset

y_pred = log_reg.predict(X) # o/p of 569 datasets entity

#make confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)


from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y,y_pred)
recall_score(y,y_pred)
f1_score(y,y_pred)

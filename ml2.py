import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('NLP/Data_Pre.csv')

X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,-1].values

#ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
#from sklearn.impute import SimpleImputer
#sim = SimpleImputer(missing_values='NaN',strategy='mean') 
#sim.fit(X[:,0:2])

from sklearn.preprocessing import Imputer
sim = Imputer(missing_values='NaN',strategy='mean')
sim.fit(X[:,0:2])
X[:,0:2] = sim.transform(X[:,0:2])   #sim.transform_fit create copy only so we have to store it in X


#now to remove the state name and coverted them into the no's LIKE 0 = kerala,1=MP
from sklearn.preprocessing import LabelEncoder
Lbl_en = LabelEncoder()
X[:,2] = Lbl_en.fit_transform(X[:,2])
Lbl_en.classes_ #to check the coverted city into no's
y = Lbl_en.fit_transform(y)
#now to give the same priority to each city we use Dummy Trap Encoding by OneHotEncoding

from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder(categorical_features = [2])
X = OHE.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)
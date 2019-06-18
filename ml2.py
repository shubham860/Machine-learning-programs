import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('')
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,-1].values

#cause error NaN or no is to large
from sklearn.impute import SimpleImputer
sim = SimpleImputer(missing_values='NaN', strategy='mean')

from sklearn.impute import Imputer
sim = Imputer(missing_values='NaN', strategy='mean')
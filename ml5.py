# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:57:59 2019

@author: SHUBHAM
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston
dataset = load_boston()

X = dataset.data
y = dataset.target

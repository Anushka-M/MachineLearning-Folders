# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 19:58:57 2017

@author: AnushkaM
"""
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

dataSet = pd.read_csv('Data_update.csv')
X = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, 3].values

#for missing data
from sklearn.preprocessing import Imputer
imputer  = Imputer(missing_values='NaN',strategy = 'mean', axis=0)
imputer  = imputer.fit(X[:,1:3])
X[: ,1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencode = OneHotEncoder(categorical_features= [0])
X = onehotencode.fit_transform(X).toarray()

labelencoder_y= LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test,  y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=0)
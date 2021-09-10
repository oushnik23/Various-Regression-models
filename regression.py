# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:03:43 2021

@author: Administrator
"""

import os
os.getcwd()
os.chdir("C:/Users/Administrator/Desktop/PYTHON/Regression")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Data_2.csv")
dataset.columns
x=dataset[['AT', 'V', 'AP', 'RH']]
y=dataset[['PE']]

dataset.isnull().any()
dataset.isnull().sum()
x.apply(lambda x:x.isnull().sum()/len(x)*100)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2, random_state=0)

#Multiple Linear Regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)

y_pred=model.predict(xtest)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), ytest.values.reshape(len(ytest),1)),1))

from sklearn.metrics import r2_score
r2_score(ytest,y_pred)


#RANDOM_Forest
from sklearn.ensemble import RandomForestRegressor
model2=RandomForestRegressor(n_estimators=10, random_state=0)
model2.fit(xtrain,ytrain)

y_pred2=model2.predict(xtest)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred2.reshape(len(y_pred2),1), ytest.values.reshape(len(ytest),1)),1))

from sklearn.metrics import r2_score
r2_score(ytest,y_pred2)

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
model3=DecisionTreeRegressor()
model3.fit(xtrain,ytrain)

y_pred3=model3.predict(xtest)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred3.reshape(len(y_pred3),1),ytest.values.reshape(len(ytest),1)),1))

from sklearn.metrics import r2_score
r2_score(y_pred3,ytest)

#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
pl=PolynomialFeatures(degree=4)
x_poly=pl.fit_transform(xtrain)
from sklearn.linear_model import LinearRegression
model4=LinearRegression()
model4.fit(x_poly,ytrain)

y_pred4=model4.predict(pl.fit_transform(xtest))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred4.reshape(len(y_pred4),1),ytest.values.reshape(len(ytest),1)),1))

from sklearn.metrics import r2_score
r2_score(y_pred4,ytest)

#Support Vector Regression

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
ytrain=sc_y.fit_transform(ytrain)

from sklearn.svm import SVR
model5=SVR(kernel='rbf')
model5.fit(xtrain,ytrain)

y_pred5=sc_y.inverse_transform(model5.predict(sc_x.fit_transform(xtest)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred5.reshape(len(y_pred5),1),ytest.values.reshape(len(ytest),1)),1))

from sklearn.metrics import r2_score
r2_score(y_pred5,ytest)



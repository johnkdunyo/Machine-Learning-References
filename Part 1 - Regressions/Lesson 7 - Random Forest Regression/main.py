#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 15:49:07 2022

@author: jondexter
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#for polynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def preprocessor(data):
    dataset = data.values
    #alwasy ensure X is a matrix and y is a vector
    X = dataset[:, 1].astype(int)
    X = X.reshape(len(X), 1)
    y = dataset[:, -1].astype(float)
    return X, y



#engine
data = pd.read_csv('Position_Salaries.csv')
X, y = preprocessor(data)

#noneed to split data because the dataset is very small
#we first build a linear regression model and then a polynomial then we compare the two
lr_regressor = LinearRegression()
lr_regressor.fit(X, y)


#now we buld the polynomial regressor
pl_regressor = PolynomialFeatures(degree=4)
#transform X to a polynomial term
X_poly = pl_regressor.fit_transform(X)

#create a new linear regression model and fit it to the X_poly
lr_regressor2 = LinearRegression()
lr_regressor2.fit(X_poly, y)



#decion tree regressor
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X, y)


#random foreset 
rf_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rf_regressor.fit(X, y)



#visualising linear polynomial regrssionmmodel
plt.scatter(X, y, c='red')
plt.plot(X, lr_regressor.predict(X), color='yellow') #comparing real salary to observation of the predicted salary
plt.plot(X, lr_regressor2.predict(X_poly), color='green') #comparing real salary to observation of the predicted salary
plt.plot(X, dt_regressor.predict(X), color='blue')
plt.plot(X, rf_regressor.predict(X), color='black')
plt.title('Salary vs Experience (Lower Resolution)')
plt.xlabel('Level(Position)')
plt.ylabel('Salary ($)')
plt.show()


#visualization for higher resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
#plt.plot(X_grid, lr_regressor.predict(X_grid), color='yellow') #comparing real salary to observation of the predicted salary
plt.plot(X, lr_regressor2.predict(X_poly), color='green') #comparing real salary to observation of the predicted salary
plt.plot(X_grid, dt_regressor.predict(X_grid), color='blue')
plt.plot(X_grid, rf_regressor.predict(X_grid), color='black')
plt.title('Salary vs Experience (Higher Resolution)')
plt.xlabel('Level(Position)')
plt.ylabel('Salary ($)')
plt.show()










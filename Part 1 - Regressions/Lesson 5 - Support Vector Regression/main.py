#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 14:24:06 2022

@author: jondexter
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#for polynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

def preprocessor(data):
    dataset = data.values
    #alwasy ensure X is a matrix and y is a vector
    X = dataset[:, 1].astype(int)
    X = X.reshape(len(X), 1)
    y = dataset[:, -1].astype(float)
    return X, y


def preprocessor_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)
    return X_train, X_test, y_train, y_test

def feature_scalling(X, y):
    y = y.reshape(len(y), 1)
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X = sc_x.fit_transform(X)
    y = sc_y.fit_transform(y)
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



#SVM goes here
#wait, we need to future scale this data before proceeding
#X_sc, y_sc = feature_scalling(X, y)
svr_regressor = make_pipeline(StandardScaler(), SVR(C=1.0, kernel='rbf', epsilon=0.2, degree=4))
svr_regressor.fit(X, y)

'''
need to look into the svr again, probaly when you get a good internet 
'''


#visualising linear polynomial regrssionmmodel
plt.scatter(X, y, c='red')
plt.plot(X, lr_regressor.predict(X), color='blue') #comparing real salary to observation of the predicted salary
plt.plot(X, lr_regressor2.predict(X_poly), color='green') #comparing real salary to observation of the predicted salary
plt.plot(X, svr_regressor.predict(X), color='yellow')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Level(Position)')
plt.ylabel('Salary ($)')
plt.show()

#increasing the degree increaces the accuracy of the model
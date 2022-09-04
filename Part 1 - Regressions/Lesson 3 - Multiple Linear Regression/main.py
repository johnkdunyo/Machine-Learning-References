#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 14:47:29 2022

@author: jondexter
"""

#multiple linear regression

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm



def preprocessor(data):
    dataset = data.values
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(float)
    return X, y


def preprocessor_cat(X):
    #state column has categorical data, so we need to process it before our ml model can work on it
    #using onehotencoder
    ct = ColumnTransformer([('OneHotEncoder', OneHotEncoder(), [3])], remainder='passthrough')
    X_enc = ct.fit_transform(X).astype(float)
    return X_enc

def avoiding_dummy_variable_trap(X):
    #need to remove at least one dummy independent variable
    #i think the library wr gon use will take care of that
    X = X[:, 1:]
    return X


def preprocessor_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)
    return X_train, X_test, y_train, y_test

def backward_elimation(X, y):
    #append arrays of 1 to the X array, since the library we using doesn factor the constant
    X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
    #this matrix contains all the most important variables.. then we remove one by one those arent significant...
    X_opt = X[:, [0,1,2,3,4,5]].astype('float64')
    #step2
    #new object for ordinary least squares  fit afterwards
    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    #step3 -> predictor with highest P-value
    summ =regressor_OLS.summary()
    print(summ)
    
    #from summary table we need to remove indexes: 2,1,4,5
    X_opt = X[:, [0,3]].astype('float64')
    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    summ =regressor_OLS.summary()
    print(summ)
    
    return X, X_opt






#engine 
data = pd.read_csv('50_Startups.csv')
X, y = preprocessor(data)
X_enc = preprocessor_cat(X)

X_enc = avoiding_dummy_variable_trap(X_enc)

X_train, X_test, y_train, y_test = preprocessor_split(X_enc, y)
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#predictions 
y_pred = regressor.predict(X_test)


#model building
X_new, X_opt = backward_elimation(X_enc, y)















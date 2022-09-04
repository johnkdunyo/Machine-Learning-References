#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 13:37:27 2022

@author: jondexter
"""

#simple Linear Regression model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def preprocess_data(data):
    #convert data to nd array
    dataset = data.values
    
    #split into dependent and independent
    X = dataset[:,0]
    y = dataset[:,1]
    #reshaping X
    X =X.reshape(len(X), 1)
    return X, y


def preprocess_data_2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=30)
    return X_train, X_test, y_train, y_test


#future scalling
#def future_scalling_process()

#def visualise_


    

#engine
data = pd.read_csv("Salary_Data.csv")
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = preprocess_data_2(X, y)

#defininig the model and fittig it
regressor  = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


#visualising
plt.scatter(X_train, y_train, c='red')
plt.plot(X_train, regressor.predict(X_train), color='blue') #comparing real salary to observation of the predicted salary
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Experience (years')
plt.ylabel('Salary ($)')
plt.show()


plt.scatter(X_test, y_test, c='red')
plt.plot(X_train, regressor.predict(X_train), color='blue') #comparing real salary to observation of the predicted salary
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Experience (years')
plt.ylabel('Salary ($)')
plt.show()





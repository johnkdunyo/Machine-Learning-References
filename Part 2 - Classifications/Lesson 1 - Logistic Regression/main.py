#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 02:35:23 2022

@author: jondexter
"""

import pandas  as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib as plt
from matplotlib.colors import ListedColormap

def preprocess_1(data):
    dataset = data.values
    X = dataset[:, 2:4]
    y = dataset[:, -1].astype(int)
    return X, y

def preprocess_2_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)
    return X_train, X_test, y_train, y_test 

def feature_scaling(X):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X


def plot_some_graph(X_test, y_test, classifier):
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() -1, stop=X_set[:, 0].max() +1, step=0.01 ),
                         np.arange(start=X_set[:, 1].min() -1, stop=X_set[:, 1].max() +1, step=0.01 ))

    plt.contour(X1, X2, classifier.predict( np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Logistic Regression (Testset)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
    
    
    

data = pd.read_csv('Social_Network_Ads.csv')
X, y = preprocess_1(data)
X = feature_scaling(X)
X_train, X_test, y_train, y_test = preprocess_2_split(X, y)


classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)


#visualising the data
#plot_some_graph(X_test, y_test, classifier)
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() -1, stop=X_set[:, 0].max() +1, step=0.01 ),
                     np.arange(start=X_set[:, 1].min() -1, stop=X_set[:, 1].max() +1, step=0.01 ))


vv = np.array([X1.ravel(), X2.ravel()]).T
vv_2 = classifier.predict(vv).reshape()
'''
plt.contour(X1, X2, classifier.predict( np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Testset)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
'''




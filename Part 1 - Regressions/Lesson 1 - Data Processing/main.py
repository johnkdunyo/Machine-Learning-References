#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:38:11 2022

@author: jondexter
"""

import numpy as np
import matplotlib as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer



myImputer = SimpleImputer()





#fuction to prepare dataset
def prepare_data(data):
    dataset = data.values
    #split data into both input X and output Y
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    Y = Y.reshape(len(Y), 1)
    
    #X has nan, replace with mean using the sklearn preprocessing imputer
    myImputer.fit(X[:, 1:3])
    X[:, 1:3] = myImputer.transform(X[:, 1:3])
    return X, Y


def prepare_part2(X, Y):
    #categorical encoding of first column using the onehotencoder
    #we gon use the make_column_transformer to select only the fist column and apply the cat..
    transformer = make_column_transformer(
                    (OneHotEncoder(), [0]), remainder='passthrough' )
    X_enc = transformer.fit_transform(X)
    
    #y is a mater of binary classification; so we use ordinalLabel
    oe = OrdinalEncoder()
    Y_enc = oe.fit_transform(Y)
    return X_enc, Y_enc

def split_dataset(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=30) 
    return X_train, X_test, Y_train, Y_test


def future_scale(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    return X_train, X_test

    #
#engine

data = pd.read_csv("Data.csv")
X, Y = prepare_data(data)

X_enc, Y_enc = prepare_part2(X, Y)

X_train, X_test, Y_train, Y_test = split_dataset(X_enc, Y_enc)

X_train, X_test = future_scale(X_train, X_test)








#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:42:48 2017

@author: joshuasomma
"""

# ------> data pre-processing template <-------

# import libs

# maths operations (particularly arrays); 2nd line sets numpy to display a whole array instead of ...
import numpy as np
np.set_printoptions(threshold=np.inf)
# data/chart visualiser
import matplotlib.pyplot as plt
# import and manage large datasets
import pandas as pd

# --->import the dataset<---
dataset = pd.read_csv('Data.csv')
# extract the first 3 cols of dataset and put them into an array; these are the independant variables
X = dataset.iloc[:,:-1].values
# extract the dependent variable and place into its own array
y = dataset.iloc[:,-1].values
                
# --->take care of missing data<---
# removing rows w missing data is bad practice so we will calculate the mean of the column and replace the missing data with this value
# import Imputer library
from sklearn.preprocessing import Imputer
# instantiate imputer object and pass params to define how we want to fix missing data; axis = 0 means cols (1 would mean rows)
imputer = Imputer(missing_values= 'NaN', strategy='mean', axis=0)
# pass data set through imputer; 1st param sets to all rows; 2nd sets to only cols 2 and 3
imputer = imputer.fit(X[:,1:3])
# replace missing data points with the mean of the column
X[:,1:3] = imputer.transform(X[:,1:3])

# --->encoding categorical data<---
# LabelEncoder is used to convert text category labels  (like 'France' etc.) into numeric values
# import the lib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Instantiate the object
labelencoder_X = LabelEncoder()
# Transform the countries into numerical values
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# But there is a problem, the values are weighted
# OneHotEncoder will encode the category values into their own array where each cat has a value of 1 or 0 against a given cat
# instantiate the object
onehotencoder = OneHotEncoder(categorical_features = [0])
# transform the object, create a sub-array with cat values of 1/0 for each value
X = onehotencoder.fit_transform(X).toarray()

# Do the same for the dependant variable column y/n
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# --->splitting the data between training and test sets<---
# We need to be able to test the predictions using some uncorellated data so we split out 20% of the table to use as test data once the system is trained on the train data
from sklearn.cross_validation import train_test_split
# set up some new vars and set up params for test_size and random_state; random state is used to normalise the numbers
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

# --->feature scaling<---
# different features might have different ranges; i.e. age v salary, you might have 25-50 age range but 25k to 90k salary range; the range on numbers are out of sync and need to be normalised
# without scaling the features, the larger range will dominate the calculations
# based on Euclidian geometry and calculating distance between two points

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)





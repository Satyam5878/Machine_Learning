import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from Utils import *

base_dir = os.path.split(os.getcwd())[0]
print(base_dir)
names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income',
]

## LogisticRegression
"""
adult_data = pd.read_csv(base_dir+"/data/adult.data.txt",names=names)                       #Loading train data
X = adult_data.iloc[:,:-1]                                                                  #Features
y = adult_data.iloc[:,-1]                                                                   #Target
census = Pipeline([('encoder',CategoryEncoder(get_categorical_columns(X))),
                  ('imputer',CategoryImputer(['occupation','workclass','native_country'])),
                  ('clf',LogisticRegression())])                                            #Pipeline
y_encoder = LabelEncoder().fit(y)                                                           #encoder for target
census.fit(X,y_encoder.transform(y))                                                        #running pipeline

adult_data_test = pd.read_csv(base_dir+"/data/adult.test.txt",names=names)                  #Loading test data
X_test = adult_data_test.iloc[:,:-1]                                                        #Features
y_test = adult_data_test.iloc[:,-1]                                                          #Target

y_pred = census.predict(X_test)                                                             #Prediction
y_test = y_encoder.transform([y.rstrip('.') for y in y_test])                               #Transforming y_test
#print(y_test.head())

#print(y,y_pred)
print(classification_report(y_test,y_pred))

print("Done")
"""

from sklearn.svm import SVC

## SVM
print("SVM")
print(1)
adult_data = pd.read_csv(base_dir+"/data/adult.data.txt",names=names)                       #Loading train data
X = adult_data.iloc[:,:-1]                                                                  #Features
y = adult_data.iloc[:,-1] 
print(2)                                                                  #Target
census = Pipeline([('encoder',CategoryEncoder(get_categorical_columns(X))),
                  ('imputer',CategoryImputer(['occupation','workclass','native_country'])),
                  ('clf',SVC())])                                            #Pipeline
y_encoder = LabelEncoder().fit(y)  
print(3)                                                         #encoder for target
census.fit(X,y_encoder.transform(y))                                                        #running pipeline
print(4)
adult_data_test = pd.read_csv(base_dir+"/data/adult.test.txt",names=names)                  #Loading test data
X_test = adult_data_test.iloc[:,:-1]                                                        #Features
y_test = adult_data_test.iloc[:,-1]                                                          #Target
print(5)
y_pred = census.predict(X_test)                                                             #Prediction
y_test = y_encoder.transform([y.rstrip('.') for y in y_test])                               #Transforming y_test
#print(y_test.head())
print(6)
#print(y,y_pred)
print(classification_report(y_test,y_pred))

print("Done")

"""
print("NB")
from sklearn.naive_bayes import GaussianNB
## Naivebayes
print(1)
adult_data = pd.read_csv(base_dir+"/data/adult.data.txt",names=names)                       #Loading train data
X = adult_data.iloc[:,:-1]                                                                  #Features
y = adult_data.iloc[:,-1] 
print(2)                                                                  #Target
census = Pipeline([('encoder',CategoryEncoder(get_categorical_columns(X))),
                  ('imputer',CategoryImputer(['occupation','workclass','native_country'])),
                  ('clf',GaussianNB())])                                            #Pipeline
y_encoder = LabelEncoder().fit(y)  
print(3)                                                         #encoder for target
census.fit(X,y_encoder.transform(y))                                                        #running pipeline
print(4)
adult_data_test = pd.read_csv(base_dir+"/data/adult.test.txt",names=names)                  #Loading test data
X_test = adult_data_test.iloc[:,:-1]                                                        #Features
y_test = adult_data_test.iloc[:,-1]                                                          #Target
print(5)
y_pred = census.predict(X_test)                                                             #Prediction
y_test = y_encoder.transform([y.rstrip('.') for y in y_test])                               #Transforming y_test
#print(y_test.head())
print(6)
#print(y,y_pred)
print(classification_report(y_test,y_pred))

print("Done")
"""
"""
## KNN
print("KNN")
from sklearn.neighbors import KNeighborsClassifier
print(1)
adult_data = pd.read_csv(base_dir+"/data/adult.data.txt",names=names)                       #Loading train data
X = adult_data.iloc[:,:-1]                                                                  #Features
y = adult_data.iloc[:,-1] 
print(2)                                                                  #Target
census = Pipeline([('encoder',CategoryEncoder(get_categorical_columns(X))),
                  ('imputer',CategoryImputer(['occupation','workclass','native_country'])),
                  ('clf',KNeighborsClassifier())])                                            #Pipeline
y_encoder = LabelEncoder().fit(y)  
print(3)                                                         #encoder for target
census.fit(X,y_encoder.transform(y))                                                        #running pipeline
print(4)
adult_data_test = pd.read_csv(base_dir+"/data/adult.test.txt",names=names)                  #Loading test data
X_test = adult_data_test.iloc[:,:-1]                                                        #Features
y_test = adult_data_test.iloc[:,-1]                                                          #Target
print(5)
y_pred = census.predict(X_test)                                                             #Prediction
y_test = y_encoder.transform([y.rstrip('.') for y in y_test])                               #Transforming y_test
#print(y_test.head())
print(6)
#print(y,y_pred)
print(classification_report(y_test,y_pred))

print("Done")

"""







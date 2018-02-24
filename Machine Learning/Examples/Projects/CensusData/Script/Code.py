# Import Stmts
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.pipeline import PipeLine
from sklearn.liner_model import LogisticRegression

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


# Loading data

adult_data = pd.read_csv(base_dir+"/data/adult.data.txt",names=names)
#print(adult_data.head(20))


# Analysis:

#test_analysis


##print(adult_data[column])


# getting unique category in each coloumn:






# Testing the label encoder
"""
lbl_enc = LabelEncoder()
lbl_enc.fit(X["workclass"])
print((lbl_enc.transform(X["workclass"])))
print(lbl_enc.classes_)
print("Done")
"""
#print(get_categorical_columns(adult_data))

adult_data = CategoryEncoder(get_categorical_columns(adult_data)).fit_transform(adult_data)

# Splittng the data into X and y

X = adult_data.iloc[:,:-1]
y = adult_data.iloc[:,-1]


##print(X.head())
#print(y.head())

"""
imputer = Imputer(missing_values=0,strategy='most_frequent')
X['native_country']=imputer.fit_transform(X[['native_country']])
print(X.head())
"""

X = CategoryImputer(['occupation','workclass','native_country']).fit_transform(X)

"""
print(X.tail(10))
print(X['native_country'].unique())
print(X['native_country'].count())
print("Done")
"""

## Entire Pipeline
adult_data = pd.read_csv(base_dir+"/data/adult.data.txt",names=names)
X = adult_data.iloc[:,:-1]
y = adult_data.iloc[:,-1]
census = PipeLine(('encoder',CategoryEncoder(get_category_columns(X))),
                  ('imputer',CategoryImputer(['occupation','workclass','native_country'])),
                  ('clf',LogisticRegression()))
census.fit(X,LabelEncoder().fit_transform(y))





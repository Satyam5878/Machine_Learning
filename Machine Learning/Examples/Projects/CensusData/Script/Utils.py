from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
import pickle
import os

class CategoryEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,columns=None):
        self.columns = columns
        self.encoders = None
    def fit(self,data,target=None):
        if self.columns is None:
            self.columns = data.columns
        #print(self.columns)
        self.encoders = { column:LabelEncoder().fit(data[column]) for column in self.columns}
        return  self
    def transform(self,data):
        output = data.copy()
        for column, encoder in self.encoders.items():
            output[column] = encoder.transform(data[column])
            #print(column+" ")
            #print(encoder.classes_)
        return output

class CategoryImputer(BaseEstimator,TransformerMixin):
    def __init__(self,columns=None):
        self.columns = columns
        self.imputer = None
    def fit(self,data,target=None):
        if self.columns is None:
            self.columns = data.columns
        self.imputer = Imputer(missing_values=0,strategy='most_frequent')
        self.imputer.fit(data[self.columns])
        return self
    def transform(self,data):
        output = data.copy()
        output[self.columns] = self.imputer.transform(data[self.columns])
        return output
          
def get_categorical_columns(data):
    categorical = [column for column in data.columns if data[column].dtype == 'object'] 
    return categorical
 
 
def dump_model(model,path,name='model'):
    with open(os.path.join(path,name+"_"+time.strftime("%Y_%m_%d_%I_%M_%S",time.localtime())+".pickle"),'wb') as file:
        pickle.write(model,file)
        
        
def load_model(path):
    with open(path,'rb') as file:
        pickle.load(file)
       









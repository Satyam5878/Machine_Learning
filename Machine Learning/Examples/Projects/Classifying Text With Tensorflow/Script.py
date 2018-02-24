import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
import re

#loading dataset

#categories to load
categories = ["comp.graphics","sci.space","rec.sport.baseball"]

newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)

#print(len(newgroups_train.data))

vocab = Counter()

for text in newsgroups_train.data:
    text = re.sub(r'[^a-zA-Z \\n]'," ",text)
    for word in text.split(' '):
        #print(word)
        vocab[word.lower()]+=1

for text in newsgroups_test.data:
    text = re.sub(r'[^a-zA-Z \\n]'," ",text)
    for word in text.split(' '):
        #print(word)
        vocab[word.lower()]+=1
       
     
print(len(vocab))  
print(vocab)    
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       

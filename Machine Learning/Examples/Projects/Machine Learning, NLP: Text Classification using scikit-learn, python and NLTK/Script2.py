from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline


import numpy as np
import sys

from sklearn.model_selection import GridSearchCV


twenty_train = fetch_20newsgroups(subset='train',shuffle=True)
pipe = Pipeline([('vect',CountVectorizer(stop_words = 'english')),(   'tfidf',TfidfTransformer()),("clf",MultinomialNB())])
txt_clf = pipe.fit(twenty_train.data,twenty_train.target)
print(type(txt_clf))

twenty_test = fetch_20newsgroups(subset='test',shuffle=True)
predicted = txt_clf.predict(twenty_test.data)
print("accuray of NaiveBayes "+str(np.mean(predicted == twenty_test.target)))

parameters = {'vect__ngram_range':[(1,1),(1,2)],'tfidf__use_idf':(True,False),"clf__alpha":(1e-2,1e-3)}

gcv = GridSearchCV(txt_clf,parameters,n_jobs=-1)
gcv = gcv.fit(twenty_train.data,twenty_train.target)
print("Best Score :"+str(gcv.best_score_))
print("Best Param "+str(gcv.best_params_))

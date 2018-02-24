## Import stmts;
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline


import numpy as np
import sys

## Loading datasets;



"""
#print(count_vect.get_feature_names())
l = count_vect.get_feature_names()
alpha = []
alpha_numeric = []
for item in l:
    if item.isalpha():
        alpha.append(item)
    else:
        alpha_numeric.append(item)

print((alpha))
print(len(alpha_numeric))    
"""

"""
#print(len(twenty_train.data))
#print((twenty_train.data[0]))
#print(twenty_train.data[0])
cv = CountVectorizer()

#print(type(cv.fit_transform([twenty_train.data[0]])))
#print(cv.get_feature_names()[24])



count_vect = CountVectorizer()                             # will hold the terms in form of features
document_term_matrix = count_vect.fit_transform(twenty_train.data)
tfidf_trans = TfidfTransformer()
X_tfidf = tfidf_trans.fit_transform(document_term_matrix)

##print((X_tfidf))
mNB_clf = MultinomialNB().fit(X_tfidf,twenty_train.target)

print("DOne")



sys.exit(1)
# some tests
#print(twenty_train.target_names)



#print("\n".join(twenty_train.data[0].split("\n")[:3])) 
#print(type(twenty_train.data))
print("Done")

"""

twenty_train = fetch_20newsgroups(subset='train',shuffle=True)
# Naive Bayes 
"""
twenty_train = fetch_20newsgroups(subset='train',shuffle=True)
pipe = Pipeline([('vect',CountVectorizer()),(   'tfidf',TfidfTransformer()),("clf",MultinomialNB())])
txt_clf = pipe.fit(twenty_train.data,twenty_train.target)
print(type(txt_clf))

twenty_test = fetch_20newsgroups(subset='test',shuffle=True)
predicted = txt_clf.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))
"""

from sklearn.linear_model import SGDClassifier

"""
pipe = Pipeline([('count',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',SGDClassifier())])
txt_clf = pipe.fit(twenty_train.data,twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test',shuffle=True)
predicted = txt_clf.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))
"""

## grid search

from sklearn.model_selection import GridSearchCV


twenty_train = fetch_20newsgroups(subset='train',shuffle=True)
pipe = Pipeline([('vect',CountVectorizer()),(   'tfidf',TfidfTransformer()),("clf",MultinomialNB())])
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



print("\n\n\n\n")
pipe = Pipeline([('vect',CountVectorizer()),(   'tfidf',TfidfTransformer()),("clf",SGDClassifier())])
txt_clf = pipe.fit(twenty_train.data,twenty_train.target)
print(type(txt_clf))

twenty_test = fetch_20newsgroups(subset='test',shuffle=True)
predicted = txt_clf.predict(twenty_test.data)
print("accuray of SGDClassifer "+str(np.mean(predicted == twenty_test.target)))

parameters = {'vect__ngram_range':[(1,1),(1,2)],'tfidf__use_idf':(True,False),"clf__alpha":(1e-2,1e-3)}

gcv = GridSearchCV(txt_clf,parameters,n_jobs=-1)
gcv = gcv.fit(twenty_train.data,twenty_train.target)
print("Best Score :"+str(gcv.best_score_))
print("Best Param "+str(gcv.best_params_))






print("Done")
























from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

count_vec = CountVectorizer()

print(count_vec.fit_transform(corpus))
print(count_vec.get_feature_names())

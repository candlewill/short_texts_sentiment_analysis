__author__ = 'NLP-PC'
from load_data import load_anew, load_extend_anew
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

def get_IDF_topn_words(data=[], n=3, vocabulary=None):
    vect = TfidfVectorizer(vocabulary=vocabulary)
    vect.fit_transform(data)
    indices = np.argsort(vect.idf_)[::-1] # idf_ and tfidf could also be used
    features = vect.get_feature_names()
    top_features = [features[i] for i in indices[:n]]
    return top_features

d=['i lave you', 'i love you', 'i and she', 'go to school with she', 'he and']
print(get_IDF_topn_words(data=d, n=5, vocabulary=None))
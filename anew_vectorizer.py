__author__ = 'NLP-PC'
from sklearn.base import BaseEstimator
import numpy as np
import nltk
from load_data import load_extend_anew
from statistics import mean
from nltk import word_tokenize

class anew_vectorizer(BaseEstimator):
    # 返回特征名称列表（list），包含用transform()返回的所有的特征
    def __init__(self):
        self.words, self.arousal, self.valence = load_extend_anew()
        self.stemmer = nltk.stem.SnowballStemmer('english')
        self.max = 9
        self.stemmed_dict = [self.stemmer.stem(w) for w in self.words]

    def get_feature_names(self):
        return np.array(
            ['max_valence', 'avg_valence', 'min_valence', 'max_arousal', 'avg_arousal', 'min_arousal']
        )

    # As we are not implementing a classifier, we can ignore this one and simply return self.
    def fit(self, documents, y=None):
        return self

    def _get_VA(self, d):
        print('Stemming, Still working...')
        english_stemmer = self.stemmer
        stemmed_sent = [english_stemmer.stem(w) for w in word_tokenize(d)]
        valence_value = []
        arousal_value = []
        words, valence, arousal = self.words, self.valence, self.arousal
        overlapping_words = (set(stemmed_sent) & set(self.stemmed_dict))
        if len(overlapping_words) != 0:
            for word in overlapping_words:
                ind = self.stemmed_dict.index(word)
                valence_value.append(valence[ind])
                arousal_value.append(arousal[ind])
            max_valence = max(valence)
            avg_valence = mean(valence)
            min_valence = min(valence)
            max_arousal = max(arousal)
            avg_arousal = mean(arousal)
            min_arousal = min(arousal)
        else:
            # if nothing mathes, the default value is 4.5
            default = 4.5
            max_valence = default
            avg_valence = default
            min_valence = default
            max_arousal = default
            avg_arousal = default
            min_arousal = default
        return np.array([max_valence, avg_valence, min_valence, max_arousal, avg_arousal, min_arousal])

    # This returns numpy.array(), containing an array of shape (len(documents), len(get_feature_names)).
    # This means that for every document in documents, it has to return a value for every feature name in get_feature_names().
    def transform(self, documents):
        max_valence, avg_valence, min_valence, max_arousal, avg_arousal, min_arousal = np.array(
            [self._get_VA(d) for d in documents]).T

        result = np.array(
            [max_valence, avg_valence, min_valence, max_arousal, avg_arousal, min_arousal]).T
        return result/self.max

        # fit_transform is no need to be completed, watch out!
        # def fit_transform(self, documents):
        #     return self.fit(documents).transform(documents)

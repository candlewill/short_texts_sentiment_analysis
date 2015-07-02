__author__ = 'NLP-PC'
from sklearn.pipeline import FeatureUnion
from feature_generating import StemmedTfidfVectorizer
from anew_vectorizer import anew_vectorizer
from preprocess import preprocessor
from parameters import parameters
from load_data import load_train_data
from load_data import load_test_data
from save_data import dump_picle
import numpy as np
from vectorizers import punctuation_estimator
from logger_manager import log_state

vectorizer_param={'preprocessor': preprocessor, 'ngram_range': parameters['ngram_range'], 'analyzer':'word',
                                    'min_df':parameters['min_df'], 'max_df': parameters['max_df'],
                                    'binary': parameters['TF_binary'], 'norm': parameters['norm'],'sublinear_tf': parameters['sublinear_tf'], 'max_features': parameters['max_features']}

if __name__ == "__main__":
    unigram = StemmedTfidfVectorizer(**vectorizer_param)
    anew = anew_vectorizer()
    pct = punctuation_estimator()
    # combined_features =FeatureUnion([('unigram',unigram),('anew',anew)])
    log_state('combine unigram and punctuation features')
    combined_features =FeatureUnion([('unigram',unigram),('pct',pct)])
    texts, _=load_train_data('Sentiment140')

    transformed_train=combined_features.fit_transform(texts)

    testdata, _ = load_test_data()
    transformed_test=combined_features.transform(testdata)

    dump_picle(combined_features.get_feature_names(), './data/features/feature_names.p')
    dump_picle(transformed_train, "./data/transformed_data/transformed_train.p")
    dump_picle(transformed_test, "./data/transformed_data/transformed_test.p")
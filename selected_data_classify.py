__author__ = 'hs'
__author__ = 'hs'
__author__ = 'NLP-PC'
import feature_generating
import classifiers
import analysis
from load_data import load_train_data, load_processed_data
from load_data import load_test_data
from save_data import dump_picle
from vectorizers import TFIDF_estimator, anew_estimator
from analysis import analysis_result
from classifiers import mNB
from load_data import load_selected_data

print('Start')
vectorizer = TFIDF_estimator()
texts, train_labels = load_selected_data(data_type='train')
transformed_train = vectorizer.fit_transform(texts)
testdata, true_labels = load_selected_data(data_type='test')
transformed_test = vectorizer.transform(testdata)

predict = mNB(transformed_train, train_labels, transformed_test)

analysis_result(predict, true_labels)

__author__ = 'NLP-PC'
__author__ = 'NLP-PC'
import feature_generating
import classifiers
import analysis
from load_data import load_train_data
from load_data import load_test_data
from save_data import dump_picle
from vectorizers import TFIDF_estimator, anew_estimator
from analysis import analysis_result
from classifiers import mNB

print('Start')
vectorizer = TFIDF_estimator()
train_type = 'Sentiment140'
texts, train_labels = load_train_data(train_type)
transformed_train = vectorizer.fit_transform(texts)
testdata, true_labels = load_test_data()
transformed_test = vectorizer.transform(testdata)

predict = mNB(transformed_train, train_labels, transformed_test)

analysis_result(predict, true_labels)
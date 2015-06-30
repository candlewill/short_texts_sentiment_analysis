__author__ = 'NLP-PC'
from sklearn.naive_bayes import MultinomialNB
from save_data import dump_picle
from load_data import load_pickle, load_train_data
import logging
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mNB(train_data, train_labels, test):
    clf = MultinomialNB()
    clf.fit(train_data, train_labels)
    predict_labels = clf.predict(test)
    predict_proba = clf.predict_proba(test)
    dump_picle(predict_labels, './data/predict_labels/predict_labels.p')
    dump_picle(predict_proba, './data/predict_labels/predict_proba.p')
    logger.info('Classifier training complete, saved predict labels to pickle')
    return


def svm_classify(train_data, train_labels, test):
    clf = svm.SVC(C=5.0, kernel='linear')
    clf.fit(train_data, train_labels)
    predict_labels = clf.predict(test)
    dump_picle(predict_labels, './data/predict_labels/predict_labels.p')
    logger.info('SVM classifier training complete, saved predict labels to pickle')
    return


def logit(train_data, train_labels, test):
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(train_data, train_labels)
    predict_labels = clf.predict(test)
    dump_picle(predict_labels, './data/predict_labels/predict_labels.p')
    logger.info('MaxEnt classifier training complete, saved predict labels to pickle')
    return


def kNN(train_data, train_labels, test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(train_data, train_labels)
    predict_labels = clf.predict(test)
    dump_picle(predict_labels, './data/predict_labels/predict_labels.p')
    logger.info('kNN classifier training complete, saved predict labels to pickle')
    return


train_data = load_pickle('./data/transformed_data/transformed_train.p')
test = load_pickle('./data/transformed_data/transformed_test.p')
_, train_labels = load_train_data()
kNN(train_data, train_labels, test)

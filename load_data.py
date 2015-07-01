__author__ = 'NLP-PC'
# coding: utf-8
import logging
import csv
from parameters import parameters
import pickle
import os

# configure the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_train_data(data_type='Sentiment140'):
    logging.info('Start loading data')
    texts = []
    labels = []
    if data_type == 'Sentiment140':
        inpTweets = csv.reader(
            open('./data/traindata/Sentiment140/' + str(parameters['test_data_size']) + '.csv', 'rt', encoding='utf8'),
            delimiter=',')
        for row in inpTweets:
            sentiment = (1 if row[0] == '4' else 0)
            tweet = row[5]
            labels.append(sentiment)
            texts.append(tweet)
    logging.info('Load data finished')
    return texts, labels


def load_test_data(classify_type=None):
    logging.info('Start load test data')
    file_name = 'data/testdata/testdata.manual.2009.06.14.csv'
    raw_data = csv.reader(open(file_name, 'rt', encoding='utf8'), delimiter=',', quotechar='"')
    tweets, sentiment = [], []
    for line in raw_data:
        if line[0] == '0' or line[0] == '4':
            tweets.append(line[5])
            sentiment.append(0 if line[0] == '0' else 1)
    logging.info('Test data loading complete')
    return tweets, sentiment


def load_pickle(filename):
    out = pickle.load(open(filename, "rb"))
    return out
    logging.info('Load pickle data complete')


def load_anew():
    data_dir = './data/lexicon/'
    with open(os.path.join(data_dir, "anew_seed.txt"), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        words, arousal, valence = [], [], []
        for line in reader:
            words.append(line[0])
            arousal.append(float(line[1]))
            valence.append(float(line[2]))
    return words, arousal, valence


def load_extend_anew(D=False):
    data_dir = './data/lexicon/'
    with open(os.path.join(data_dir, "extend_anew.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        words, arousal, valence, dominance = [], [], [], []
        for line in reader:
            if reader.line_num == 1:
                continue
            words.append(line[1])
            arousal.append(float(line[5]))
            valence.append(float(line[2]))
            if D == True:
                dominance.append(float(line[8]))
    if D == True:
        return words, arousal, valence, dominance
    else:
        return words, arousal, valence

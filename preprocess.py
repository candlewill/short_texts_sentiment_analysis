__author__ = 'NLP-PC'
# encoding: utf-8
import re, const_values as const
import string
from save_data import csv_save
from os_check import get_os_name
import nltk
from nltk import word_tokenize

def preprocessor(tweet):
    emo_repl_order = const.emo_repl_order
    emo_repl = const.emo_repl
    re_repl = const.re_repl

    tweet = tweet.lower()
    for k in emo_repl_order:
        tweet = tweet.replace(k, emo_repl[k])

    for r, repl in re_repl.items():
        tweet = re.sub(r, repl, tweet)

    # remove all punctuation
    for c in string.punctuation:
        tweet = tweet.replace(c, " ")
    tweet = re.sub(r" +", " ", tweet).strip()

    english_stemmer = nltk.stem.SnowballStemmer('english')
    tweet = ' '.join([english_stemmer.stem(w) for w in word_tokenize(tweet)])

    return tweet


def preprocess_tweeets(tweets_list, tweets_labels, filename):
    def isEnglish(s):
        try:
            s.encode('ascii')
        except UnicodeEncodeError:
            return False
        else:
            return True

    processed_texts = []
    for line, l in zip(tweets_list, tweets_labels):
        if isEnglish(line):
            processed_texts.append((l, preprocessor(line)))
        # else: # print or not ?
        #     print(line)

    os_name = get_os_name()
    if os_name == 'windows':
        file_dir = 'C:/Corpus/'
    elif os_name == 'ubuntu':
        file_dir = '/home/hs/Data/'
    else:
        return
    csv_save(processed_texts, file_dir + filename)


if __name__ == '__main__':

    from load_data import load_test_data
    test_texts, test_labels =load_test_data()
    preprocess_tweeets(test_texts, test_labels, 'preprocessed_test_data_359.csv')
    exit()

    from load_data import load_train_data
    texts, labels = load_train_data()
    processed_texts = []
    preprocess_tweeets(texts, labels, 'preprocessed_training_data_160000.csv')

__author__ = 'NLP-PC'
# coding: utf-8
import logging
import csv
from parameters import parameters
import pickle
import os
from gensim.models import Word2Vec
import time as time
from os_check import get_os_name

# configure the logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def load_train_data(data_type='Sentiment140'):
    logging.info('Start loading data')
    texts = []
    labels = []
    if data_type == 'Sentiment140':
        if parameters['test_data_size'] == 160000:
            os_name = get_os_name()
            if os_name == "windows":
                file_name = 'C:/Corpus/training.csv'
            elif os_name == 'ubuntu':
                file_name = '/home/hs/Data/Corpus/training.csv'
        else:
            file_name = './data/traindata/Sentiment140/' + str(parameters['test_data_size']) + '.csv'
        inpTweets = csv.reader(
            open(file_name, 'rt', encoding='ISO-8859-1'),  # Please watch out the encoding format
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
    logging.info('Loading anew lexicon')
    data_dir = './data/lexicon/'
    with open(os.path.join(data_dir, "anew_seed.txt"), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        words, arousal, valence = [], [], []
        for line in reader:
            words.append(line[0])
            arousal.append(float(line[1]))
            valence.append(float(line[2]))
    logging.info('Loading anew lexicon completed')
    return words, arousal, valence


def load_extend_anew(D=False):
    logging.info('Loading extend_anew lexicon')
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
    logging.info('Loading extend_anew lexicon complete')
    if D == True:
        return words, arousal, valence, dominance
    else:
        return words, arousal, valence


def load_word_embedding(data_name='google_news', data_type='bin'):
    logger.info('Start load word2vec word embedding')
    os_name = get_os_name()
    if os_name == "windows":
        file1 = 'D:/Word_Embeddings/GoogleNews-vectors-negative300.bin.gz'
        file2 = 'D:/Word_Embeddings/freebase-vectors-skipgram1000.bin.gz'
        file3 = 'D:/Word_Embeddings/GoogleNews-vectors-negative300.bin'
        file4 = 'D:/Word_Embeddings/freebase-vectors-skipgram1000.bin'
    elif os_name == 'ubuntu':
        file1 = '/home/hs/Data/Word_Embeddings/GoogleNews-vectors-negative300.bin.gz'
        file2 = '/home/hs/Data/Word_Embeddings/freebase-vectors-skipgram1000.bin.gz'
        file3 = '/home/hs/Data/Word_Embeddings/google_news.bin'
        file4 = '/home/hs/Data/Word_Embeddings/freebase.bin'
    if data_name == 'google_news':
        if data_type == 'bin':
            model = Word2Vec.load_word2vec_format(file3, binary=True)
        else:  # load .bin.gz data
            model = Word2Vec.load_word2vec_format(file1, binary=True)
    else:  # load freebase
        if data_type == 'bin':
            model = Word2Vec.load_word2vec_format(file4, binary=True)
        else:
            model = Word2Vec.load_word2vec_format(file2, binary=True)

    # using gzipped/bz2 input works too, no need to unzip:
    logging.info('Loading word embedding complete')
    return model


def load_processed_data(data_type='train', stem=True):
    logging.info('Start Loading Data')
    if stem == True:
        if data_type == 'train':
            if parameters['test_data_size'] == 160000:
                os_name = get_os_name()
                if os_name == "windows":
                    file_name = 'C:/Corpus/preprocessed_training_data_1600000.csv'
                elif os_name == 'ubuntu':
                    file_name = '/home/hs/Data/Corpus/preprocessed_training_data_1600000.csv'
            else:
                file_name = './data/traindata/Sentiment140/pre-processed/preprocessed_training_data_' + str(
                    parameters['test_data_size']) + '.csv'
        elif data_type == 'test':
            file_name = './data/testdata/preprocessed_test_data_359.csv'
    elif stem == False:
        if data_type == 'train':
            if parameters['test_data_size'] == 160000:
                os_name = get_os_name()
                if os_name == "windows":
                    file_name = 'C:/Corpus/preprocessed_training_data_nostem_160000.csv'
                elif os_name == 'ubuntu':
                    file_name = '/home/hs/Data/Corpus/preprocessed_training_data_nostem_160000.csv'
            else:
                file_name = './data/traindata/Sentiment140/pre-processed/preprocessed_training_data_nostem_' + str(
                    parameters['test_data_size']) + '.csv'
        elif data_type == 'test':
            file_name = './data/testdata/preprocessed_test_data_nostem_359.csv'

    with open(file_name, 'r', encoding= 'ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        text, label = [], []
        for line in reader:
            text.append(line[1])
            label.append(int(line[0]))
    logging.info('Load Data Completed')
    return text, label

if __name__ == "__main__":
    st = time.time()
    # examplenshu
    model = load_word_embedding()

    print(model['computer'])
    elapsed_time = time.time() - st
    print("Elapsed time: %.3fmin" % (elapsed_time / 60))
    # cosine similarity
    print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
    elapsed_time = time.time() - st
    print("Elapsed time: %.3fmin" % (elapsed_time / 60))
    print(model.doesnt_match("breakfast cereal dinner lunch".split()))
    elapsed_time = time.time() - st
    print("Elapsed time: %.3fmin" % (elapsed_time / 60))
    print(model.similarity('woman', 'man'))
    elapsed_time = time.time() - st
    print("Elapsed time: %.3fmin" % (elapsed_time / 60))
    # Find the top-N most similar words, using the multiplicative combination objective proposed by Omer Levy and Yoav Goldberg in [4]. Positive words still contribute positively towards the similarity, negative words negatively, but with less susceptibility to one large distance dominating the calculation.
    # In the common analogy-solving case, of two positive and one negative examples, this method is equivalent to the “3CosMul” objective (equation (4)) of Levy and Goldberg.
    # Additional positive or negative examples contribute to the numerator or denominator, respectively – a potentially sensible but untested extension of the method. (With a single positive example, rankings will be the same as in the default most_similar.)
    print(model.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'], topn=10))
    elapsed_time = time.time() - st
    print("Elapsed time: %.3fmin" % (elapsed_time / 60))
    # Compute cosine similarity between two sets of words.
    print(model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant']))
    elapsed_time = time.time() - st
    print("Elapsed time: %.3fmin" % (elapsed_time / 60))

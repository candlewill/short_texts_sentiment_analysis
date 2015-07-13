from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from load_data import load_train_data, load_processed_data
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# The following skills is useful
# train_test_split(np.array(texts), np.array(sentiemnt), test_size=0.2)

x_train, y_train = load_processed_data(stem=False)
x_test, y_test = load_processed_data(data_type='test', stem=False)

from preprocess import preprocessor as preprocess

n_dim = 300
scaling = False


# Build word vector for training set by using the average value of all word vectors in the tweet, then scale
from load_data import load_word_embedding

imdb_w2v = load_word_embedding()


def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


from sklearn.preprocessing import scale

train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
if scaling == True:
    train_vecs = scale(train_vecs)

# Train word2vec on test tweets
# imdb_w2v.train(x_test)

# Build test tweet vectors then scale
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
if scaling == True:
    test_vecs = scale(test_vecs)

# scaling to [0, 1] interval
min_max_scaler = MinMaxScaler()
train_vecs = min_max_scaler.fit_transform(train_vecs)
test_vecs = min_max_scaler.fit_transform(test_vecs)

# Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
from classifiers import gNB, mNB
from analysis import analysis_result

pre = mNB(train_vecs, y_train, test_vecs)
analysis_result(pre, y_test)

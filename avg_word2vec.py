from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from load_data import load_train_data
import numpy as np

texts, sentiemnt = load_train_data()
x_train, x_test, y_train, y_test = train_test_split(np.array(texts), np.array(sentiemnt), test_size=0.2)

from preprocess import preprocessor as preprocess
# Do some very minor text preprocessing
def cleanText(corpus):
    # corpus = [z.lower().replace('\n', '').split() for z in corpus]
    corpus = [preprocess(z) for z in corpus]
    return corpus


x_train = cleanText(x_train)
x_test = cleanText(x_test)

n_dim = 300

# I had the same problem with 64bit machine. I think it is related to the version of Python (32 bit vs 64 bit). To solve the problem in this project, I passed my own hash function as a parameter to word2vec constructor:
def hash32(value):
    return hash(value) & 0xffffffff

# Initialize model and build vocab
imdb_w2v = Word2Vec(size=n_dim, min_count=10, hashfxn=hash32)
imdb_w2v.build_vocab(x_train)

# Train the model over train_reviews (this may take several minutes)
imdb_w2v.train(x_train)

# Build word vector for training set by using the average value of all word vectors in the tweet, then scale
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
train_vecs = scale(train_vecs)

# Train word2vec on test tweets
imdb_w2v.train(x_test)

# Build test tweet vectors then scale
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)

# Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
from sklearn.linear_model import SGDClassifier

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print('Test Accuracy: %.2f' % lr.score(test_vecs, y_test))

# Create ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

pred_probas = lr.predict_proba(test_vecs)[:, 1]

fpr, tpr, _ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')

plt.show()

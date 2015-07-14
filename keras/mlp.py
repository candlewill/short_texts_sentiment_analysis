__author__ = 'hs'
import keras
import os
import sys
import logging

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    # initialize constant variable
    batch_size = 128
    nb_classes = 2
    nb_epoch = 20

    logging.info("loading training data...")

    from load_data import load_qrcode
    from sklearn.cross_validation import train_test_split

    features, labels = load_qrcode()
    features = features.astype('float32')
    labels = np_utils.to_categorical(labels, 2)
    (trainData, testData, trainLabel, testY) = train_test_split(features, labels, test_size=0.2)
    # print(testData[2])
    # print(testY[2])
    # exit()
    model = Sequential()
    model.add(Dense(2025, 512, init="uniform"))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(512, 512, init="uniform"))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(512, 128, init="uniform"))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(128, 128, init="uniform"))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(128, 2, init="uniform"))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    model.fit(trainData, trainLabel, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)

    logging.info("Training process finished!")
    logging.info("Predict for testData...")

    score = model.evaluate(testData, testY, batch_size=batch_size)
    print(score)
    exit()

    y_pred = model.predict(testData, )
    testLabel = np.argmax(y_pred, axis=1)

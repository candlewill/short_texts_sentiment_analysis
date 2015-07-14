__author__ = 'hs'
import keras
import os
import sys
import logging

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, MaxPooling2D

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

    trainData = trainData.reshape(trainData.shape[0], 1, 45, 45)
    testData = testData.reshape(testData.shape[0], 1, 45, 45)

    model = Sequential()

    model.add(Convolution2D(45, 1, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(45, 45, 4, 4))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(21780, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    model.fit(trainData, trainLabel, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)
    score = model.evaluate(testData, testY, batch_size=batch_size)
    print(score)
    exit()

    logging.info("Training process finished!")
    logging.info("Predict for testData...")
    y_pred = model.predict(testData, )
    testLabel = np.argmax(y_pred, axis=1)

    # logging.info("Save result...")
    # util.saveResult(testLabel, './result/keras_cnn_result.csv')

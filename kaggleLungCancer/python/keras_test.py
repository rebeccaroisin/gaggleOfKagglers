#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
from keras.optimizers import SGD

import numpy as np

IMAGE_X = 80
IMAGE_Y = 95
IMAGE_Z = 95

sample = np.zeros((1, IMAGE_X, IMAGE_Y, IMAGE_Z))
X_train = np.array([sample for _ in range(100)])
y_train = np.array([i%2 for i in range(100)])

model = Sequential()

model.add(Convolution3D(
    32, 3, 3, 3,
    dim_ordering='th',
    border_mode='valid',
    input_shape=(1, IMAGE_X, IMAGE_Y, IMAGE_Z)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Convolution3D(32, 3, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Convolution3D(32, 3, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, nb_epoch=10, verbose=1)
model.evaluate(X_train, y_train, verbose=1)

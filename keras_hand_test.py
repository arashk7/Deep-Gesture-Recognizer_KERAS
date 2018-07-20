import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import h5py
from keras.models import load_model
from test_hand_h5.AkTools import excel_write, calc_error

from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


def ak_test():
    h5f = h5py.File('hand_train_32_S4.h5', 'r')
    X = h5f['X'].value
    Y = h5f['Y'].value

    h5f = h5py.File('hand_test_32_S4.h5', 'r')
    X_test = h5f['X'].value
    Y_test = h5f['Y'].value

    # model = VGG16()
    model=Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model = load_model('model_S4.h5')
    # model.fit(X, Y, batch_size=32, epochs=10)
    # score = model.evaluate(X, Y, batch_size=32)
    # print(score)

    p = model.predict(X_test, batch_size=30, verbose=1)
    excel_write('test_S4.xls', p, 30)
    precent = calc_error(p, 30)
    return precent

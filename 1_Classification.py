#!/usr/bin/env python


    
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

import numpy as np

### load the data
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
print(len(train_images))

### data normalization
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype("float32") / 255


if 1:
### NN model and loss function
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential([layers.Dense(512,activation='relu'),
    layers.Dense(10,activation='softmax')])
    model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    ### train the model
    model.fit(train_images,train_labels,epochs = 5, batch_size=128)

    # ### test the image
    # test_img = test_images[0:2][:]
    # test_labl = test_labels[0:2]
    # pred_labl = model.predict(test_img)
    # ### test the error
    # print(f"Predicted error {test_labl[0] - pred_labl[0].argmax()}") 
    # #sample test
    # test_loss, test_accuracy = model.evaluate(test_img,test_labl)
    # print('Test Loss: ', test_loss)
    # print('Test Accuracy: ',test_accuracy*100)

    #complete test images
    test_loss, test_accuracy = model.evaluate(test_images,test_labels)
    print('Test Loss: ', test_loss)
    print('Test Accuracy: ',test_accuracy*100)
else:

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adadelta
    from tensorflow.keras.losses import categorical_crossentropy


    (train_images,train_labels),(test_images,test_labels) = mnist.load_data()

    img_rows, img_cols=28, 28
    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images /= 255
    test_images /= 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    inpx = (img_rows, img_cols, 1)
    inpx = Input(shape=inpx)
    layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx)
    layer2 = Conv2D(64, (3, 3), activation='relu')(layer1)
    layer3 = MaxPooling2D(pool_size=(3, 3))(layer2)
    layer4 = Dropout(0.5)(layer3)
    layer5 = Flatten()(layer4)
    layer6 = Dense(250, activation='sigmoid')(layer5)
    layer7 = Dense(10, activation='softmax')(layer6)

    model = Model([inpx], layer7)
    model.compile(optimizer=Adadelta(),
			loss=categorical_crossentropy,
			metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5, batch_size=500)

    score = model.evaluate(test_images, test_labels, verbose=0)
    print('loss=', score[0])
    print('accuracy=', score[1])




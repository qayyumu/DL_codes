#!/usr/bin/env python

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

### load the data
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
print(len(train_images))

### NN model and loss function
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([layers.Dense(512,activation='relu'),
layers.Dense(10,activation='softmax')])
model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

### data normalization
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype("float32") / 255


### train the model
model.fit(train_images,train_labels,epochs = 5, batch_size=128)

### test the image
test_img = test_images[0:2][:]
test_labl = test_labels[0:2]
pred_labl = model.predict(test_img)

### test the error
print(f"Predicted error {test_labl[0] - pred_labl[0].argmax()}") 
#sample test
test_loss, test_accuracy = model.evaluate(test_img,test_labl)
print('Test Loss: ', test_loss)
print('Test Accuracy: ',test_accuracy*100)

#complete test images
test_loss, test_accuracy = model.evaluate(test_images,test_labels)
print('Test Loss: ', test_loss)
print('Test Accuracy: ',test_accuracy*100)


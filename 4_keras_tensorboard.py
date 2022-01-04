#!/usr/bin/env python
### Tensorflow keras with tensorboard and classification example
import numpy as np
import tensorflow as tf

from sklearn.datasets import make_classification,make_circles

x,Y = make_classification(n_features=2,n_classes=2,n_samples=100,n_redundant=0,n_clusters_per_class=1)
#x,Y = make_circles(n_samples=100,noise=0.03,factor=0.7)


tf.random.set_seed(42)

#### we have to show how the model enhanced with layer performs on circular data
model = tf.keras.Sequential([
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)


import time
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time.time()))
# tensorboard --logdir=logs


history = model.fit(x, Y, epochs=200,callbacks=[tensorboard])
accuracy = max(history.history.get('accuracy'))*100
# from mlxtend.plotting import plot_decision_regions
# import matplotlib.pyplot as plt
# plot_decision_regions(x, Y, clf=model, legend=2)
# plt.suptitle('Accuracy'+str(accuracy))
# plt.show()
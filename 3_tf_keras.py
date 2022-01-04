#!/usr/bin/env python

import numpy as np
import tensorflow as tf
x = tf.ones(shape = (2,2))
print(x)

y0 = np.zeros(shape=(2,2))   ## numpy vs tf initialization
y = tf.zeros(shape = (2,2))
print(y)

xx = tf.random.normal(shape=(2,2), mean=0., stddev=1.)
print(xx)

## tensorflow variable and on the fly assignment
v = tf.Variable(initial_value=tf.random.normal(shape=(3,1)))
v.assign(tf.ones((3,1)))
v[0,0].assign(np.random.rand())  ## add some value at fixed location
v.assign_add(tf.ones((3,1)))


### tf basic maths
x = tf.ones(shape=(2,2))
y = tf.square(x)
z = tf.matmul(x,y)
z1 = tf.add(x,y)

### calculate tf gradient
def Grad_Tape(input):
    with tf.GradientTape() as tape:
            tape.watch(input)    ## to handle the tf.const
            result = tf.square(input)
    grad = tape.gradient(result,input)
    return grad        

input_var = tf.Variable(initial_value=5.)
print(Grad_Tape(input_var))
print(Grad_Tape(input_var).numpy())

input_var = tf.constant(5.)
print(Grad_Tape(input_var))


### calculate second order gradient
time = tf.Variable(initial_value=2.)

with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2
    speed = inner_tape.gradient(position,time)
acceleration = outer_tape.gradient(speed,time)

print(f'Position {position} Speed {speed} Acceleration {acceleration}')


#### classification example with tensorflow
from sklearn.datasets import make_classification,make_circles

x,Y = make_classification(n_features=2,n_classes=2,n_samples=100,n_redundant=0,n_clusters_per_class=1)
#x,Y = make_circles(n_samples=100,noise=0.03,factor=0.7)


import tensorflow as tf
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
history = model.fit(x, Y, epochs=100)
accuracy = max(history.history.get('accuracy'))*100
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

plot_decision_regions(x, Y, clf=model, legend=2)
plt.suptitle('Accuracy'+str(accuracy))
plt.show()
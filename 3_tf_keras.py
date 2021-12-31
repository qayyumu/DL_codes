#!/usr/bin/env python


import tensorflow as tf
x = tf.ones(shape = (2,2))
print(x)

y = tf.zeros(shape = (2,2))
print(y)

xx = tf.random.normal(shape=(2,2), mean=0., stddev=1.)
print(xx)
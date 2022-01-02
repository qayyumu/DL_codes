#! \usr\bin\env python
### Sigmoid on line equation
###
import numpy as np
import matplotlib.pyplot as plt

def line(m,c,x):
         return (m*x+c)

def sigmoid(x):
    z = (1./(1. + np.exp(-x)))
    return z



m = 2;c = 0
x = np.linspace(-12, 12, 100)
y = line(m,c,x)
y_max = max(y)

plt.subplot(1,2,1)
plt.plot(x,y,'r')
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(1,2,2)
plt.plot(x,0.5+y/y_max,'r')
plt.plot(x,sigmoid(y),'g')
plt.xlabel('x')
plt.ylabel('y_squeeze,sig(y)')
plt.suptitle('Sigmoid Decision Boundry')
plt.show()
# print(sigmoid(y))


import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

y_sig = tf.math.sigmoid(y)
y_sig = y_sig.numpy()

y_tan = tf.math.tanh(y)
y_tan = y_tan.numpy()

plt.subplot(1,2,1)
plt.plot(x,0.5+y/y_max,'r')
plt.plot(x,y_sig,'g')
plt.xlabel('x')
plt.ylabel('y_squeeze,sig(y)')

plt.subplot(1,2,2)
plt.plot(x,y/y_max,'r')
plt.plot(x,y_tan,'g')
plt.xlabel('x')
plt.ylabel('y_squeeze,tan(y)')
plt.suptitle('Sigmoid n Tanh Decision Boundry')
plt.show()


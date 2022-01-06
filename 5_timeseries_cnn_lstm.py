#!/usr/bin/env python
### Forecasting timeseries with 1D CNN and LSTM
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.python.keras.layers.core import Activation

skip_plot =5  ### Plot strides

url = 'https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'

openpower_germany_df = pd.read_csv(url, sep=',', index_col=0, 
                             parse_dates=[0]) 
openpower_germany_df.columns

openpower_germany_df['Consumption'][::skip_plot].plot(marker='*')
plt.xlabel('time')
plt.ylabel('electricity consumption')
plt.show()
### get the consumption as numpy array
consumption_energy = openpower_germany_df['Consumption'].to_numpy()
consumption_energy.shape

### process the data for training and testing
def make_data(time_series,step_x,step_y):
    x = list()
    Y = list()
    for i in range(len(time_series)):
        ind_x = i + step_x
        ind_y = ind_x + step_y
        if (ind_y>len(time_series)):  #as step_y can be big and bounding condition
            break

        seq_x, seq_y = time_series[i:ind_x], time_series[ind_x:ind_y]
        x.append(seq_x)
        Y.append(seq_y)
    return x,Y


step_x = 25
step_y = 1

x,Y = make_data(consumption_energy,step_x,step_y)
x = np.array(x)
Y = np.array(Y)
feature_in = 1
x = x.reshape(x.shape[0],x.shape[1],feature_in)
x.shape,Y.shape
### now we can apply different algorithms


#Average
def avg_baseline(x):
    return np.mean(x,axis=1)

Y_pred_avg = avg_baseline(x)


plt.plot(Y[::skip_plot],alpha=0.5,color='r')
plt.plot(Y_pred_avg[::skip_plot],'b.')
plt.legend(['True','Avg-pred'])
plt.show()
r2_score(Y, Y_pred_avg)*100.

#### RNN, 1D conv model

from tensorflow.keras.layers import Conv1D, Dense,LSTM, GRU,SimpleRNN, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

model = Sequential()

model_try = 1

batch_size =16
epochs = 15


if(model_try==1): ###simple RNN
        model.add(SimpleRNN(100,activation='relu',input_shape=(step_x,feature_in)))
elif(model_try==2):## GRU
        model.add(GRU(100,activation='relu',input_shape=(step_x,feature_in),return_sequences=True))
        model.add(GRU(100,activation='relu',return_sequences=False))
        # model.add(BatchNormalization())
elif(model_try==3):  ### 1D conv
        
        model.add(tf.keras.layers.Conv1D(filters=200, kernel_size=3, padding="same",input_shape=(step_x,feature_in)))
        model.add(tf.keras.layers.ReLU())
        
        model.add(tf.keras.layers.Conv1D(filters=200, kernel_size=3, padding="same"))
        model.add(tf.keras.layers.ReLU())
        
        model.add(tf.keras.layers.Conv1D(filters=200, kernel_size=3, padding="same"))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.GlobalAveragePooling1D())
   


    # gap = keras.layers.GlobalAveragePooling1D()(conv3)

    # output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)
elif(model_try==4):
        model.add(LSTM(100,activation='relu',input_shape=(step_x,feature_in),return_sequences=True))
        model.add(LSTM(100,activation='relu',return_sequences=False))
        # model.add(BatchNormalization())

# 

# model.add(BatchNormalization())

model.add(Dense(step_y))

model.compile(optimizer='adam',loss='mse',metrics=['mean_squared_error'])

model.summary()


history = model.fit(x,Y,epochs=epochs,verbose=1)

Y_pred_rnn = model.predict(x)

print(r2_score(Y,Y_pred_rnn)*100)

plt.plot(Y[::skip_plot],alpha=0.5,color='r')
plt.plot(Y_pred_rnn[::skip_plot],'b.')
plt.legend(['True','RNN-pred'])
plt.show()

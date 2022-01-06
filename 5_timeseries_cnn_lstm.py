#!/usr/bin/env python
### Forecasting timeseries with 1D CNN and LSTM
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'

openpower_germany_df = pd.read_csv(url, sep=',', index_col=0, 
                             parse_dates=[0]) 
openpower_germany_df.columns

openpower_germany_df['Consumption'][::5].plot(marker='*')

### get the consumption as numpy array
consumption_energy = openpower_germany_df['Consumption'].to_numpy()
consumption_energy.shape

### process the data for training and testing



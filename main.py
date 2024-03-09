import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from keras import layers
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
# from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# avoid warnings when running
# import logging
# tf.get_logger().setLevel(logging.ERROR)

# import training data
training_set = pd.read_csv('Google_Stock_Price_Train.csv', index_col = 'Date', parse_dates = True)

# preview imported dataset, check missing values and display info
training_set.head()
training_set.isna().any()
training_set.info()

# preview training dataset
training_set['Open'].plot(figsize = (16,6))

# convert column from string to float
training_set['Volume'] = training_set['Volume'].str.replace(',', '').astype(float)

# feature scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# import timestamps into data
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# creating LSTM(RNN) model
regressor = Sequential()

# adding first LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# adding second LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# adding third LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# adding fourth LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# adding output layer
regressor.add(Dense(units = 1))

# compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# import test data
test_dataset = pd.read_csv('Google_Stock_Price_Test.csv', index_col = 'Date', parse_dates = True)
test_dataset['Volume'] = test_dataset['Volume'].str.replace(',', '').astype(float)
real_stock_price = test_dataset.iloc[:, 1:2].values

# total dataset
dataset_total = pd.concat((training_set['Open'], test_dataset['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(test_dataset) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# price prediction
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price.info()

# result plotting
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
# Part 1: Import packages ===========================================================

# import packages
import tensorflow as tf
import logging

# avoid warnings when running
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler


# Part 2: Import training dataset  ==================================================

# import training data
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_train = pd.read_csv(os.path.join(current_dir, 'datasets', 'Google_Stock_Price_Train.csv'), index_col = 'Date', parse_dates = True)

# preview imported dataset, check missing values and display info
print(dataset_train.head())
print(dataset_train.isna().any())
dataset_train.info()

# convert columns from string to float
dataset_train['Close'] = dataset_train['Close'].str.replace(',', '').astype(float)
dataset_train['Volume'] = dataset_train['Volume'].str.replace(',', '').astype(float)

# preview dataset column which will be used for training
plt.figure(1)
dataset_train['Close'].plot(figsize = (16,6))
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.show()

# use column Close for training only
train_set = dataset_train['Close']
train_set = pd.DataFrame(train_set)

# feature scaling
sc = MinMaxScaler(feature_range = (0, 1))
train_set_scaled = sc.fit_transform(train_set)

# import timestamps into data
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(train_set_scaled[i-60:i, 0])
    y_train.append(train_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 3: Building LSTM model =======================================================

# creating RNN model
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
# epochs = 100, batch_size = 32
regressor.fit(X_train, y_train, epochs = 1, batch_size = 16)


# Part 4: Making predictions and visualising the results ============================

# import test data
dataset_test = pd.read_csv(os.path.join(current_dir, 'datasets', 'Google_Stock_Price_Test.csv'), index_col = 'Date', parse_dates = True)
dataset_test['Volume'] = dataset_test['Volume'].str.replace(',', '').astype(float)

# preview imported dataset, check missing values and display info
print(dataset_test.head())
print(dataset_test.isna().any())
dataset_test.info()

real_stock_price = dataset_test.iloc[:, 1:2].values

# use column Close for test only
test_set = dataset_test['Close']
test_set = pd.DataFrame(test_set)

# total dataset
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# import timestamps into data
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

# reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# price prediction
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price = pd.DataFrame(predicted_stock_price)

# result plotting
plt.figure(2)
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.legend()
plt.show()
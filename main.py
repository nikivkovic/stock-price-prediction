# Part 1: Import packages ===========================================================

# import packages
import tensorflow as tf
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# avoid warnings when running
tf.get_logger().setLevel(logging.ERROR)


# Part 2: Import training dataset  ==================================================

# import training data
current_dir = os.path.dirname(os.path.abspath(__file__))
stock = 'BTC'

if stock == 'BTC':
    dataset_training = pd.read_csv(os.path.join(current_dir, 'datasets', 'BTC_Train.csv'), index_col = 'Date', parse_dates = True)
    train_samples = 300
    timestamp = 20
elif stock == 'TSLA':
    dataset_training = pd.read_csv(os.path.join(current_dir, 'datasets', 'TSLA_Train.csv'), index_col = 'Date', parse_dates = True)
    train_samples = 200
    timestamp = 40
elif stock == 'S&P500':
    dataset_training = pd.read_csv(os.path.join(current_dir, 'datasets', 'S&P500_Train.csv'), index_col = 'Date', parse_dates = True)
    train_samples = 200
    timestamp = 60
else:
    dataset_training = pd.read_csv(os.path.join(current_dir, 'datasets', 'GOOGL_Train.csv'), index_col = 'Date', parse_dates = True)
    train_samples = 1258
    timestamp = 60

# convert columns from string to float
dataset_training['Close'] = dataset_training['Close'].str.replace(',', '').astype(float)

# preview imported dataset, check missing values and display info
print('Training set ==========')
print(dataset_training.head())
print(dataset_training.isna().any())
dataset_training.info()

# preview dataset column which will be used for training
plt.figure(1)
dataset_training['Close'].plot(figsize = (16,6))
plt.xlabel('Time')
plt.ylabel(stock + ' Stock Price')
plt.title(stock + ' Stock Price - Train Set')
# plt.show()

# use only column Close for training
trainining_set = dataset_training['Close']
trainining_set = pd.DataFrame(trainining_set)

# feature scaling
sc = MinMaxScaler(feature_range = (0, 1))
trainining_set_scaled = sc.fit_transform(trainining_set)

# import timestamps into data
X_train = []
y_train = []
for i in range(timestamp, train_samples):
    X_train.append(trainining_set_scaled[i-timestamp:i, 0])
    y_train.append(trainining_set_scaled[i, 0])
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

# model summary
regressor.summary()

# compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# fitting the RNN to the training set
# epochs = 100, batch_size = 32
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)


# Part 4: Making predictions and visualising the results ============================

# import test data
if stock == 'BTC':
    dataset_test = pd.read_csv(os.path.join(current_dir, 'datasets', 'BTC_Test.csv'), index_col = 'Date', parse_dates = True)
    test_samples = 67
elif stock == 'TSLA':
    dataset_test = pd.read_csv(os.path.join(current_dir, 'datasets', 'TSLA_Test.csv'), index_col = 'Date', parse_dates = True)
    test_samples = 53
elif stock == 'S&P500':
    dataset_test = pd.read_csv(os.path.join(current_dir, 'datasets', 'S&P500_Test.csv'), index_col = 'Date', parse_dates = True)
    test_samples = 53
else:
    dataset_test = pd.read_csv(os.path.join(current_dir, 'datasets', 'GOOGL_Test.csv'), index_col = 'Date', parse_dates = True)
    test_samples = 20

# convert columns from string to float
dataset_test['Close'] = dataset_test['Close'].str.replace(',', '').astype(float)

# preview imported dataset, check missing values and display info
print('Testing set ==========')
print(dataset_test.head())
print(dataset_test.isna().any())
dataset_test.info()

# preview dataset column which will be used for training
plt.figure(2)
dataset_test['Close'].plot(figsize = (16,6))
plt.xlabel('Time')
plt.ylabel(stock + ' Stock Price')
plt.title(stock + ' Stock Price - Test Set')
# plt.show()

# get the real stock price (from test dataset)
real_stock_price = dataset_test.iloc[:, 3:4].values
print(real_stock_price.shape)

# use only column Close for the test
test_set = dataset_test['Close']
test_set = pd.DataFrame(test_set)

# total dataset
dataset_total = pd.concat((dataset_training['Close'], dataset_test['Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - timestamp:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# import timestamps into data
X_test = []
for i in range(timestamp, timestamp + test_samples):
    X_test.append(inputs[i-timestamp:i, 0])
X_test = np.array(X_test)

# reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# price prediction
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price = pd.DataFrame(predicted_stock_price)

# prediction result plotting
plt.figure(3)
plt.plot(real_stock_price, color = 'red', label = 'Real ' + stock + ' Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted ' + stock + ' Stock Price')
plt.xlabel('Time')
plt.ylabel(stock + ' Stock Price')
plt.title(stock + ' Stock Price Prediction')
plt.legend()

# get the real stock price (from total dataset)
real_stock_price = dataset_total.values

# add NaN values before the predicted values
temp = predicted_stock_price.values
predicted_stock_price = np.full((train_samples + test_samples,1), np.nan)
predicted_stock_price[train_samples:,] = temp

# prediction result plotting in wider context
plt.figure(4)
plt.plot(real_stock_price, color = 'red', label = 'Real ' + stock + ' Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted ' + stock + ' Stock Price')
plt.xlabel('Time')
plt.ylabel(stock + ' Stock Price')
plt.title(stock + ' Stock Price Prediction')
plt.legend()
plt.show()

# TO-DO:
#TODO: check nmm_cnn.main.py file and look at the compilation of the nn
#TODO: add date in the results plot instead of samples (x-axis)?
#TODO: add timestamp value to legend?
#TODO: add vertical line where the prediction starts on the wider context chart?
#TODO: check other stocks and play with parameters

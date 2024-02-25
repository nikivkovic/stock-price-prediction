import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tfpip

import logging
tf.get_logger().setLevel(logging.ERROR) # da ne bi dolazilo do warning-a u izvrsavanju

from keras import layers
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# print('Hello world!')
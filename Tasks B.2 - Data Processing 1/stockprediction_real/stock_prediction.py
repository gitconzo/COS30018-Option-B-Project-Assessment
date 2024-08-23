# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import keras
import os
from keras import layers

from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer


# ------------------------------------------------------------------------------
# Load and Process Data
# ------------------------------------------------------------------------------

def load_and_process_data(company, start_date, end_date, price_column="Close", prediction_days=60, data_dir="data",
                          scaler_range=(0, 1), features=None, handle_NaN='drop', fill_value=None):
    # Makes sure that the directory for storing the data exists on the computer
    os.makedirs(data_dir, exist_ok=True)

    # Creates filename for csv based on the company name, the start date, and the end date of the data
    data_file = os.path.join(data_dir, f"{company}_{start_date}_{end_date}.csv")

    # Loads data from a CSV if it exists, else it downloads the data utilizing yfinance
    if os.path.exists(data_file):
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    else:
        # Downloading the data based on the company, and defining the parameters for yfinance
        data = yf.download(company, start=start_date, end=end_date)
        # Creates csv file
        data.to_csv(data_file)

    # Handles NaN values
    if handle_NaN == 'drop':
        data.dropna(inplace=True)
    elif handle_NaN == 'fill':
        data.fillna(fill_value, inplace=True)

    # If features aren't provided, it defaults to using only the price column
    if features is None:
        features = [price_column]

    # Makes sure that the price column isn't included in the features list
    if price_column not in features:
        features.append(price_column)

    # Calculates the midpoint (M) of 'Open' and 'Close' prices incase if required
    # M = (A+B)/2
    if price_column == "Mid":
        data["Mid"] = (data["Open"] + data["Close"]) / 2
        price_column = "Mid"

    # price_column data is scaled to the specified range using the MinMaxScaler
    # initializes the scaler with a specified range input in the scaler_range parameter
    scaler = MinMaxScaler(feature_range=scaler_range)

    # scaled_data is an array that contains the scaled stock prices over time
    # Reshapes the data to the new specified range
    scaled_data = scaler.fit_transform(data[price_column].values.reshape(-1, 1))

    x_data = []
    y_data = []

    # Creates sequences of prediction_days length from the scaled data
    # Loop begins at prediction_days and goes up to the length of scaled_data
    # As prediction days = 60, the loop starts at index 60 and continues to the end of the dataset
    # Prepares data for the training the LSTM (Long Short-Term memory) model
    # x_data represents the actual stock prices from the previous prediction_days (0-59)
    # y_data represents the stock price on the next day (Day 60)
    for x in range(prediction_days, len(scaled_data)):
        x_data.append(scaled_data[x - prediction_days:x])
        y_data.append(scaled_data[x])

    # Converts the lists to numpy arrays for training
    x_data, y_data = np.array(x_data), np.array(y_data)

    # Reshapes x_data to fit the LSTM input format
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

    return x_data, y_data, scaler, data[price_column]


# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

COMPANY = 'CBA.AX' # Stock ticker symbol
TRAIN_START = '2020-01-01' # Start date for training data
TRAIN_END = '2023-08-01' # End date for training data
TEST_START = '2023-08-02' # Start date for testing data
TEST_END = '2024-07-02' # End date for testing data
PRICE_COLUMN = "Close"  # Column for price prediction
PREDICTION_DAYS = 60  # The number of past days utilized to try and predict the future price
SCALER_RANGE = (0, 1) # Scaling data range
FEATURES = ["Open", "High", "Low", "Volume", "Adj Close"]  # Features


# ------------------------------------------------------------------------------
# Load and Process Data
# ------------------------------------------------------------------------------

# Calls the load_and_process_data function to prepare the training data
# load_and_process_data is called to obtain:
# The training data (x_train and y_train)
# The scaler object (scaler)
# The raw training data (train_data)
x_train, y_train, scaler, train_data = load_and_process_data(
    COMPANY, TRAIN_START, TRAIN_END, PRICE_COLUMN, PREDICTION_DAYS, scaler_range=SCALER_RANGE, features=FEATURES
)


#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer.
# That's why we specify the input shape for this layer;
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True
# when stacking LSTM layers so that the next LSTM layer has a
# three-dimensional sequence input.

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.


#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------

# Prepares the test dataset using a similar function as preparing the training data
# x_test is sequences of data from the test period (used to make predictions)
# y_test is the values of the stock prices that correspond to the sequences in x_test
# _ represents a placeholder for the scaler
# test_data is the raw test data

x_test, y_test, _, test_data = load_and_process_data(
    COMPANY, TEST_START, TEST_END, PRICE_COLUMN, PREDICTION_DAYS, scaler_range=SCALER_RANGE, features=FEATURES
)

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------

# produces predictions for the sequences in x_test
# model.predict takes a look at the prior stock prices (which are scaled) and outputs the predicted future prices
# inverse_transform converts the normalized predictions back to the original values, that being the actual stock prices
# test_data includes all the features but using .values extracts only the raw price values
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

actual_prices = test_data.values

#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------

# real_data = [x_test[-1]] selects the last sequence of x_test --> to predict the next day’s price
# real_data = np.array(real_data) converts the list to NumPy array, the format expected by the model
# prediction = model.predict(real_data) uses the last sequence to predict the next day’s price
# prediction = scaler.inverse_transform(prediction) Converts the predicted value from the scaled range to original scale
real_data = [x_test[-1]]  # Use the last available data
real_data = np.array(real_data)
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import keras
import os
from keras import layers

from collections import deque
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
from collections import deque


# ------------------------------------------------------------------------------
# Load Data Function
# ------------------------------------------------------------------------------

def load_data(company, start_date, end_date, data_dir='data', split_ratio=0.8, split_by_date=True, random_state=None,
              scale_features=False, feature_columns=None):
    """
        Loads stock data for a company between specified dates.
    """

    # Sets default feature columns if none are provided
    if feature_columns is None:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

    # Checks the data directory exists
    os.makedirs(data_dir, exist_ok=True)
    # Constructs the data file path
    data_file = os.path.join(data_dir, f"{company}_{start_date}_{end_date}.csv")

    # Checks existence of data file; if so, its downloaded, if not the data is downloaded
    if os.path.exists(data_file):
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    else:
        # Downloads the data using yfinance
        data = yf.download(company, start=start_date, end=end_date)
        # Saved to a CSV
        data.to_csv(data_file)

    # Drops rows with missing values
    data.dropna(inplace=True)

    if split_by_date:
        # Splits data sequentially by date
        train_size = int(len(data) * split_ratio)
        train_data = data.iloc[:train_size].copy()
        test_data = data.iloc[train_size:].copy()
    else:
        # Randomly splits the data into training and testing sets
        train_data, test_data = train_test_split(
            data, train_size=split_ratio, random_state=random_state, shuffle=True)

    scalers = {}

    if scale_features:
        for column in feature_columns:
            if column in data.columns:
                # Fit scaler on training data
                scaler = MinMaxScaler(feature_range=(0, 1))
                # Transforms test data using the same scaler
                train_data[[column]] = scaler.fit_transform(train_data[[column]])
                test_data[[column]] = scaler.transform(test_data[[column]])
                # Storing scaler for potential future inverse transforms
                scalers[column] = scaler
            else:
                # Warning if specified column doesn't exist in data
                print(f"Warning: Column '{column}' not found in data. Skipping scaling for this column.")

    # Returns the split data and the scalers
    return train_data, test_data, scalers


# ------------------------------------------------------------------------------
# Prepare Data Function
# ------------------------------------------------------------------------------

def prepare_data(data, feature_columns, prediction_days):
    """
    Prepares the data for LSTM training.
    """

    x_data = []
    y_data = []

    # Creating sequences for the LSTM model
    # Adds the previous 'prediction_days' days of feature data to x_data
    for i in range(prediction_days, len(data)):
        # Adds the current day's closing price to y_data
        x_data.append(data[feature_columns].iloc[i - prediction_days:i].values)
        y_data.append(data['Close'].iloc[i])

    # Converts the lists to numpy arrays for training
    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data


# ------------------------------------------------------------------------------
# Build Model Function
# ------------------------------------------------------------------------------

def build_lstm_model(input_shape):
    """
    Builds and returns a Sequential LSTM model for stock price prediction.
    """

    # Initializes the Sequential model
    model = Sequential()

    # First LSTM layer with Dropout regularization
    # 'units' specifies the number of neurons in the layer
    # 'return_sequences' is True because we are stacking LSTM layers
    # 'input_shape' defines the shape of input data for this layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    # Dropout layer to reduce overfitting by randomly setting 20% of inputs to zero
    model.add(Dropout(0.2))
    # Second LSTM layer w/ Dropout
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Third LSTM layer w/ Dropout
    # 'return_sequences' = False since it is the last LSTM layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # Outputs a single value (the predicted stock price)
    model.add(Dense(units=1))
    # Compiled with optimizer and loss function
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# ------------------------------------------------------------------------------
# Plotting Functions
# ------------------------------------------------------------------------------

def plot_candlestick(data, n_days=30, title="Candlestick Chart"):
    """
    Generates a candlestick chart over a specified number of days.
    """

    # Creates copy of data to avoid needing to modify original DataFrame
    data = data.copy()
    # Ensures data is sorted by date
    data.sort_index(inplace=True)
    data.index = pd.to_datetime(data.index)

    if n_days > 1:
        # Resets index to use integer indexing for grouping the data into chunks of n_days
        data.reset_index(inplace=True)
        # Creates a new column called 'group', assigns rows based on integer division of the index by n_days
        data['group'] = data.index // n_days

        # Aggregate data for each group to create candlesticks
        data_resampled = data.groupby('group').agg({
            'Date': 'first',
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        # Sets the 'Date' column as the index of the new DataFrame
        }).set_index('Date')
    else:
        # Use the original data if n_days is 1
        data_resampled = data.set_index('Date')

    # Ensures that 'Date' is datetime
    data_resampled.index = pd.to_datetime(data_resampled.index)

    # Plots the candlestick chart
    mpf.plot(
        data_resampled,
        type='candle',
        title=title,
        style='charles',
        volume=False,
        figsize=(12, 8)
    )


def plot_boxplot(data, n_days=30, title="Boxplot of Stock Prices"):
    """
    Generates a boxplot chart for stock prices over a moving window of n_days.
    """

    # Calculates window for n days and stores them in a list
    windows = [data['Close'].iloc[i:i + n_days].dropna().values for i in range(0, len(data), n_days)]

    # Includes only the windows with the correct size (e.g. n_days)
    windows = [window for window in windows if len(window) == n_days]

    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(windows, patch_artist=True)
    plt.title(title)
    plt.xlabel(f'Windows of {n_days} Days')
    plt.ylabel('Stock Price (Close)')
    plt.show()

# ------------------------------------------------------------------------------
# Main Program to Use the Functions + Parameters
# ------------------------------------------------------------------------------
# Parameter values for model


if __name__ == "__main__":
    company = 'CBA.AX'
    start_date = '2020-01-01'
    end_date = '2023-08-01'
    PRICE_COLUMN = "Close"
    PREDICTION_DAYS = 60

    # Specified feature columns used as inputs
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

    # Loads data, using load_data function
    train_data, test_data, scalers = load_data(company, start_date, end_date, scale_features=True,
                                               feature_columns=feature_columns)

    # Prepares data for training, creating sequences for the LSTM model, using prepare_data function
    x_train, y_train = prepare_data(train_data, feature_columns, PREDICTION_DAYS)

    # Update the input shape for model
    # (prediction_days, num_features)
    input_shape = (x_train.shape[1], x_train.shape[2])

    # Builds LSTM model
    model = build_lstm_model(input_shape)

    # Trained the model with training data x_train, y_train
    # epochs = number of times the entire training dataset is passed through the model
    # batch_size = number of samples per gradient update
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Prepares test data
    x_test, y_test = prepare_data(test_data, feature_columns, PREDICTION_DAYS)

    # Predicts stock prices
    predicted_prices = model.predict(x_test)

    # Inverse transforms the predicted prices using the 'Close' scaler
    predicted_prices = scalers['Close'].inverse_transform(predicted_prices)

    # Inverse transforms the actual prices
    actual_prices = y_test.reshape(-1, 1)
    actual_prices = scalers['Close'].inverse_transform(actual_prices)

    # Flattens the arrays for plotting
    predicted_prices = predicted_prices.flatten()
    actual_prices = actual_prices.flatten()

    # Plots the actual prices vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
    plt.title(f"{company} Share Price Prediction")
    plt.xlabel("Time")
    plt.ylabel(f"{company} Share Price")
    plt.legend()
    plt.show()

    # Calculates Root Mean Squared Error (RMSE) to evaluate the model performance
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse}")

    # ------------------------------------------------------------------------------
    # Additional New Plots
    # ------------------------------------------------------------------------------

    # Generates Candlestick Chart
    plot_candlestick(train_data, n_days=30, title=f"{company} Candlestick Chart (30-day Candlesticks)")

    # Generates Boxplot Chart
    plot_boxplot(train_data, n_days=30, title=f"{company} Boxplot of Stock Prices (30-day Windows)")


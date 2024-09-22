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
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, Bidirectional, GRU, SimpleRNN
from sklearn.metrics import mean_squared_error
from math import sqrt

# ------------------------------------------------------------------------------
# Load Data Function
# ------------------------------------------------------------------------------


def load_data(company, start_date, end_date, data_dir='data', split_ratio=0.8, split_by_date=True, random_state=None,
              scale_features=False, feature_columns=None):

    if feature_columns is None:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, f"{company}_{start_date}_{end_date}.csv")

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    else:
        data = yf.download(company, start=start_date, end=end_date)
        data.to_csv(data_file)

    data.dropna(inplace=True)

    if split_by_date:
        train_size = int(len(data) * split_ratio)
        train_data = data.iloc[:train_size].copy()
        test_data = data.iloc[train_size:].copy()
    else:
        train_data, test_data = train_test_split(
            data, train_size=split_ratio, random_state=random_state, shuffle=True)

    scalers = {}

    if scale_features:
        for column in feature_columns:
            if column in data.columns:
                scaler = MinMaxScaler(feature_range=(0, 1))
                # Fit scaler on training data
                train_data[[column]] = scaler.fit_transform(train_data[[column]])
                # Transform test data using the same scaler
                test_data[[column]] = scaler.transform(test_data[[column]])
                scalers[column] = scaler
            else:
                print(f"Warning: Column '{column}' not found in data. Skipping scaling for this column.")

    return train_data, test_data, scalers

# ------------------------------------------------------------------------------
# Prepare Data Function
# ------------------------------------------------------------------------------

def prepare_data(data, feature_columns, prediction_days):
    """
    Prepare the data for model training/testing.

    Args:
    - data: Stock market data (DataFrame)
    - feature_columns: List of column names to be used as features.
    - prediction_days: Number of previous days to base predictions on.

    Returns:
    - x_data, y_data: Features and labels for training/testing.
    """
    x_data = []
    y_data = []

    for i in range(prediction_days, len(data)):
        x_data.append(data[feature_columns].iloc[i - prediction_days:i].values)
        y_data.append(data['Close'].iloc[i])

    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data

# ------------------------------------------------------------------------------
# Flexible Model Builder Function
# ------------------------------------------------------------------------------

def build_model(sequence_length, n_features, cell=LSTM, units=256, n_layers=2, dropout=0.3,
                optimizer='rmsprop', loss='mean_absolute_error', bidirectional=False):
    """
    Build and return a Sequential model for stock price prediction based on specified parameters.

    Args:
    - sequence_length: Length of the input sequences.
    - n_features: Number of features in the input data.
    - cell: Recurrent cell class (e.g., LSTM, GRU, SimpleRNN).
    - units: Number of units in each layer.
    - n_layers: Number of recurrent layers.
    - dropout: Dropout rate after each layer.
    - optimizer: Optimizer to use during compilation.
    - loss: Loss function to use during compilation.
    - bidirectional: Whether to use bidirectional layers.

    Returns:
    - model: The compiled model.
    """
    input_shape = (sequence_length, n_features)

    # Initialize the model
    model = Sequential()

    for i in range(n_layers):
        is_last_layer = (i == n_layers - 1)
        return_seq = not is_last_layer

        if i == 0:
            # First layer with input_shape
            layer_input_shape = input_shape
        else:
            layer_input_shape = None

        if bidirectional:
            model.add(Bidirectional(cell(units, return_sequences=return_seq), input_shape=layer_input_shape))
        else:
            model.add(cell(units, return_sequences=return_seq, input_shape=layer_input_shape))
        model.add(Dropout(dropout))

    # Add the output layer
    model.add(Dense(units=1, activation='linear'))

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error'])

    return model

# ------------------------------------------------------------------------------
# Plotting Functions
# ------------------------------------------------------------------------------

def plot_candlestick(data, n_days=30, title="Candlestick Chart"):
    data = data.copy()
    data.sort_index(inplace=True)
    data.index = pd.to_datetime(data.index)

    if n_days > 1:
        # Reset index to use integer indexing for grouping
        data.reset_index(inplace=True)
        data['group'] = data.index // n_days

        # Aggregate data for each group
        data_resampled = data.groupby('group').agg({
            'Date': 'first',
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).set_index('Date')
    else:
        data_resampled = data.set_index('Date')

    # Ensure that 'Date' is datetime
    data_resampled.index = pd.to_datetime(data_resampled.index)

    # Plot the candlestick chart
    mpf.plot(
        data_resampled,
        type='candle',
        title=title,
        style='charles',  # You can choose any style you like
        volume=False,
        figsize=(12, 8)
    )

def plot_boxplot(data, n_days=30, title="Boxplot of Stock Prices"):
    """
    Generates a boxplot for stock prices over a moving window of n_days.

    Args:
    - data (pd.DataFrame): The stock market data with a 'Close' column.
    - n_days (int): Number of consecutive trading days for the moving window.
    - title (str): Title of the boxplot chart.

    Returns:
    - None: Displays the boxplot chart.
    """

    # Calculate the rolling window for n days and store them in a list
    rolling_windows = [data['Close'].iloc[i:i + n_days].dropna().values for i in range(0, len(data), n_days)]

    # Only include windows that have the correct size (i.e., n_days)
    rolling_windows = [window for window in rolling_windows if len(window) == n_days]

    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(rolling_windows, patch_artist=True)
    plt.title(title)
    plt.xlabel(f'Windows of {n_days} Days')
    plt.ylabel('Stock Price (Close)')
    plt.show()

# ------------------------------------------------------------------------------
# Main Program to Use the Functions + Parameters
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    company = 'CBA.AX'
    start_date = '2020-01-01'
    end_date = '2023-08-01'
    PRICE_COLUMN = "Close"
    PREDICTION_DAYS = 60

    # Specify feature columns
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

    # Load the data
    train_data, test_data, scalers = load_data(
        company, start_date, end_date, scale_features=True, feature_columns=feature_columns)

    # Prepare the data for training
    x_train, y_train = prepare_data(train_data, feature_columns, PREDICTION_DAYS)

    # Update the input shape
    sequence_length = x_train.shape[1]
    n_features = x_train.shape[2]

    # Experiment configurations
    experiments = [
        {
            'cell': LSTM,
            'units': 50,
            'n_layers': 3,
            'dropout': 0.2,
            'bidirectional': False,
            'epochs': 25,
            'batch_size': 32,
            'optimizer': 'adam',
            'loss': 'mean_squared_error'
        },
        {
            'cell': GRU,
            'units': 64,
            'n_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'epochs': 30,
            'batch_size': 16,
            'optimizer': 'adam',
            'loss': 'mean_absolute_error'
        },
        {
            'cell': SimpleRNN,
            'units': 128,
            'n_layers': 4,
            'dropout': 0.1,
            'bidirectional': False,
            'epochs': 20,
            'batch_size': 64,
            'optimizer': 'rmsprop',
            'loss': 'mean_squared_error'
        }
    ]

    # Loop over experiments
    for idx, exp in enumerate(experiments):
        print(f"\n--- Experiment {idx+1}: {exp['cell'].__name__} Model ---")

        # Build the model
        model = build_model(
            sequence_length=sequence_length,
            n_features=n_features,
            cell=exp['cell'],
            units=exp['units'],
            n_layers=exp['n_layers'],
            dropout=exp['dropout'],
            optimizer=exp['optimizer'],
            loss=exp['loss'],
            bidirectional=exp['bidirectional']
        )

        # Train the model
        history = model.fit(
            x_train, y_train,
            epochs=exp['epochs'],
            batch_size=exp['batch_size'],
            validation_split=0.1,
            verbose=1
        )

        # Prepare the test data using the same function
        x_test, y_test = prepare_data(test_data, feature_columns, PREDICTION_DAYS)

        # Predict the stock prices
        predicted_prices = model.predict(x_test)

        # Inverse transform the predicted prices using the 'Close' scaler
        predicted_prices = scalers['Close'].inverse_transform(predicted_prices)

        # Inverse transform the actual prices
        actual_prices = y_test.reshape(-1, 1)
        actual_prices = scalers['Close'].inverse_transform(actual_prices)

        # Flatten the arrays for plotting
        predicted_prices = predicted_prices.flatten()
        actual_prices = actual_prices.flatten()

        # Calculate RMSE
        mse = mean_squared_error(actual_prices, predicted_prices)
        rmse = sqrt(mse)
        print(f"Test RMSE: {rmse}")

        # Plot the predictions
        plt.figure(figsize=(12, 6))
        plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
        plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
        plt.title(f"{company} Share Price Prediction ({exp['cell'].__name__} Model)")
        plt.xlabel("Time")
        plt.ylabel(f"{company} Share Price")
        plt.legend()
        plt.show()

    # ------------------------------------------------------------------------------
    # Additional New Plots
    # ------------------------------------------------------------------------------

    # Candlestick Chart (e.g., each candlestick represents 30 trading days)
    plot_candlestick(train_data, n_days=30, title=f"{company} Candlestick Chart (30-day Candlesticks)")

    # Boxplot Chart (e.g., for a moving window of 30 trading days)
    plot_boxplot(train_data, n_days=30, title=f"{company} Boxplot of Stock Prices (30-day Windows)")
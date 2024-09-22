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

    # Checks existence of data file; if so, it's loaded; if not, the data is downloaded
    if os.path.exists(data_file):
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    else:
        # Downloads the data using yfinance
        data = yf.download(company, start=start_date, end=end_date)
        # Saves to a CSV
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
                train_data[[column]] = scaler.fit_transform(train_data[[column]])
                # Transforms test data using the same scaler
                test_data[[column]] = scaler.transform(test_data[[column]])
                # Stores scaler for potential future inverse transforms
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
        x_data.append(data[feature_columns].iloc[i - prediction_days:i].values)
        # Adds the current day's closing price to y_data
        y_data.append(data['Close'].iloc[i])

    # Converts the lists to numpy arrays for training
    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data

# ------------------------------------------------------------------------------
# Build Model Function
# ------------------------------------------------------------------------------


def build_model(sequence_length, n_features, cell=LSTM, units=256, n_layers=2, dropout=0.3,
                optimizer='rmsprop', loss='mean_absolute_error', bidirectional=False):
    """
    Builds a sequential model for stock price predictions based on parameters.
    """

    # Defines the shape of the input data, which will depend on the length and number of features
    input_shape = (sequence_length, n_features)

    # Initializes the Sequential model (a linear stack of layers)
    model = Sequential()

    # For loop that adds specified amount of layers (n_layers) to the model
    for i in range(n_layers):
        # If the current layer is the last layer
        is_last_layer = (i == n_layers - 1)
        # 'return_sequences' is equal to True except for the last layer
        # This allows each of the time outputs to be fed to the next layer
        return_seq = not is_last_layer

        # Specify the input shape for the first layer
        if i == 0:
            layer_input_shape = input_shape
        else:
            layer_input_shape = None    # Keras can sort it out

        if bidirectional:
            # Adds a Bidirectional layer if needed
            # Allows cell to process input sequences in both directions
            model.add(Bidirectional(cell(units, return_sequences=return_seq), input_shape=layer_input_shape))
        else:
            # Adds the specified cell (LSTM, GRU, SimpleRNN, etc.) to model
            model.add(cell(units, return_sequences=return_seq, input_shape=layer_input_shape))

        # A dropout layer after each recurrent layer to reduce overfitting
        # Randomly setting some input units to 0 at each update during training
        model.add(Dropout(dropout))

    # Dense layer that produces the final output
    # Linear activation is used since a continous value is being predicted
    model.add(Dense(units=1, activation='linear'))

    # Compiles the model with the specified optimizer and loss function
    # Able to monitor the model's performance with metrics such as 'mean_absolute_error'
    model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error'])

    # Returns model, ready for training
    return model

# ------------------------------------------------------------------------------
# Plotting Functions
# ------------------------------------------------------------------------------


def plot_candlestick(data, n_days=30, title="Candlestick Chart"):
    """
    Generates a candlestick chart over a specified number of days.
    """

    # Creates copy of data to avoid modifying the original DataFrame
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
    Generates a boxplot for stock prices over a moving window of n_days.
    """

    # Calculates the rolling window for n days and stores them in a list
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
# Main Program + Parameters
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    company = 'CBA.AX'
    start_date = '2020-01-01'
    end_date = '2023-08-01'
    PRICE_COLUMN = "Close"
    PREDICTION_DAYS = 60

    # Specified feature columns used as inputs
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

    # Loads data, using load_data function
    train_data, test_data, scalers = load_data(
        company, start_date, end_date, scale_features=True, feature_columns=feature_columns)

    # Prepares data for training, creating sequences for the model's input
    x_train, y_train = prepare_data(train_data, feature_columns, PREDICTION_DAYS)

    # Update the input shape for model
    sequence_length = x_train.shape[1]  # Time steps in each sequence
    n_features = x_train.shape[2]       # Features in each time step

    # Configurations for each experiment to compare the model performances
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
            'dropout': 0.2,
            'bidirectional': False,
            'epochs': 40,
            'batch_size': 64,
            'optimizer': 'adam',
            'loss': 'mean_squared_error'
        }
    ]

    # Loops over experiments to train / test the model configurations
    for idx, exp in enumerate(experiments):
        print(f"\n--- Experiment {idx+1}: {exp['cell'].__name__} Model ---")

        # Builds the model with the specified parameters in the experiment above
        model = build_model(
            sequence_length=sequence_length,      # sequence length
            n_features=n_features,                # no. features per time step
            cell=exp['cell'],                     # RNN cell
            units=exp['units'],                   # no. neurons per layer
            n_layers=exp['n_layers'],             # no. recurrent layers
            dropout=exp['dropout'],               # dropout rate
            optimizer=exp['optimizer'],           # optimizer
            loss=exp['loss'],                     # loss functions
            bidirectional=exp['bidirectional']    # bidirectional layers Y/N?
        )

        # Trains the model with training data x_train, y_train
        model.fit(
            x_train, y_train,                     # training features, training labels
            epochs=exp['epochs'],                 # no. epochs
            batch_size=exp['batch_size'],         # no. samples per gradient
            validation_split=0.1,                 # training data used for validation
            verbose=1                             # verbosity mode (progress bar)
        )

        # Prepares test data using same function as training data
        x_test, y_test = prepare_data(test_data, feature_columns, PREDICTION_DAYS)

        # Trained model predicts stock prices on test data
        predicted_prices = model.predict(x_test)

        # Inverse transform predicted prices using the 'Close' scaler
        predicted_prices = scalers['Close'].inverse_transform(predicted_prices)

        # Reshape and inverse transform the actual prices to og scale
        actual_prices = y_test.reshape(-1, 1)
        actual_prices = scalers['Close'].inverse_transform(actual_prices)

        # Flattens the arrays to one dimension for plotting
        predicted_prices = predicted_prices.flatten()
        actual_prices = actual_prices.flatten()

        """
        # Calculates Root Mean Squared Error (RMSE) to evaluate the model performance
        mse = mean_squared_error(actual_prices, predicted_prices)
        rmse = sqrt(mse)
        print(f"Test RMSE: {rmse}")
        """

        # Plots the actual prices vs predicted prices
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

    # Generates Candlestick Chart
    plot_candlestick(train_data, n_days=30, title=f"{company} Candlestick Chart (30-day Candlesticks)")

    # Generates Boxplot Chart
    plot_boxplot(train_data, n_days=30, title=f"{company} Boxplot of Stock Prices (30-day Windows)")
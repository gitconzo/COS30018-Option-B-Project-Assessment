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
    - Download data from Yahoo Finance or load from a CSV file if already downloaded.
    - Scales the features if needed.
    - Splits data into training and testing sets either by date or randomly.
    """

    # Sets default feature columns if none are provided
    if feature_columns is None:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']

    # Ensures the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Creates the path to the CSV file where the stock data will be stored
    data_file = os.path.join(data_dir, f"{company}_{start_date}_{end_date}.csv")

    # Check if the data is already stored locally
    if os.path.exists(data_file):
        # Load the data from the local CSV file
        data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    else:
        # Otherwise, download the data from Yahoo Finance
        data = yf.download(company, start=start_date, end=end_date)
        # Save the data to a CSV file for future use
        data.to_csv(data_file)

    # Drop any rows with missing values (NaNs) in the dataset
    data.dropna(inplace=True)

    # Splitting the data based on whether the user wants to split by date or randomly
    if split_by_date:
        # Splits data sequentially by date
        train_size = int(len(data) * split_ratio)
        train_data = data.iloc[:train_size].copy()  # Training data until the split ratio
        test_data = data.iloc[train_size:].copy()  # Test data after the split
    else:
        # Randomly splits the data into training and testing sets
        train_data, test_data = train_test_split(data, train_size=split_ratio, random_state=random_state, shuffle=True)

    # Dictionary to store scalers for each feature
    scalers = {}

    # If scaling is enabled, scale the feature columns between 0 and 1
    if scale_features:
        for column in feature_columns:
            if column in data.columns:
                # Initialize a MinMaxScaler for the current column
                scaler = MinMaxScaler(feature_range=(0, 1))
                # Fit the scaler to the training data and transform it
                train_data[[column]] = scaler.fit_transform(train_data[[column]])
                # Use the same scaler to transform the test data
                test_data[[column]] = scaler.transform(test_data[[column]])
                # Store the scaler for future inverse transformations
                scalers[column] = scaler
            else:
                # If the column is not found, warn the user
                print(f"Warning: Column '{column}' not found in data. Skipping scaling for this column.")

    # Returns the training data, testing data, and the scalers
    return train_data, test_data, scalers


# ------------------------------------------------------------------------------
# Prepare Data Function
# ------------------------------------------------------------------------------

def prepare_data(data, feature_columns, prediction_days, steps_ahead=1, multistep=False):
    """
    Prepares data for training an LSTM model by creating sequences.
    - If multistep is True, creates data for multistep prediction.
    """

    # Initializes lists to store input sequences (x_data) and target values (y_data)
    x_data = []
    y_data = []

    if multistep:
        # Creates sequences of historical data for the input and multiple future steps for the output
        for i in range(prediction_days, len(data) - steps_ahead + 1):
            # Takes 'prediction_days' worth of data as input
            x_data.append(data[feature_columns].iloc[i - prediction_days:i].values)
            # Predicts multiple future 'Close' values (steps ahead)
            y_data.append(data['Close'].iloc[i:i + steps_ahead].values)
    else:
        # For just one day ahead (single step)
        for i in range(prediction_days, len(data)):
            # Takes 'prediction_days' worth of data as input
            x_data.append(data[feature_columns].iloc[i - prediction_days:i].values)
            # Predicts the next day's 'Close' value
            y_data.append(data['Close'].iloc[i])

    # Converting to numpy arrays for training
    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data


# ------------------------------------------------------------------------------
# Build Model Function
# ------------------------------------------------------------------------------

def build_model(sequence_length, n_features, steps_ahead=1, cell=LSTM, units=256, n_layers=2, dropout=0.3,
                optimizer='rmsprop', loss='mean_absolute_error', bidirectional=False):
    """
    Builds a Sequential neural network model for stock price predictions.
    """

    # Defines the input shape for the model based on sequence length and features
    input_shape = (sequence_length, n_features)

    # Initializes the Sequential model
    model = Sequential()

    # Loop to add the specified number of layers
    for i in range(n_layers):
        # Check if the current layer is the last one in the model
        is_last_layer = (i == n_layers - 1)
        # If it's the last layer, return_sequences is False (don't output full sequence)
        return_seq = not is_last_layer

        # Specify input shape for the first layer
        if i == 0:
            layer_input_shape = input_shape
        else:
            layer_input_shape = None  # Keras infers the input shape automatically for subsequent layers

        if bidirectional:
            # Add Bidirectional LSTM/GRU layer if bidirectional is True
            model.add(Bidirectional(cell(units, return_sequences=return_seq), input_shape=layer_input_shape))
        else:
            # Add regular LSTM/GRU layer
            model.add(cell(units, return_sequences=return_seq, input_shape=layer_input_shape))

        # Add dropout layer after each recurrent layer to prevent overfitting
        model.add(Dropout(dropout))

    # Add final dense layer to output prediction
    # For multistep predictions, output size equals steps_ahead
    model.add(Dense(units=steps_ahead, activation='linear'))

    # Compile the model using the specified optimizer and loss function
    model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error'])

    # Return the compiled model ready for training
    return model


# ------------------------------------------------------------------------------
# Plotting Functions
# ------------------------------------------------------------------------------

def plot_candlestick(data, n_days=30, title="Candlestick Chart"):
    """
    Generates a candlestick chart for stock prices.
    """

    # Create a copy of the data to avoid modifying the original DataFrame
    data = data.copy()
    # Sort data by date to ensure chronological order
    data.sort_index(inplace=True)
    data.index = pd.to_datetime(data.index)

    # Group the data by the specified number of days (n_days)
    if n_days > 1:
        # Reset index to use integer indexing for grouping
        data.reset_index(inplace=True)
        # Create a 'group' column for grouping rows by n_days
        data['group'] = data.index // n_days

        # Aggregate data for each group to create candlesticks
        data_resampled = data.groupby('group').agg({
            'Date': 'first',
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
            # Set the 'Date' column as the index
        }).set_index('Date')
    else:
        # If n_days is 1, use the original data
        data_resampled = data.set_index('Date')

    # Ensure 'Date' is in datetime format
    data_resampled.index = pd.to_datetime(data_resampled.index)

    # Plot the candlestick chart using mplfinance
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

    # Create rolling windows of 'Close' prices for n_days
    rolling_windows = [data['Close'].iloc[i:i + n_days].dropna().values for i in range(0, len(data), n_days)]

    # Only include windows that have the correct size (i.e., exactly n_days)
    rolling_windows = [window for window in rolling_windows if len(window) == n_days]

    # Plot the boxplot using matplotlib
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
    STEPS_AHEAD = 30  # Number of future steps to predict for multistep forecasting

    # Feature columns used in the analysis
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Load stock data, scaling features and splitting into train/test sets
    train_data, test_data, scalers = load_data(
        company, start_date, end_date, scale_features=True, feature_columns=feature_columns)

    # Configurations for each model experiment (single-step vs multistep, univariate vs multivariate)
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
            'loss': 'mean_squared_error',
            'multistep': False,  # Single-step prediction
            'steps_ahead': 1,  # Predicting 1 day ahead
            'multivariate': False,  # Univariate (using only 'Close')
            'task': 'Single-Step Univariate'
        },
        {
            'cell': LSTM,
            'units': 128,
            'n_layers': 4,
            'dropout': 0.2,
            'bidirectional': False,
            'epochs': 40,
            'batch_size': 64,
            'optimizer': 'adam',
            'loss': 'mean_squared_error',
            'multistep': False,
            'steps_ahead': 1,
            'multivariate': True,
            'task': 'Single-Step Multivariate'
        },
        {
            'cell': LSTM,
            'units': 128,
            'n_layers': 4,
            'dropout': 0.2,
            'bidirectional': False,
            'epochs': 40,
            'batch_size': 64,
            'optimizer': 'adam',
            'loss': 'mean_squared_error',
            'multistep': True,
            'steps_ahead': STEPS_AHEAD,
            'multivariate': False,
            'task': 'Multistep'
        },
        {
            'cell': LSTM,
            'units': 128,
            'n_layers': 3,
            'dropout': 0.2,
            'bidirectional': False,
            'epochs': 50,
            'batch_size': 32,
            'optimizer': 'adam',
            'loss': 'mean_absolute_error',
            'multistep': True,  # Multistep prediction
            'steps_ahead': STEPS_AHEAD,  # Predicting 30 days ahead
            'multivariate': True,
            'task': 'Multistep Multivariate'
        }
    ]

    # Loop through each experiment to train and evaluate models
    for idx, exp in enumerate(experiments):
        # Printing out info on the model and task
        print(f"\n--- Experiment {idx + 1}: {exp['cell'].__name__} Model ({exp['task']}) ---")

        # If statement for either multivariate (multiple features) or just using 'Close'
        if exp['multivariate']:
            input_features = feature_columns  # If multivariate, use all the features defined
        else:
            input_features = ['Close']  # Else, 'Close' for univariate

        # Prepare the training data depending on whether it's predicting one day ahead or multiple days
        x_train, y_train = prepare_data(train_data, input_features, PREDICTION_DAYS,
                                        steps_ahead=exp['steps_ahead'], multistep=exp['multistep'])

        # Same with the test data
        x_test, y_test = prepare_data(test_data, input_features, PREDICTION_DAYS,
                                      steps_ahead=exp['steps_ahead'], multistep=exp['multistep'])

        # Sequence length (number of days) and the number of features per day.
        sequence_length = x_train.shape[1]
        n_features = x_train.shape[2]

        # Build the model based on the parameters.
        model = build_model(
            sequence_length=sequence_length,
            n_features=n_features,
            steps_ahead=exp['steps_ahead'],
            cell=exp['cell'],
            units=exp['units'],
            n_layers=exp['n_layers'],
            dropout=exp['dropout'],
            optimizer=exp['optimizer'],
            loss=exp['loss'],
            bidirectional=exp['bidirectional']
        )

        # Train the model using the training data. Using 10% of the data for validation
        history = model.fit(
            x_train, y_train,
            epochs=exp['epochs'],
            batch_size=exp['batch_size'],
            validation_split=0.1,
            verbose=1  # Set to 1 so we can see the training progress
        )

        # Make predictions on the test data using the trained model
        predicted_prices = model.predict(x_test)

        # If multistep predictions (multiple days at once), data needs to be reshaped
        if exp['multistep']:
            # Reshape the predicted prices and convert them back to their original scale.
            predicted_prices_reshaped = predicted_prices.reshape(-1, 1)
            predicted_prices_inv = scalers['Close'].inverse_transform(predicted_prices_reshaped)
            predicted_prices_inv = predicted_prices_inv.reshape(predicted_prices.shape)

            # Same for the real prices from the test set --> convert them back to the original scale.
            actual_prices_reshaped = y_test.reshape(-1, 1)
            actual_prices_inv = scalers['Close'].inverse_transform(actual_prices_reshaped)
            actual_prices_inv = actual_prices_inv.reshape(y_test.shape)

            # plot the first sample from the test set to compare predicted vs actual prices
            sample_idx = 0
            plt.figure(figsize=(10, 5))
            plt.plot(actual_prices_inv[sample_idx], color="black", label=f"Actual {company} Price", linewidth=1)
            plt.plot(predicted_prices_inv[sample_idx], color="green", label=f"Predicted {company} Price", linewidth=1)
            plt.title(f"{company} {exp['task']} ({exp['cell'].__name__} Model)")
            plt.xlabel("Days Ahead")
            plt.ylabel(f"{company} Share Price")
            plt.legend()
            plt.grid(True)

            # Adjusting the x-axis and y-axis for better visibility.
            num_points = len(actual_prices_inv[sample_idx])
            plt.xlim(0, num_points + 5)
            min_price = min(np.min(actual_prices_inv[sample_idx]), np.min(predicted_prices_inv[sample_idx]))
            max_price = max(np.max(actual_prices_inv[sample_idx]), np.max(predicted_prices_inv[sample_idx]))
            y_margin = (max_price - min_price) * 0.1
            plt.ylim(min_price - y_margin, max_price + y_margin)

            plt.show()

        else:
            # Single-step predictions (one day at a time
            # Inverse transform the predictions and actual prices
            predicted_prices_inv = scalers['Close'].inverse_transform(predicted_prices)
            actual_prices_inv = scalers['Close'].inverse_transform(y_test.reshape(-1, 1))

            # Flatten the arrays
            predicted_prices_inv = predicted_prices_inv.flatten()
            actual_prices_inv = actual_prices_inv.flatten()

            # Plot the actual vs predicted prices for the entire test set
            plt.figure(figsize=(10, 5))
            plt.plot(actual_prices_inv, color="black", label=f"Actual {company} Price", linewidth=1)
            plt.plot(predicted_prices_inv, color="green", label=f"Predicted {company} Price", linewidth=1)
            plt.title(f"{company} {exp['task']} ({exp['cell'].__name__} Model)")
            plt.xlabel("Time")
            plt.ylabel(f"{company} Share Price")
            plt.legend()
            plt.grid(True)

            # Adjust the x-axis and y-axis
            num_points = len(actual_prices_inv)
            plt.xlim(0, num_points + 5)
            min_price = min(np.min(actual_prices_inv), np.min(predicted_prices_inv))
            max_price = max(np.max(actual_prices_inv), np.max(predicted_prices_inv))
            y_margin = (max_price - min_price) * 0.1
            plt.ylim(min_price - y_margin, max_price + y_margin)

            plt.show()

    # ------------------------------------------------------------------------------
    # Additional Plots
    # ------------------------------------------------------------------------------

    # Generate Candlestick Chart
    plot_candlestick(train_data, n_days=30, title=f"{company} Candlestick Chart (30-day Candlesticks)")

    # Generate Boxplot Chart
    plot_boxplot(train_data, n_days=30, title=f"{company} Boxplot of Stock Prices (30-day Windows)")
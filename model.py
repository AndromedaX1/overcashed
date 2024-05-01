import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# Fetch historical stock data
def fetch_data(stock_ticker, start_date, end_date):
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    return stock_data['Close']

# Prepare Data for LSTM
def prepare_data(data, look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    X, Y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        Y.append(scaled_data[i, 0])
    return np.array(X), np.array(Y), scaler

# Build LSTM Model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Parameters
stock_ticker = 'AAPL'
start_date = '2023-04-30'
end_date = '2024-04-30'
look_back = 60

# Fetch and prepare data
data = fetch_data(stock_ticker, start_date, end_date)
X, Y, scaler = prepare_data(data, look_back)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM

# Build and train the model
model = build_model((look_back, 1))
model.fit(X, Y, epochs=50, batch_size=32, verbose=1)

# Save the model
model.save('stock_model.keras')  # Saves the model in HDF5 format

import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

class StockPricePredictor:
    def __init__(self, model_path, stock_ticker, start_date, end_date, look_back):
        self.model_path = model_path
        self.stock_ticker = stock_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.look_back = look_back
        self.model = load_model(model_path)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self):
        stock_data = yf.download(self.stock_ticker, start=self.start_date, end=self.end_date)
        return stock_data['Close']

    def prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        X = []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
        return np.array(X)

    def predict_future_prices(self, days_ahead):
        data = self.fetch_data()
        inputs = self.prepare_data(data)
        last_sequence = inputs[-1]
        predictions = []
        for _ in range(days_ahead):
            prediction = self.model.predict(last_sequence.reshape(1, -1, 1))
            predictions.append(prediction.flatten()[0])
            last_sequence = np.append(last_sequence[1:], prediction)
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

if __name__ == "__main__":
    predictor = StockPricePredictor(
        model_path='stock_model.keras',
        stock_ticker='AAPL',
        start_date='2023-04-30',
        end_date='2024-04-30',
        look_back=60
    )
    days_ahead = 30  # Number of days ahead to predict
    predicted_prices = predictor.predict_future_prices(days_ahead)
    print("Predicted prices for the next {} days: {}".format(days_ahead, predicted_prices))


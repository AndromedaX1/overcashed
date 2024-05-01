from predictor import StockPricePredictor
from datetime import datetime, timedelta

def main(stock_ticker, start_date, end_date, look_back, specific_dates):
    # Calculate days ahead based on the furthest date in specific_dates
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    specific_dates_dt = [datetime.strptime(date, "%Y-%m-%d") for date in specific_dates]
    max_date_dt = max(specific_dates_dt)

    # Calculate days ahead
    days_ahead = (max_date_dt - end_date_dt).days + 1

    # Initialize the predictor object
    predictor = StockPricePredictor(
        model_path='stock_model.keras',
        stock_ticker=stock_ticker,
        start_date=start_date,
        end_date=end_date,
        look_back=look_back
    )

    # Predict future prices
    predicted_prices = predictor.predict_future_prices(days_ahead)
    date_range = [end_date_dt + timedelta(days=i) for i in range(days_ahead + 1)]

    # Create a dictionary from date_range and predicted_prices
    date_to_price_map = dict(zip([date.strftime("%Y-%m-%d") for date in date_range], predicted_prices.flatten()))

    # Print prices on specific dates
    print("Predicted prices on specific dates:")
    for date in specific_dates:
        print(f"Date: {date}, Price: {date_to_price_map.get(date, 'No prediction available')}")

    return predicted_prices

if __name__ == "__main__":
    # Changing these require changing and retraining model.py
    stock_ticker = 'AAPL'
    start_date = '2023-04-30'
    end_date = '2024-04-30'
    
    # Hyperparamters    
    look_back = 60 # maximum of the difference between start_date and end_date
    specific_dates = ['2024-05-05', '2024-05-15', '2024-05-30', '2025-05-31']  # look for specific dates you want predictions for

    predictions = main(stock_ticker, start_date, end_date, look_back, specific_dates)


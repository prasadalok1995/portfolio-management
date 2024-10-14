import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

def load_data():
    # Load NSE ticker symbols
    nse_tickers = pd.read_csv("equity.csv")
    nse_tickers['SYMBOL'] = nse_tickers['SYMBOL'] + ".NS"
    nse_tickers['symbol_name'] = nse_tickers['SYMBOL']

    # Load BSE ticker symbols
    bse_tickers = pd.read_csv("Equitybse.csv")
    bse_tickers['SYMBOL'] = bse_tickers['SYMBOL'] + ".BO"
    bse_tickers['symbol_name'] = bse_tickers['SYMBOL']

    # Combine both lists
    combined_tickers = pd.concat([nse_tickers, bse_tickers], ignore_index=True)
    #index symbol
    index_tickers = pd.DataFrame({
    'SYMBOL': ['^NSEI', '^NSEBANK', '^BSESN'],
    'symbol_name': ['^NSEI', '^NSEBANK', '^BSESN']
    })

    # Combine both lists
    combined_tickers = pd.concat([combined_tickers, index_tickers], ignore_index=True)
    return combined_tickers

def main():
    st.title("Stock Price Predictor App")

    # Load the ticker list
    ticker_list = load_data()
    selected_stock = st.selectbox("Select a Stock Ticker", ticker_list['symbol_name'])
    stock = selected_stock
    
    # Date range for stock data
    end = datetime.now()
    start = datetime(end.year-20, end.month, end.day)
    
    # Fetch stock data
    google_data = yf.download(stock, start, end)
    model = load_model("Latest_stock_price_model.keras")
    
    # Display stock data
    st.subheader("Stock Data")
    st.write(google_data.iloc[::-1])

    # Calculate moving averages and plot them
    for days in [100, 200, 250]:
        ma_col = f'MA_for_{days}_days'
        google_data[ma_col] = google_data['Close'].rolling(days).mean()
        st.subheader(f'Original Close Price and {ma_col}')
        fig = plt.figure(figsize=(15,6))
        plt.plot(google_data['Close'], 'b')
        plt.plot(google_data[ma_col], 'orange')
        plt.title(f'Close Price vs {ma_col}')
        plt.grid(True)
        st.pyplot(fig)

    # Prepare data for the model
    splitting_len = int(len(google_data) * 0.7)
    x_test = pd.DataFrame(google_data['Close'][splitting_len:])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])
    
    x_data = []
    y_data = []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Predictions and plotting
    predictions = model.predict(x_data)
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)
    ploting_data = pd.DataFrame({
        'original_test_data': inv_y_test.flatten(),
        'predictions': inv_pre.flatten()
    }, index=google_data.index[splitting_len+100:])
    
    st.subheader("Original values vs Predicted values")
    st.write(ploting_data.iloc[::-1])

    # Display future forecast
    future_dates = [end + timedelta(days=i) for i in range(1, 4)]
    forecasted_values = []
    last_data = scaled_data[-100:].reshape(1, 100, 1)
    for _ in future_dates:
        prediction = model.predict(last_data)
        forecasted_values.append(scaler.inverse_transform(prediction)[0, 0])
        last_data = np.append(last_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Price': forecasted_values})
    st.subheader("Future Forecasted Values")
    st.write(forecast_df)

    # Optional: Stock ratios display
    ticker = yf.Ticker(stock)
    info = ticker.info
    ratios = {
        'P/E Ratio': info.get('trailingPE', 'N/A'),
        'Forward P/E Ratio': info.get('forwardEps', 'N/A'),
        'PEG Ratio': info.get('pegRatio', 'N/A'),
        'Price/Sales Ratio': info.get('priceToSalesTrailing12Months', 'N/A'),
        'Price/Book Ratio': info.get('priceToBook', 'N/A'),
        'Dividend Yield': info.get('dividendYield', 'N/A'),
        '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
        '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A')
    }
    st.subheader("Stock Ratios")
    st.write(pd.DataFrame(list(ratios.items()), columns=['Metric', 'Value']))

if __name__ == "__main__":
    main()

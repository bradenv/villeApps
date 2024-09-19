import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Function to fetch data and train model
def train_model(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(period="1y")
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data = data.dropna()
    
    X = data[['Close', '50_MA']]
    y = data['Close'].shift(-1).dropna()
    X = X[:-1]  # Align X with y after shifting
    
    # Randomly split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test, data

# Function to fetch data and make predictions
def fetch_and_predict(stock_symbol, forecast_period, model, historical_data):
    X = historical_data[['Close', '50_MA']].iloc[-1].values.reshape(1, -1)
    
    # Generate forecast dates
    forecast_dates = pd.date_range(start=historical_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_period)
    
    # Predict future prices
    predicted_prices = []
    for _ in range(forecast_period):
        predicted_price = model.predict(X)[0]
        predicted_prices.append(predicted_price)
        X[0][0] = predicted_price  # Update 'Close' for the next prediction
        X[0][1] = (X[0][1] * 49 + predicted_price) / 50  # Update '50_MA'
    
    return historical_data, forecast_dates, predicted_prices

# Streamlit app
st.title("Stock Price Prediction")

symbol = st.text_input("Stock Symbol:")
forecast_period = st.number_input("Forecast Period (days):", min_value=1, value=7)

if st.button("Get Forecast"):
    if symbol:
        # Train model and fetch historical data
        model, X_train, X_test, y_train, y_test, historical_data = train_model(symbol)
        
        # Make predictions
        historical_data, forecast_dates, predictions = fetch_and_predict(symbol, forecast_period, model, historical_data)
        
        # Plot historical data and predictions
        plt.figure(figsize=(12, 8))
        
        # Plot historical data
        plt.plot(historical_data['Date'], historical_data['Close'], label='Historical Prices', color='blue')
        plt.plot(historical_data['Date'], historical_data['50_MA'], label='50-Day Moving Average', color='orange')
        
        # Plot forecasted data
        forecast_dates = pd.Series(forecast_dates)
        plt.plot(forecast_dates, predictions, label='Forecasted Prices', color='red', linestyle='--')
        
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Stock Price Forecast for {symbol}')
        plt.legend()
        plt.grid(True)
        
        # Display the plot
        st.pyplot(plt)
        
        # Format predicted prices (round to 2 decimal places)
        predictions_formatted = [round(float(pred), 2) for pred in predictions]
        
        # Create a DataFrame for the predictions and forecast dates
        forecast_data = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted Price': predictions_formatted
        })
        
        # Calculate the average predicted price
        average_predicted_price = round(np.mean(predictions_formatted), 2)
        
        # Display the table
        st.table(forecast_data)
        
        # Display the average predicted price
        st.write(f"Average Predicted Price: {average_predicted_price}")
        
        # Accuracy metrics calculation
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        # Display accuracy metrics in a table
        st.subheader("Model Accuracy")
        st.write(f"Mean Squared Error (MSE) on Training Data: {mse_train:.2f}")
        st.write(f"Mean Squared Error (MSE) on Test Data: {mse_test:.2f}")
        st.write(f"Mean Absolute Error (MAE) on Test Data: {mae_test:.2f}")
        st.write(f"R-squared (R²) on Test Data: {r2_test:.2f}")

    else:
        st.error("Please enter a stock symbol.")

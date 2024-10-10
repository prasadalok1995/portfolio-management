import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to calculate daily returns
def calculate_returns(stock_df):
    return stock_df.pct_change().dropna()

# Function to calculate the correlation matrix
def calculate_correlation(returns_df):
    return returns_df.corr()

# Function to calculate portfolio return
def portfolio_return(weights, mean_returns):
    return np.sum(weights * mean_returns) * 252

# Function to calculate portfolio volatility
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

# Function for portfolio optimization
def optimize_portfolio(mean_returns, cov_matrix, scenario='ORP'):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # weights must sum to 1
    bounds = tuple((0, 1) for asset in range(num_assets))  # weights between 0 and 1

    if scenario == 'ORP':
        # Optimal Risk Portfolio (ORP) - Max Sharpe Ratio
        def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.069):
            p_return = portfolio_return(weights, mean_returns)
            p_volatility = portfolio_volatility(weights, cov_matrix)
            return -(p_return - risk_free_rate) / p_volatility  # negative Sharpe ratio

        result = minimize(neg_sharpe, num_assets*[1./num_assets], args=args, bounds=bounds, constraints=constraints)
    
    elif scenario == 'GMV':
        # Global Minimum Variance (GMV) - Minimize volatility
        def portfolio_variance(weights, mean_returns, cov_matrix):
            return portfolio_volatility(weights, cov_matrix) ** 2

        result = minimize(portfolio_variance, num_assets*[1./num_assets], args=args, bounds=bounds, constraints=constraints)
    
    elif scenario == 'Naive':
        # Naive Portfolio - Equal Weights
        result = {'x': np.array([1./num_assets]*num_assets)}
    
    return result['x']  # Optimized weights

# UI for user input: Tickers, number of years, number of stocks in the portfolio
st.title("Portfolio Recommendation System")
st.markdown("### Enter your portfolio details")

tickers_input = st.text_input("Enter ticker symbols separated by commas (e.g., AAPL, MSFT, TSLA):")
years = st.slider("Select the number of years for historical data:", 1, 10, 5, key="years_slider")
num_stocks_input = st.slider("Select the number of stocks for your portfolio:", 2, 10, key="num_stocks_slider")

# Split and clean the ticker input
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

if tickers_input:
    try:
        # Fetch stock data from yfinance
        stock_data = yf.download(tickers, period=f"{years}y")['Adj Close']
        
        if stock_data.empty:
            st.write("No data found for the entered tickers.")
        else:
            st.write("### Fetched Data")
            st.dataframe(stock_data)

            # Allow user to select the tickers for the portfolio
            selected_tickers = st.multiselect("Select tickers for portfolio recommendation:", tickers, default=tickers)
            valid_tickers = [ticker for ticker in selected_tickers if ticker in stock_data.columns]

            if len(valid_tickers) >= num_stocks_input:
                # Calculate returns and covariance matrix for selected tickers
                returns_df = calculate_returns(stock_data)
                selected_returns = returns_df[valid_tickers]
                cov_matrix = selected_returns.cov()

                # Calculate correlation matrix and show heatmap
                correlation_matrix = calculate_correlation(selected_returns)
                st.write("### Correlation Matrix of Selected Tickers")
                st.dataframe(correlation_matrix)

                # Plot heatmap of the correlation matrix
                st.write("### Correlation Heatmap")
                fig, ax = plt.subplots()
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

                # Optimization
                scenarios = ['ORP', 'GMV', 'Naive']
                scenario = st.selectbox("Select the optimization scenario:", scenarios)
                
                optimized_weights = optimize_portfolio(selected_returns.mean(), cov_matrix, scenario)
                portfolio_return_value = portfolio_return(optimized_weights, selected_returns.mean())
                portfolio_volatility_value = portfolio_volatility(optimized_weights, cov_matrix)

                if len(valid_tickers) >= num_stocks_input:
                    st.write(f"### Portfolio Optimization ({scenario})")
                    st.write(f"**Weights:** {', '.join(f'{ticker}: {weight*100:.2f}%' for ticker, weight in zip(valid_tickers, optimized_weights))}")
                    st.write(f"**Expected Annual Return:** {portfolio_return_value*100:.2f}%")
                    st.write(f"**Annual Volatility (Risk):** {portfolio_volatility_value*100:.2f}%")
                else:
                    st.write("Please select at least as many tickers as the number of stocks you want in your portfolio.")
    except Exception as e:
        st.write(f"An error occurred: {e}")
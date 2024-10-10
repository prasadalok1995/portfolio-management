import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Streamlit app title and description
st.title("Enhanced Portfolio Recommendation System")
st.markdown("""
    This application helps you create an optimized portfolio based on historical stock data. 
    Choose your stocks, specify a time period, and select an optimization strategy to get started!
""")

# Sidebar for user inputs
st.sidebar.header("Portfolio Inputs")
tickers_input = st.sidebar.text_input("Ticker Symbols (comma-separated)", "AAPL, MSFT, TSLA", help="Enter ticker symbols like AAPL, MSFT, TSLA")
years = st.sidebar.selectbox("Historical Data (Years)", list(range(1, 11)), index=4, help="Select the number of years of historical data")

# Split and clean ticker input
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

if tickers_input:
    try:
        end_date= datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=years * 365)
        # Fetch stock data
        stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if stock_data.empty:
            st.warning("No data found for the entered tickers.")
        else:
            # Display fetched data
            with st.expander("View Fetched Data"):
                st.dataframe(stock_data.iloc[::-1])

            # User selects tickers for portfolio
            selected_tickers = st.multiselect("Select Tickers for Portfolio", tickers, default=tickers)
            valid_tickers = [ticker for ticker in selected_tickers if ticker in stock_data.columns]

            if valid_tickers:
                # Calculate returns and covariance matrix
                returns_df = stock_data[valid_tickers].pct_change().dropna()
                cov_matrix = returns_df.cov()

                # Display correlation matrix and heatmap
                correlation_matrix = returns_df.corr()
                st.subheader("Correlation Matrix")
                st.dataframe(correlation_matrix)
                
                st.subheader("Correlation Heatmap")
                sns.set_theme(style="white")
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(correlation_matrix,
                annot=True,
                cmap="coolwarm",
                ax=ax,
                cbar=True,
                annot_kws={"size": 12, "color": "black"},  # Make annotations larger and in black
                linewidths=0.5,  # Add grid lines for better separation
                linecolor="gray")
                ax.tick_params(axis='x', colors='black', labelsize=12, labelrotation=45)
                ax.tick_params(axis='y', colors='black', labelsize=12)
                fig.patch.set_facecolor('white')
                ax.set_facecolor('#f5f5f5')
                st.pyplot(fig)

                # Portfolio Optimization
                scenario = st.selectbox("Optimization Strategy", ['Optimal Risk Portfolio', 'Global Minimum Variance', 'Equal Weights'])
                
                def get_optimized_weights(scenario):
                    num_assets = len(returns_df.columns)
                    args = (returns_df.mean(), cov_matrix)
                    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                    bounds = tuple((0, 1) for _ in range(num_assets))
                    
                    if scenario == 'Optimal Risk Portfolio':
                        def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.069):
                            p_return = np.sum(weights * mean_returns) * 252
                            p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                            return -(p_return - risk_free_rate) / p_volatility
                        result = minimize(neg_sharpe, num_assets * [1. / num_assets], args=args, bounds=bounds, constraints=constraints)
                    
                    elif scenario == 'Global Minimum Variance':
                        def portfolio_volatility(weights):
                            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                        result = minimize(portfolio_volatility, num_assets * [1. / num_assets], bounds=bounds, constraints=constraints)
                    
                    else:
                        result = {'x': np.array([1. / num_assets] * num_assets)}
                    
                    return result['x']

                # Calculate and display optimized weights
                weights = get_optimized_weights(scenario)
                portfolio_return = np.sum(weights * returns_df.mean()) * 252
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

                st.subheader(f"{scenario} Results")
                st.write("**Optimized Weights:**")
                weights_df = pd.DataFrame(weights * 100, index=valid_tickers, columns=["Weight (%)"]).round(2)
                st.bar_chart(weights_df)
                
                st.markdown(f"**Expected Annual Return:** {portfolio_return*100:.2f}%")
                st.markdown(f"**Annual Volatility (Risk):** {portfolio_volatility*100:.2f}%")
            else:
                st.warning("Please select at least one valid ticker for your portfolio.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def main():
    st.title("Enhanced Portfolio Recommendation System")
    st.markdown("""
        This application helps you create an optimized portfolio based on historical stock data. 
        Choose your stocks from the list, specify a time period, and select an optimization strategy to get started!
    """)

    # Sidebar for user inputs
    st.sidebar.header("Portfolio Inputs")
    
    @st.cache_data
    def load_data():
        nse_ticker_list = pd.read_csv("equity.csv")
        nse_ticker_list = nse_ticker_list[nse_ticker_list['SYMBOL'].notna()]
        nse_ticker_list['SYMBOL'] = nse_ticker_list['SYMBOL'] + ".NS"
    
         # Load BSE-listed tickers and add '.BO' suffix
        bse_ticker_list = pd.read_csv("Equitybse.csv")
        bse_ticker_list = bse_ticker_list[bse_ticker_list['SYMBOL'].notna()]
        bse_ticker_list['SYMBOL'] = bse_ticker_list['SYMBOL'] + ".BO"
    
        # Concatenate NSE and BSE tickers
        ticker_list = pd.concat([nse_ticker_list, bse_ticker_list], ignore_index=True)
    
        # Prepare the combined ticker list for selection
        ticker_list['symbol_name'] = ticker_list['SYMBOL']
        return ticker_list

    ticker_list = load_data()
    tickers_input = st.sidebar.multiselect('portfolio builder', options=ticker_list['symbol_name'], placeholder="search tickers")
    sel_tickers_list = ticker_list[ticker_list['symbol_name'].isin(tickers_input)]['SYMBOL'].tolist()  # Convert to list

    # User-selectable number of years for historical data
    years = st.sidebar.selectbox("Historical Data (Years)", list(range(1, 11)), index=4, help="Select the number of years of historical data")

    if tickers_input:
        if len(sel_tickers_list) < 2:
            st.warning("Please select at least two stocks to proceed with portfolio analysis.")
        else:      # Check if there are any selected tickers
            try:
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=years * 365)

                # Fetch stock data for selected tickers
                stock_data = yf.download(sel_tickers_list, start=start_date, end=end_date)['Adj Close']

                if stock_data.empty:
                    st.warning("No data found for the entered tickers.")
                else:
                    # Display fetched data
                    with st.expander("View Fetched Data"):
                        st.dataframe(stock_data.iloc[::-1])

                    # Allow users to select tickers for optimization
                    selected_tickers = st.multiselect("Select Tickers for Portfolio", sel_tickers_list, default=sel_tickers_list)
                    valid_tickers = [ticker for ticker in selected_tickers if ticker in stock_data.columns]

                    if valid_tickers:
                        returns_df = stock_data[valid_tickers].pct_change().dropna()
                        cov_matrix = returns_df.cov()

                        # Display correlation matrix and heatmap
                        correlation_matrix = returns_df.corr()
                        st.subheader("Correlation Matrix")
                        st.dataframe(correlation_matrix)

                        st.subheader("Correlation Heatmap")
                        sns.set_theme(style="white")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax, cbar=True,
                                    annot_kws={"size": 12, "color": "black"}, linewidths=0.5, linecolor="gray")
                        ax.tick_params(axis='x', colors='black', labelsize=12, labelrotation=45)
                        ax.tick_params(axis='y', colors='black', labelsize=12)
                        fig.patch.set_facecolor('white')
                        ax.set_facecolor('#f5f5f5')
                        st.pyplot(fig)

                        # Portfolio Optimization
                        scenario = st.selectbox("Optimization Strategy", ['Optimal Risk Portfolio', 'Global Minimum Variance', 'Equal Weights', 'Equal Risk Contribution', 'Most Diversified Portfolio'])
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
                            elif scenario == 'Equal Risk Contribution':
                                def risk_contribution(weights):
                                    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                                    marginal_contributions = np.dot(cov_matrix, weights)
                                    return (weights * marginal_contributions) / port_variance  

                                def erc_objective(weights):
                                    rc = risk_contribution(weights)
                                    return np.sum((rc - rc.mean()) ** 2)  
                                result = minimize(erc_objective, num_assets * [1. / num_assets], bounds=bounds, constraints=constraints)
                            elif scenario == 'Most Diversified Portfolio':
                                def mdp_objective(weights):
                                    return -np.linalg.det(cov_matrix @ np.diag(weights))

                                result = minimize(mdp_objective, num_assets * [1. / num_assets], bounds=bounds, constraints=constraints)
                            else:
                                result = {'x': np.array([1. / num_assets] * num_assets)}
                            
                            return result['x']

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

if __name__ == "__main__":
    main()

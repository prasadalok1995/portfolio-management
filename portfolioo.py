import streamlit as st
import portfolio  # This imports your portfolio.py module

def show_portfolio():
    st.title("Portfolio Management Using Quantitative Model")
    st.write("Analyze Risk And Return using an advanced model.")

    # Include the contents of portfolio.py here
    portfolio.main()
if __name__ == "__main__":
    show_portfolio()

import streamlit as st
import prediction  # This will import your prediction.py module

def stock_prediction():
    st.title("Stock Prediction using LLM ")
    st.write("Analyze stock predictions using an advanced model.")

    # Include the contents of prediction.py here
    prediction.main()

if __name__ == "__main__":
    stock_prediction()

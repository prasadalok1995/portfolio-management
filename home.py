import streamlit as st
from streamlit_lottie import st_lottie
import json
from About_me import show_about_me
from stockprediction import stock_prediction  # Import your stock prediction function
from portfolioo import show_portfolio  # Import your portfolio function

st.set_page_config(page_title="Financial App",
                   layout='wide',
                   initial_sidebar_state='expanded')
def load_lottie_file(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

def main():
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = [
    "🏠 Home", 
    "📈 Stock Prediction", 
    "💼 Portfolio Management", 
    "👤 About Me"
] 
    # Adding a unique key to the selectbox
    choice = st.sidebar.selectbox("Select a page", options)
    lottie_animation_1 = load_lottie_file(r"example_animation.json")

    # Page navigation logic
    if choice == "🏠 Home":
        # Directly show the home page content instead of calling main() again
        st.title("Welcome to My Project Portfolio")
        st_lottie(lottie_animation_1, height=300, key="home_animation")
        st.write(""" 
        ## Hi, I'm Alok Kumar.
        Financial and Quantitative Analyst by profession, data enthusiast by passion. 
        I specialize in breaking down complex numbers to help drive smart investment decisions and uncover market opportunities. 
        Here you can explore my work on portfolio management and stock prediction.
        """)
        
        st.markdown(""" 
        ### Disclaimer
        This application is for educational purposes only and not for actual financial trading.
        """)
        st.markdown(""" 
        ### Ticker
        - Choose from NSE-listed stocks like AXISBANK.NS, HDFCBANK.NS, and RELIANCE.NS
        - Dropdown menu for easy selection of individual stocks (indices not included)
        - Select at least two stocks to analyze and optimize your portfolio
        - Explore optimization strategies for optimal returns and risk management
        """)
        st.markdown(""" 
        ### What I Do
        - Analyze stocks for long-term portfolio building
        - Assist investors in diversifying wealth across asset classes and sectors
        - Provide guidance for loan restructuring
        - Help investors reach financial goals with mutual fund investments
        - Support strategic asset allocation to maximize returns
        """)

    elif choice == "👤 About Me":
        show_about_me()  # Call the About Me page content
    elif choice == "📈 Stock Prediction":
        stock_prediction()  # Call your stock prediction function
    elif choice == "💼 Portfolio Management":
        show_portfolio()  # Call your portfolio function
st.sidebar.text("Made with 💕 by Alok")

if __name__ == "__main__":
    main()

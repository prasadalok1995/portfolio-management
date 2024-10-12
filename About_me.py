import streamlit as st
from PIL import Image, ImageOps

def show_about_me():
    st.title("About Me")
    st.write("I am an MBA student specializing in finance and business analytics.")

    col1, col2 = st.columns([1, 2], gap='small', vertical_alignment="center")

    with col1:
    # Load and process the image to be circular
        image = Image.open("DSC_3220.jpg")
        size = (min(image.size), min(image.size))  # Set size to be square for the circular mask

        # Create a mask for circular cropping
        mask = Image.new("L", size, 0)
        circle = Image.new("L", size, 255)
        mask.paste(circle, (0, 0))

        # Crop and mask the image to make it circular
        image = ImageOps.fit(image, size, centering=(0.5, 0.5))
        image.putalpha(mask) 

        st.image(image, width=200)  # Adjust width as needed

    with col2:
        st.subheader("ALOK KUMAR")
        st.write("Financial Data Analyst, Equity Market Research, Technical Analyst")
    

    st.subheader("Education")
    st.write("MBA in Finance and Business Analytics - Jaipuria Institute of Management Indore")
    st.write("BSc in Mathematics - Maharaja Sayajirao University Baroda")

    st.subheader("Experience")
    st.write("""
    Internship at Invest4Edu Ltd, focusing on advanced technical analysis for stock prediction.
    - Time Cycle Analysis: Worked on a project utilizing time cycles to analyze market trends.
    - Fibonacci Applications: Applied Fibonacci theory for price extension and retracement analysis.
    - Gap Theory Implementation: Used Gap Theory to identify new price trends in securities.
             """)
    st.write("""
    Internship at B School Bulls, focusing on equity market and Sector analysis.
    - Equity Market Analysis: Conducted in-depth analysis of equity markets, identifying trends and potential investment opportunities.
    - Sector Research: Researched and evaluated various sectors, assessing their performance and impact on overall market dynamics.
    - Reporting and Insights: Compiled and presented detailed reports on findings, providing actionable insights to enhance investment strategies.        
             """)
    st.write("""
    Completed a winter internship at Annat NGO, raising awareness about food label interpretation.
    """)

    st.subheader("Skills")
    st.write("""
    - Financial modeling and analysis
    - Data visualization with Streamlit
    - Machine learning for stock predictions
    - Strong analytical and quantitative skills
    """)
    st.subheader("Certifications")
    st.write("""
    - Investment Banking Job Simulation from JP Morgan
    - Investment Risk Management from Coursera
    - Google Analytics from Google
    - Introduction to Data Analysis using Microsoft Excel from Coursera
    """)
    st.subheader("Connect with Me",anchor=False)
    st.write("[LinkedIn Profile](https://www.linkedin.com/in/aalokkmr)")
    st.write("☎️ 9097878279")
    st.write("📩 prasadalok86@gmail.com")         

if __name__ == "__main__":
    show_about_me()


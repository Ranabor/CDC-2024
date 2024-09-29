# app.py
import streamlit as st

def main():
    st.title("Basic Streamlit App")
    
    st.write("This is a test Streamlit app.")
    
    number = st.slider("Pick a number", 0, 100, 50)
    
    st.write(f"You selected: {number}")
    
if __name__ == "__main__":
    main()

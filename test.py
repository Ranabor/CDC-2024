# test_app.py
import streamlit as st
from streamlit.testing import ScriptRunner

def test_streamlit_app():
    # Create an instance of ScriptRunner
    runner = ScriptRunner('app.py')
    
    # Run the Streamlit script
    result = runner.run()
    
    # Check if the script ran successfully
    assert result.success, "The app did not run successfully."
    
    # Check if the expected output (e.g., title) is present in the generated HTML
    assert "Basic Streamlit App" in result.stdout, "Title not found in the app output."
    
    # Check if slider is present and its default value is set correctly
    assert "Pick a number" in result.stdout, "Slider not found in the app output."
    assert "You selected: 50" in result.stdout, "Slider default value output is incorrect."


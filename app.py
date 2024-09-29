# app.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('data/Business_Dataset.csv')
flight_stats = df[["Year", "airport_2", "passengers"]]
grouped_flight_stats = flight_stats.groupby(["airport_2", "Year"]).sum().reset_index()
airports = ["ATL", "DFW", "DEN", "LAX", "ORD", "JFK", "MCO", "LAS", "CLT", "MIA"]
colors = {
    'ATL': '#BF3B6A',  # Dark Pink
    'DFW': '#BF8B3B',  # Dark Gold
    'DEN': '#BFBF3B',  # Olive Green
    'LAX': '#3BBFB3',  # Dark Teal
    'ORD': '#3B6ABF',  # Dark Blue
    'JFK': '#6A3BBF',  # Dark Purple
    'MCO': '#BF3B3B',  # Dark Red
    'LAS': '#BF6A3B',  # Dark Orange
    'CLT': '#6ABF3B',  # Dark Lime Green
    'MIA': '#3BBF6A'   # Dark Mint
}
specific_aiports = grouped_flight_stats[np.isin(grouped_flight_stats["airport_2"], airports)]
specific_aiports = specific_aiports[(specific_aiports['Year'] > 1995) & ~np.isin(specific_aiports["Year"], [2020, 2021, 2024])]
specific_aiports = specific_aiports.pivot(index='Year', columns='airport_2', values='passengers').fillna(0)
fig, ax = plt.subplots()
for airport in specific_aiports.columns:
    ax.plot(specific_aiports.index, specific_aiports[airport], marker='o', label=airport, color=colors.get(airport, 'black'))

ax.set_xlabel('Year')
ax.set_ylabel('Passengers')
ax.set_title('Passenger traffic per Airport over Time')
ax.legend()

st.pyplot(fig)


st.title("Basic Streamlit App")

st.write("This is a test Streamlit app.")

number = st.slider("Pick a number", 0, 100, 50)

st.write(f"You selected: {number}")
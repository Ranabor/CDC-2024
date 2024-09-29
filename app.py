import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('data/Business_Dataset.csv')
flight_stats = df[["Year", "airport_2", "passengers"]]
grouped_flight_stats = flight_stats.groupby(["airport_2", "Year"]).sum().reset_index()
airports = ["ATL", "DFW", "DEN", "LAX", "ORD", "JFK", "MCO", "LAS", "CLT", "MIA"]
colors = {
    'ATL': '#F4A3B6',  # Pastel Pink
    'DFW': '#F4D1A3',  # Pastel Gold
    'DEN': '#F4F4A3',  # Pastel Yellow
    'LAX': '#A3F4F0',  # Pastel Teal
    'ORD': '#A3C2F4',  # Pastel Blue
    'JFK': '#C2A3F4',  # Pastel Purple
    'MCO': '#F4A3A3',  # Pastel Red
    'LAS': '#F4BDA3',  # Pastel Orange
    'CLT': '#C2F4A3',  # Pastel Lime Green
    'MIA': '#A3F4C2'   # Pastel Mint
}
specific_airports = grouped_flight_stats[np.isin(grouped_flight_stats["airport_2"], airports)]
specific_airports = specific_airports[(specific_airports['Year'] > 1995) & ~np.isin(specific_airports["Year"], [2020, 2021, 2024])]
specific_airports = specific_airports.pivot(index='Year', columns='airport_2', values='passengers').fillna(0)

min_year = int(specific_airports.index.min())
max_year = int(specific_airports.index.max())

selected_years = st.slider('Select Year Range:', min_year, max_year, (min_year, max_year))

filtered_airports = specific_airports.loc[selected_years[0]:selected_years[1]]

fig, ax = plt.subplots()
for airport in filtered_airports.columns:
    ax.plot(filtered_airports.index, filtered_airports[airport], marker='o', label=airport, color=colors.get(airport, 'black'))

ax.set_xlabel('Year')
ax.set_ylabel('Passengers')
ax.set_title('Passenger traffic per Airport over Time')
ax.legend()

st.pyplot(fig)

airports = flight_stats['airport_2'].unique()
selected_airport = st.selectbox('Select Airport:', airports)
selected_airport2 = st.selectbox('Select Airport:', airports)

st.title("Basic Streamlit App")

st.write("This is a test Streamlit app.")

number = st.slider("Pick a number", 0, 100, 50)

st.write(f"You selected: {number}")
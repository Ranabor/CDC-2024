import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
st.title("Airport Traffic Outbound Passenger Overview")
image = Image.open('./mapAirport.png')
st.image(image, caption='Airport Map', use_column_width=True)

st.title("Passenger Inbound Traffic per Major Airport over Time")
df = pd.read_csv('data/Business_Dataset.csv')
flight_stats = df[["Year", "airport_2", "passengers"]]
grouped_flight_stats = flight_stats.groupby(["airport_2", "Year"]).sum().reset_index()
airports = ["ATL", "DFW", "DEN", "LAX", "ORD", "JFK", "MCO", "LAS", "CLT", "MIA"]
# colors = {
#     'ATL': '#F4A3B6',  # Pastel Pink
#     'DFW': '#F4D1A3',  # Pastel Gold
#     'DEN': '#F4F4A3',  # Pastel Yellow
#     'LAX': '#A3F4F0',  # Pastel Teal
#     'ORD': '#A3C2F4',  # Pastel Blue
#     'JFK': '#C2A3F4',  # Pastel Purple
#     'MCO': '#F4A3A3',  # Pastel Red
#     'LAS': '#F4BDA3',  # Pastel Orange
#     'CLT': '#C2F4A3',  # Pastel Lime Green
#     'MIA': '#A3F4C2'   # Pastel Mint
# }


df_relevant = df[(df["Year"] > 1995) & (df["Year"] != 2024)]

def generate_traffic_graph(airports, start_year, end_year, removed_years = [2024], inbound = True):
    direction = "airport_2" if inbound else "airport_1"
    direction_word = "Inbound" if inbound else "Outbound"
    chosen_airports = df_relevant[np.isin(df_relevant[direction], airports)]
    chosen_airports["passengers"] *= 365 / 1_000_000 / 4
    chosen_airports = chosen_airports[np.isin(chosen_airports["Year"], np.arange(start_year, end_year))]
    chosen_airports = chosen_airports[~np.isin(chosen_airports["Year"], removed_years)]
    grouped_flights = chosen_airports.groupby([direction, "Year"]).sum().reset_index()
    grouped_flights = grouped_flights.pivot(index="Year", columns=direction, values="passengers").fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for airport in grouped_flights.columns:
        ax.plot(grouped_flights.index, grouped_flights[airport], marker='o', label=airport)

    ax.set_title("Number of " + direction_word + " Passengers by Airport Over the Years")
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Passengers (in Millions)')
    ax.set_xticks(grouped_flights.index)
    ax.set_xticklabels(grouped_flights.index, rotation=45)
    ax.legend(title='Airports')
    ax.grid()
    return fig

def get_top_10_airports(inbound = True):
    direction = "airport_2" if inbound else "airport_1"
    summed_airports = df_relevant[[direction, "passengers"]].groupby(direction).sum().reset_index()
    summed_airports = summed_airports.sort_values(by=["passengers"], ascending=False)
    return list(summed_airports[direction][:10])

def get_top_n_airports(n, inbound = True):
    direction = "airport_2" if inbound else "airport_1"
    summed_airports = df_relevant[[direction, "passengers"]].groupby(direction).sum().reset_index()
    summed_airports = summed_airports.sort_values(by=["passengers"], ascending=False)
    return list(summed_airports[direction][:n])


top_inbound_airports = get_top_10_airports()
top_outbound_airports = get_top_10_airports(False)

fig = generate_traffic_graph(top_inbound_airports, 1996, 2024)
st.pyplot(fig)
st.title("Passenger Outbound Traffic per Major Airport over Time")
fig = generate_traffic_graph(top_outbound_airports, 1996, 2024, inbound=False)
st.pyplot(fig)

min_year = 1996
max_year = 2023
st.title("Airport Traffic Comparison Tool")
st.write("This tool allows you to compare the traffic between two airports over a range of years.")
st.write("You can select the airports, the years, and whether you want to see inbound or outbound traffic.")
inbound = st.checkbox('Inbound', value=True)
selected_years = st.slider('Select Year Range:', min_year, max_year, (min_year, max_year))

filtered_airports = df_relevant.loc[selected_years[0]:selected_years[1]]

airports = np.sort(get_top_n_airports(100)).tolist()
selected_airport = st.selectbox('Select Airport 1:', airports, key='airport_1', index=airports.index('SFO'))
selected_airport2 = st.selectbox('Select Airport 2:', airports[1:], key='airport_2', index=airports.index('LAX') + 1)
print([selected_airport, selected_airport2], selected_years[0], selected_years[1] + 1)
fig = generate_traffic_graph([selected_airport, selected_airport2], selected_years[0], selected_years[1] + 1, inbound=inbound)
st.pyplot(fig)

agg_methods = {
    'passengers': 'sum'
}

def weighted_average(x):
    weights = df.loc[x.index, "passengers"]
    if sum(weights) == 0:
        weights = [1 for _ in range(len(weights))]
    return np.average(x, weights=weights)


for col in ['nsmiles','fare','large_ms','fare_lg','lf_ms','fare_low']:
    agg_methods[col] = weighted_average

def sort_group(group):
    group = group[(group['Year'] > 1995) & ~np.isin(group["Year"], [2020, 2021, 2024])]
    group = group.sort_values(by=['Year', 'quarter'])
    return group

df = df.drop(columns=['Geocoded_City1','Geocoded_City2'])
df = df.dropna().reset_index()
inbounds = df.groupby(['airport_2', 'Year', 'quarter']).agg(agg_methods).reset_index().groupby(['airport_2']).apply(sort_group)
inbounds = inbounds.rename(columns={'passengers': 'inbound_passengers', "airport_2": "inbound_ap"})

def linreg(airport, color):
    X = inbounds.reset_index()
    airport_data = X.groupby(['inbound_ap']).get_group(airport)

    X_train, y_train, X_test, y_test = airport_data[airport_data['Year'] < 2017], airport_data[airport_data['Year'] < 2017]['inbound_passengers'], \
        airport_data[airport_data['Year'] >= 2017], airport_data[airport_data['Year'] >= 2017]['inbound_passengers']

    X_train, y_train, X_test, y_test = X_train.reset_index(), y_train.reset_index(), X_test.reset_index(), y_test.reset_index()
    
    model = LinearRegression()
    model.fit(X_train.drop(columns=['inbound_passengers', 'inbound_ap', 'airport_2']), y_train)

    y_pred = model.predict(X_test.drop(columns=['inbound_passengers', 'inbound_ap', 'airport_2']))
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mape.item()

    X_train['date'] = X_train["Year"].astype(str).str.slice(2) + "q" + X_train["quarter"].astype(str)
    X_test['date'] = X_test["Year"].astype(str).str.slice(2) + "q" + X_test["quarter"].astype(str)
    
    airport_indices = list([X_test[X_test['inbound_ap'] == a].index for a in [airport]][0])
    airport_indices_train = list([X_train[X_train['inbound_ap'] == a].index for a in [airport]][0])
    
    # plt.plot(pd.concat([X_train.iloc[airport_indices_train]['date'], X_test.iloc[airport_indices]['date']]), 
    #          pd.concat([y_train.iloc[airport_indices_train]['inbound_passengers'], y_test.iloc[airport_indices]['inbound_passengers']]), 
    #          color=color, label=airport)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pd.concat([X_train.iloc[airport_indices_train]['date']]), 
             pd.concat([y_train.iloc[airport_indices_train]['inbound_passengers']]), 
             color=color, label=airport)
    ax.plot(X_test.iloc[airport_indices]['date'], y_pred[airport_indices][:, 1], color=color, linestyle='dashed')

    ax.set_xlabel('Year + Quarter')
    ax.set_ylabel('Inbound Passengers')
    ax.set_title('Inbound Passengers across time')
    ax.axvline(x=X_train['date'].iloc[-1], color='black', linestyle=':', label='Prediction Start')

    ax.tick_params(left=True, right=False, labelleft=True, labelbottom=False, bottom=True) 
    return fig

st.title(f"Inbound Passengers Prediction {selected_airport}")
fig = linreg(selected_airport, 'blue')
st.pyplot(fig)

st.title(f"Inbound Passengers Prediction {selected_airport2}")
fig = linreg(selected_airport2, 'red')
st.pyplot(fig)
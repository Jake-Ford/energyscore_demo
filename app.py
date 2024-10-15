import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd  # For handling GeoJSON data
import folium
import branca.colormap as cm
from streamlit_folium import folium_static  # Import to render folium maps in Streamlit
from sklearn.metrics import accuracy_score


# Load the GeoJSON file
#zip_geojson = gpd.read_file('us_zips.geojson')
zip_geojson = gpd.read_file('select_zips.geojson')

# Load person data
person_data = pd.read_csv('data/standard/experian_rf_stats_person.csv')
person_data = person_data.head(10000)

person_data['ZIP'] = person_data['ZIP'].astype(str).str.zfill(5)
zip_geojson['ZIP'] = zip_geojson['ZCTA5CE10'].astype(str).str.zfill(5)

# App title and description
st.title("EnergyScore ZIP Code-level Default Risk Analysis")
st.write("""
         This app displays the default risk predictions from EnergyScore and allows comparison with FICO scores across different ZIP codes.
         """)

# Sidebar for user inputs
fico_threshold = st.sidebar.slider("Select FICO Score Threshold", 300, 850, 650, step=10)
energy_score_threshold = st.sidebar.slider("EnergyScore Threshold for High Risk", 0.0, 1.0, 0.75, step=0.01)

# Function to calculate metrics for each ZIP
def calculate_zip_metrics(stats_data_person, fico_cutoff, energyscore_cutoff):
    # Create masks for conditions
    stats_data_person['FICO_PASS'] = stats_data_person['FICO_V9_SCORE'] > fico_cutoff
    stats_data_person['ENERGYSCORE_PASS'] = stats_data_person['WEIGHTED_ENERGYSCORE'] > energyscore_cutoff

    # Function to calculate various metrics
    def calc_metrics(group):
        total_population = len(group)
        if total_population == 0:
            return pd.Series({
                'Total Population': 0,
                'Percent Below FICO': 0,
                'Percent Above FICO': 0,
                'FICO Accuracy': np.nan,
                'EnergyScore Accuracy': np.nan,
                'Qualification Increase': 0,
            })
        
        
        below_fico = group[group['FICO_PASS'] == False]
        above_fico = group[group['FICO_PASS'] == True]

        if len(below_fico) == 0:
            return pd.Series({
                'Total Population': total_population,
                'Percent Below FICO': 0,
                'Percent Above FICO': len(above_fico) / total_population,
                'FICO Accuracy': np.nan,
                'EnergyScore Accuracy': np.nan,
                'Qualification Increase': 0,
            })

        below_fico_pass = below_fico[below_fico['WEIGHTED_ENERGYSCORE'] <= energyscore_cutoff]
        below_fico_fail = below_fico[below_fico['WEIGHTED_ENERGYSCORE'] > energyscore_cutoff]

        pct_below_fico = len(below_fico) / total_population
        pct_above_fico = len(above_fico) / total_population

        if len(below_fico_pass) == 0:
            percent_increase_in_qualifications = 0
        else:
            percent_increase_in_qualifications = (len(below_fico_pass) / total_population) * 100

        # get FICO accuracy, precision, recall, f1 and roc_auc score
       # fico_accuracy = accuracy_score(below_fico['WEIGHTED_ACTUAL_OUTPUT'], below_fico['WEIGHTED_ENERGYSCORE'] > energyscore_cutoff) if len(below_fico) > 0 else np.nan
        fico_accuracy = accuracy_score(below_fico['WEIGHTED_ACTUAL_OUTPUT'], below_fico['FICO_PASS']) if len(below_fico) > 0 else np.nan

        energy_accuracy = accuracy_score(below_fico['WEIGHTED_ACTUAL_OUTPUT'], below_fico['ENERGYSCORE_PASS']) if len(below_fico) > 0 else np.nan

      #  accuracy_increase = energy_accuracy - fico_accuracy

        return pd.Series({
            'Total Population': total_population,
            'Percent Below FICO': pct_below_fico,
            'Percent Above FICO': pct_above_fico,
            'FICO Accuracy': fico_accuracy,
            'EnergyScore Accuracy': energy_accuracy,
            'Qualification Increase': percent_increase_in_qualifications,
          #  'Accuracy Percentage Increase': accuracy_increase,

           # 'Accuracy Percentage Increase': (energy_accuracy - fico_accuracy) / fico_accuracy * 100 if fico_accuracy * energy_accuracy> 0 else 0,
           

        })

    # Group by ZIP and apply metrics calculation
    zip_metrics = stats_data_person.groupby('ZIP').apply(calc_metrics)

    zip_metrics = zip_metrics.reset_index()
    return zip_metrics

# Calculate metrics and merge with geo data
zip_metrics = calculate_zip_metrics(person_data, fico_threshold, energy_score_threshold)
zip_level_geo = pd.merge(zip_metrics, zip_geojson, on='ZIP', how='left')
zip_level_geo = zip_level_geo.dropna(subset=['geometry'])
zip_level_geo = gpd.GeoDataFrame(zip_level_geo, geometry='geometry')

# Set up Folium map
m = folium.Map()#location=[37.7749, -122.4194], zoom_start=10)

# Create colormap based on EnergyScore Accuracy
min_value = zip_level_geo['Qualification Increase'].min()
max_value = zip_level_geo['Qualification Increase'].max()
colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'red'],
                             vmin=min_value, vmax=max_value,
                             caption='Qualification Increase')

# Function to style polygons based on EnergyScore Accuracy
def style_function(feature):
    accuracy = feature['properties']['Qualification Increase']
    return {
        'fillOpacity': 0.7,
        'weight': 0.5,
        'color': 'black',
        'fillColor': colormap(accuracy) if accuracy else 'gray'
    }

# Add GeoJson layer with styling
folium.GeoJson(
    zip_level_geo.__geo_interface__,
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(fields=['ZIP', 'EnergyScore Accuracy', 'FICO Accuracy', 'Qualification Increase' ], aliases=['ZIP Code', 'EnergyScore Accuracy', 
                                                                                                                               'FICO Accuracy', 'Qualification Increase'])
).add_to(m)

# Add colormap legend
colormap.add_to(m)

# Display the map using folium_static
folium_static(m)

# Display metrics in sidebar
selected_zip = st.sidebar.selectbox('Select a ZIP Code for Detailed Analysis', zip_level_geo['ZIP'].unique())
selected_zip_data = zip_level_geo[zip_level_geo['ZIP'] == selected_zip]
st.sidebar.write(f"EnergyScore Accuracy: {selected_zip_data['EnergyScore Accuracy'].values[0]:.2f}")
st.sidebar.write(f"FICO Accuracy Accuracy: {selected_zip_data['FICO Accuracy'].values[0]:.2f}")
st.sidebar.write(f"Qualification Increase: {selected_zip_data['Qualification Increase'].values[0]:.2f}")
st.sidebar.write(f"Accuracy Percentage Increase: {selected_zip_data['Accuracy Percentage Increase'].values[0]:.2f}")



# Additional data details
if st.checkbox('Show detailed data for selected ZIP'):
    st.write(zip_level_geo[zip_level_geo['ZIP'] == selected_zip])

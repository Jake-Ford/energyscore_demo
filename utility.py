import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd  # For handling GeoJSON data
import folium
import branca.colormap as cm
import hmac
# Import to render folium maps in Streamlit
from streamlit_folium import folium_static
from sklearn.metrics import accuracy_score


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


with st.spinner("Loading map..."):
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    st.title("EnergyScore Utility Level Default Risk Analysis")
    st.write("""
           This app displays the default risk predictions from EnergyScore and allows comparison with FICO scores across different Utility zones.
           """)

    # Sidebar for user inputs
    fico_threshold = st.sidebar.slider(
        "Select FICO Score Threshold", 300, 850, 650, step=10)
    energy_score_threshold = st.sidebar.slider(
        "EnergyScore Threshold for High Risk", 0.0, 1.0, 0.75, step=0.01)

    # Load the GeoJSON file
    zip_geojson = gpd.read_file('demo_zips.geojson')

    # Load person data, forcing ZIP to be read as strings
    person_data = pd.read_csv('data.csv', dtype={'ZIP': str})

    # Ensure ZIP codes have leading zeros and handle floats
    person_data['ZIP'] = person_data['ZIP'].apply(
        lambda x: str(int(float(x))).zfill(5) if pd.notnull(x) else '')

    # Ensure GeoJSON ZIP codes are formatted as strings with leading zeros
    zip_geojson['ZIP'] = zip_geojson['ZCTA5CE10'].astype(str).str.zfill(5)

    # Function to load utility zone data based on state

    def load_state_util(state_name):
        if state_name == 'New Mexico':
            temp = gpd.read_file('nm_utils.geojson')
            temp = temp[['new_name', 'geometry']]
            return temp
        elif state_name == 'Massachusetts':
            return gpd.read_file('ma_utils.geojson')

    # Function to calculate metrics for each ZIP

    def calculate_zip_metrics(stats_data_person, fico_threshold, energy_score_threshold):
        stats_data_person['FICO_PASS'] = stats_data_person['FICO_V9_SCORE'] > fico_threshold
        stats_data_person['ENERGYSCORE_PASS'] = stats_data_person['WEIGHTED_ENERGYSCORE'] > energy_score_threshold

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

            below_fico_pass = below_fico[below_fico['WEIGHTED_ENERGYSCORE']
                                         <= energy_score_threshold]
            pct_below_fico = len(below_fico) / total_population
            pct_above_fico = len(above_fico) / total_population

            percent_increase_in_qualifications = (
                len(below_fico_pass) / total_population) * 100 if len(below_fico_pass) > 0 else 0

            fico_accuracy = accuracy_score(
                below_fico['WEIGHTED_ACTUAL_OUTPUT'], below_fico['FICO_PASS']) if len(below_fico) > 0 else np.nan
            energy_accuracy = accuracy_score(
                below_fico['WEIGHTED_ACTUAL_OUTPUT'], below_fico['ENERGYSCORE_PASS']) if len(below_fico) > 0 else np.nan

            return pd.Series({
                'Total Population': total_population,
                'Percent Below FICO': pct_below_fico,
                'Percent Above FICO': pct_above_fico,
                'FICO Accuracy': fico_accuracy,
                'EnergyScore Accuracy': energy_accuracy,
                'Qualification Increase': percent_increase_in_qualifications,
            })

        # Group by ZIP and apply metrics calculation
        zip_metrics = stats_data_person.groupby('ZIP').apply(calc_metrics)
        zip_metrics = zip_metrics.reset_index()
        return zip_metrics

    # Function to calculate ZIP to utility mapping and display on the map

    def calculate_zip_to_util(zip_level_geo, state_name):
        # Load the utility data for the state
        state_util = load_state_util(state_name)
        state_util.rename(columns={'new_name': 'Utility'}, inplace=True)

        # Ensure ZIP code geometries have the same projection as the utility data
        zip_level_geo = zip_level_geo.to_crs(state_util.crs)

        # Convert the ZIP geometries to representative points
        zip_level_geo['geometry'] = zip_level_geo.representative_point()

        # Perform spatial join with utility data based on point locations
        zip_level_geo = gpd.sjoin(
            zip_level_geo, state_util, how='left', predicate='within')

        # Group by utility name ('new_name') and calculate the mean of 'Qualification Increase'
        zip_to_util = zip_level_geo.groupby(
            'Utility')['Qualification Increase'].mean().reset_index()

        # Merge utility data with the calculated qualification increase
        state_util = state_util.merge(zip_to_util, on='Utility', how='left')

        return state_util

    # Streamlit sidebar input to select a state
    state_name = st.sidebar.selectbox(
        "Select State", ["New Mexico", "Massachusetts"])

    # Calculate metrics and merge with geo data
    zip_metrics = calculate_zip_metrics(
        person_data, fico_threshold, energy_score_threshold)
    zip_level_geo = pd.merge(zip_metrics, zip_geojson, on='ZIP', how='left')
    zip_level_geo = zip_level_geo.dropna(subset=['geometry'])
    zip_level_geo = gpd.GeoDataFrame(zip_level_geo, geometry='geometry')

    # Use the new function to calculate ZIP to utility metrics
    state_util = calculate_zip_to_util(zip_level_geo, state_name)

    # Display utility-level metrics
    st.write(f"Utility Qualification Increase for {state_name}")
    st.write(state_util[['Utility', 'Qualification Increase']])

    def get_state_coordinates(state_name):
        if state_name == 'New Mexico':
            return 34.9727, -105.0324
        elif state_name == 'Massachusetts':
            return 42.4072, -71.3824

    # Display map with Folium
    lat, lng = get_state_coordinates(state_name=state_name)
    m = folium.Map(location=[lat, lng], zoom_start=6)

    # Create colormap based on Qualification Increase
    min_value = min(zip_level_geo['Qualification Increase'].min(
    ), state_util['Qualification Increase'].min())
    max_value = max(zip_level_geo['Qualification Increase'].max(
    ), state_util['Qualification Increase'].max())
    colormap = cm.LinearColormap(colors=["#fde725", "#35b779", "#31688e", "#440154"],
                                 vmin=min_value, vmax=max_value,
                                 caption='Qualification Increase')

    # Add ZIP layer
    # zip_layer = folium.FeatureGroup(name='ZIP Codes')
    # folium.GeoJson(
    #     zip_level_geo.__geo_interface__,
    #     style_function=lambda feature: {
    #         'fillOpacity': 0.7,
    #         'weight': 0.5,
    #         'color': 'black',
    #         'fillColor': colormap(feature['properties']['Qualification Increase']) if feature['properties']['Qualification Increase'] else 'gray'
    #     },
    #     tooltip=folium.GeoJsonTooltip(fields=['ZIP', 'EnergyScore Accuracy', 'FICO Accuracy', 'Qualification Increase'],
    #                                   aliases=['ZIP Code', 'EnergyScore Accuracy', 'FICO Accuracy', 'Qualification Increase'])
    # ).add_to(zip_layer)

    # Add Utility Layer
    utility_layer = folium.FeatureGroup(name='Utility Zones')
    folium.GeoJson(
        state_util.__geo_interface__,
        style_function=lambda feature: {
            'fillOpacity': 0.7,
            'weight': 0.5,
            'color': 'black',
            'fillColor': colormap(feature['properties']['Qualification Increase']) if feature['properties']['Qualification Increase'] else 'gray'
        },
        tooltip=folium.GeoJsonTooltip(fields=['Utility', 'Qualification Increase'], aliases=[
                                      'Utility Zone', 'Qualification Increase'])
    ).add_to(utility_layer)

    # Add layers to map
    # zip_layer.add_to(m)
    utility_layer.add_to(m)

    # Add LayerControl so users can toggle between layers
    folium.LayerControl().add_to(m)

    # Add colormap legend
    colormap.add_to(m)

    # Display the map using folium_static
    folium_static(m)

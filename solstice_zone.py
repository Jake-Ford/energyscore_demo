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
import plotly.express as px

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


with st.spinner("Loading map..."):
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    st.title("EnergyScore Utility Level Default Risk Analysis")
    st.write("""
            By using over 30 million tradeline records across 2 million individuals, EnergyScore is able to more accurately and inclusively predict risk of default payment than FICO scores alone. This tool allows you to explore the default risk predictions for different Solstice territories based on FICO Score and EnergyScore thresholds.
             
            To begin, select the Solstice Territory, FICO Score and EnergyScore thresholds to view the default risk predictions. The model will calculate the inputs and output aggregated statistics. Disaggregated data is available on the zip code level and is displayed on the map below. Note, samples sizes for some zip codes are small and may not be representative of the entire population.
           """)
    
    st.write(""" 
                - **Average EnergyScore**: The average EnergyScore for the selected Solstice Territory; 0 indicates low risk and 1 indicates high risk.
                - **Average FICO Score**: The average FICO Score for the selected Solstice Territory; on a scale from 300 to 850.
                - **Accuracy Percentage Inrease**: The percentage increase in accuracy when using EnergyScore compared to FICO Score.
                - **Average Qualification Increase**: The average increase across all zip codes for the number of individuals who would be approved using EnergyScore compared to FICO Score.
                - **Sub-FICO EnergyScore Accuracy**: The accuracy of EnergyScore for individuals below the FICO threshold.
             """)

    # Sidebar for user inputs
    fico_threshold = st.sidebar.slider(
        "Select FICO Score Threshold", 300, 850, 650, step=10)
    energy_score_threshold = st.sidebar.slider(
        "EnergyScore Threshold for High Risk", 0.0, 1.0, 0.5, step=0.01)
    
    solstice_territory_name = st.sidebar.selectbox(
        "Select Solstice Territory", ["Ameren Illinois", "Eversource - Western MA", 
                                      "Ameren - Mercer County & Surrounding", "Central Hudson",
                                      "ComEd - IL", "Con Edison - NY", "EPE (El Paso Electric)",
                                      "Eversource - Eastern MA", "Eversource - Western MA",
                                      "Eversource - Greater Boston", "Eversource - MA", 
                                      "Eversource - Southeast MA", "JCP&L", "National Grid - MA", 
                                      "National Grid - NY", "National Grid MA NEMA"]
    )

    def get_solstice_territory_geojson(solstice_territory_name):
        load_name = "filtered_geojsons/" + solstice_territory_name + '.geojson'
        return gpd.read_file(load_name)

    zip_geojson = get_solstice_territory_geojson(solstice_territory_name)
    #person_data = pd.read_csv('data.csv', dtype={'ZIP': str})
    person_data = pd.read_csv('data/combined_rf_stats_person.csv')
    person_data['ZIP'] = person_data['ZIP'].apply(
        lambda x: str(int(float(x))).zfill(5) if pd.notnull(x) else '')
    zip_geojson['ZIP'] = zip_geojson['ZCTA5CE10'].astype(str).str.zfill(5)

   

    def calculate_zip_metrics(stats_data_person, fico_threshold, energy_score_threshold):

        stats_data_person['FICO_PASS'] = np.where(stats_data_person['FICO_V9_SCORE'] >= fico_threshold, 0, 1)
        stats_data_person['ENERGYSCORE_PASS'] = np.where(stats_data_person['WEIGHTED_ENERGYSCORE'] >= energy_score_threshold, 1, 0)
      #  stats_data_person['ENERGYSCORE_PASS'] = stats_data_person['WEIGHTED_ENERGYSCORE'] > energy_score_threshold
       # stats_data_person['FICO_PASS'] = stats_data_person['FICO_V9_SCORE'] < fico_threshold
       # stats_data_person['ENERGYSCORE_PASS'] = stats_data_person['WEIGHTED_ENERGYSCORE'] > energy_score_threshold

        total_population = len(stats_data_person)

        total_below_fico = stats_data_person[stats_data_person['FICO_PASS'] == 0]

        total_es_accuracy = accuracy_score(stats_data_person['WEIGHTED_ACTUAL_OUTPUT'], stats_data_person['ENERGYSCORE_PASS'])
        total_fico_accuracy = accuracy_score(stats_data_person['WEIGHTED_ACTUAL_OUTPUT'], stats_data_person['FICO_PASS'])

        # how many people below the fico threshold would have been approved by energyscore
        below_fico_pass = total_below_fico[total_below_fico['WEIGHTED_ENERGYSCORE'] <= energy_score_threshold]
        below_fico_fail = total_below_fico[total_below_fico['WEIGHTED_ENERGYSCORE'] > energy_score_threshold]

        below_fico_pass_count = len(below_fico_pass)


        # what is the accuracy of this marginal approval using EnergyScore

        below_fico_es_accuracy_control = accuracy_score(total_below_fico['WEIGHTED_ACTUAL_OUTPUT'], total_below_fico['ENERGYSCORE_PASS'])

        percent_increase_in_qualifications_total = (below_fico_pass_count / total_population) * 100


    def calculate_zip_metrics(stats_data_person, fico_threshold, energy_score_threshold):

        stats_data_person['FICO_FAIL'] = np.where(stats_data_person['FICO_V9_SCORE'] >= fico_threshold, 0, 1)
        stats_data_person['ENERGYSCORE_FAIL'] = np.where(stats_data_person['WEIGHTED_ENERGYSCORE'] >= energy_score_threshold, 1, 0)

        total_population = len(stats_data_person)

        total_below_fico = stats_data_person[stats_data_person['FICO_FAIL'] == 1]

        total_es_accuracy = accuracy_score(stats_data_person['WEIGHTED_ACTUAL_OUTPUT'], stats_data_person['ENERGYSCORE_FAIL'])
        total_fico_accuracy = accuracy_score(stats_data_person['WEIGHTED_ACTUAL_OUTPUT'], stats_data_person['FICO_FAIL'])

        # how many people below the fico threshold would have been approved by energyscore
        below_fico_pass = total_below_fico[total_below_fico['ENERGYSCORE_FAIL'] ==1]
        below_fico_fail = total_below_fico[total_below_fico['ENERGYSCORE_FAIL'] == 0 ]

        below_fico_pass_count = len(below_fico_pass)


        # what is the accuracy of this marginal approval using EnergyScore

        below_fico_es_accuracy_control = accuracy_score(total_below_fico['WEIGHTED_ACTUAL_OUTPUT'], total_below_fico['ENERGYSCORE_FAIL'])

        percent_increase_in_qualifications_total = (below_fico_pass_count / total_population) * 100

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

            below_fico = group[group['FICO_FAIL'] == 1]
            above_fico = group[group['FICO_FAIL'] == 0]

            below_fico_pass = below_fico[below_fico['ENERGYSCORE_FAIL'] ==0]
            
            pct_below_fico = len(below_fico) / total_population
            pct_above_fico = len(above_fico) / total_population
            percent_increase_in_qualifications = (len(below_fico_pass) / total_population) * 100 

            #group['FICO_PREDICTION'] = group['FICO_FAIL'] == (group['WEIGHTED_ACTUAL_OUTPUT'] == 0)
            fico_accuracy = accuracy_score(group['WEIGHTED_ACTUAL_OUTPUT'], group['FICO_FAIL'])

        #  group['ENERGYSCORE_PREDICTION'] = group['ENERGYSCORE_FAIL'] == (group['WEIGHTED_ACTUAL_OUTPUT'] == 0)

            energy_accuracy = accuracy_score(group['WEIGHTED_ACTUAL_OUTPUT'], group['ENERGYSCORE_FAIL'])

            below_fico_es_accuracy = accuracy_score(below_fico['WEIGHTED_ACTUAL_OUTPUT'], below_fico['ENERGYSCORE_FAIL'])

            avg_fico = group['FICO_V9_SCORE'].mean()
            avg_es = (group['WEIGHTED_ENERGYSCORE'].mean())*100

            if fico_accuracy == 0:
                pct_increase_accuracy_es = 0
            else:
                pct_increase_accuracy_es = (energy_accuracy - fico_accuracy) / fico_accuracy * 100

            return pd.Series({
                'Total Population': total_population,
                'Percent Below FICO': pct_below_fico,
                'Percent Above FICO': pct_above_fico,
                'FICO Accuracy': fico_accuracy,
                'Total EnergyScore Accuracy': energy_accuracy,
                'Sub-FICO EnergyScore Accuracy': below_fico_es_accuracy,
                'Increase in Accuracy': pct_increase_accuracy_es,
                'Qualification Increase': percent_increase_in_qualifications,
                'Average FICO': avg_fico,
                'Average EnergyScore': avg_es,
                'Control ES Accuracy': total_es_accuracy, 
                'Control FICO Accuracy' : total_fico_accuracy,
                'Control Marginal Accuracy': (total_es_accuracy - total_fico_accuracy) / total_fico_accuracy * 100,
                'below_fico_pass_count': below_fico_pass_count, 
                'below_fico_es_accuracy': below_fico_es_accuracy_control, 
                'percent_increase_in_qualifications': percent_increase_in_qualifications_total

            })

        zip_metrics = stats_data_person.groupby('ZIP').apply(calc_metrics)
        zip_metrics = zip_metrics.reset_index()
        return zip_metrics

    def calculate_zip_to_util(solstice_territory_name):
        state_util = get_solstice_territory_geojson(solstice_territory_name)
        zip_metrics = calculate_zip_metrics(person_data, fico_threshold, energy_score_threshold)
        zip_level_geo = pd.merge(zip_metrics, state_util, on='ZIP', how='left')
        zip_level_geo = zip_level_geo.dropna(subset=['geometry'])
        zip_level_geo = gpd.GeoDataFrame(zip_level_geo, crs="EPSG:4326", geometry=zip_level_geo['geometry'])
        return zip_level_geo

   # temp = calculate_zip_to_util(solstice_territory_name)

    zip_level_geo = calculate_zip_to_util(solstice_territory_name)

    # Display the top 10 ZIP codes by Qualification Increase (less than 100)
    st.subheader(f"Top 10 ZIP Codes for Qualification Increase in {solstice_territory_name}")
    top_10_zip_codes = zip_level_geo[zip_level_geo['Qualification Increase'] < 100].nlargest(10, 'Qualification Increase')
   # st.dataframe(top_10_zip_codes[['ZIP', 'Qualification Increase']])

    top_10_zip_codes['ZIP'] = top_10_zip_codes['ZIP'].astype(str)
    top_10_zip_codes = top_10_zip_codes.sort_values(by='Qualification Increase', ascending=False)



    # Bar plot for top 10 ZIP codes
    fig = px.bar(top_10_zip_codes, x='ZIP', y='Qualification Increase', title=" ")
    fig.update_xaxes(type='category', tickmode='array', tickvals=top_10_zip_codes['ZIP'], ticktext=top_10_zip_codes['ZIP'])

    st.plotly_chart(fig)


    # to do: add how many people would have been approved 
    # below the fico threshold: % of sub-FICO approved, accuracy sub-FICO 

    # Average Qualification Increase for the territory
    avg_qualification_increase = zip_level_geo['Qualification Increase'].mean()
    avg_fico = zip_level_geo['Average FICO'].mean()
    avg_es = zip_level_geo['Average EnergyScore'].mean()
    avg_accuracy_imp = zip_level_geo['Control Marginal Accuracy'].mean()

    avg_sub_fico_es = (zip_level_geo['Sub-FICO EnergyScore Accuracy'].mean())*100

    # Add the avg_qualification_increase to the sidebar as a metric
    
    st.sidebar.metric(label="Average EnergyScore: ", value = f"{avg_es:.1f}")

    st.sidebar.metric(label="Average FICO Score: ", value = f"{avg_fico:.0f}")

    st.sidebar.metric(label = "Accuracy Percentage Increase", value = f"{avg_accuracy_imp:.1f}%")

    st.sidebar.metric(label="Average Qualification Increase", value=f"{avg_qualification_increase:.1f}%")


    st.sidebar.metric(label="Sub-FICO EnergyScore Accuracy: ", value = f"{avg_sub_fico_es:.1f}%")
    

    st.subheader(f"Average Qualification Increase for {solstice_territory_name}")
    st.write(f"The average qualification increase for {solstice_territory_name} is {avg_qualification_increase:.2f}%.")


    def get_state_coordinates(solstice_territory_name):
        # Illinois
        if solstice_territory_name in ['Ameren - Mercer County & Surrounding', 'Ameren Illinois']:
            return 39.7727, -89.6501
        # MA & NY
        elif solstice_territory_name in ['Eversource - Western MA', 'Eversource - Eastern MA', 'Eversource - Greater Boston',
                                      'Eversource - MA', 'Eversource - Southeast MA', 'National Grid - MA', 'National Grid - NY',
                                      'National Grid MA NEMA']:
            return 42.4072, -71.3824
        elif solstice_territory_name in ['PNM (Public Service Co. of New Mexico)', 'El Paso Electric']:
            return 35.0844, -106.6504
        else:
            return 40.7128, -74.0060
            

    # Display map with Folium
    lat, lng = get_state_coordinates(solstice_territory_name=solstice_territory_name)
    m = folium.Map(location=[lat, lng], zoom_start=5)

    # Create colormap based on Qualification Increase
    min_value = min(zip_level_geo['Sub-FICO EnergyScore Accuracy'].min(
    ), zip_level_geo['Sub-FICO EnergyScore Accuracy'].min())
    max_value = max(zip_level_geo['Qualification Increase'].max(
    ), zip_level_geo['Sub-FICO EnergyScore Accuracy'].max())
    colormap = cm.LinearColormap(colors=["#fde725", "#35b779", "#31688e", "#440154"],
                                 vmin=min_value, vmax=max_value,
                                 caption='Sub-FICO EnergyScore Accuracy')

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
        zip_level_geo.__geo_interface__,
        style_function=lambda feature: {
            'fillOpacity': 0.7,
            'weight': 0.5,
            'color': 'black',
            'fillColor': colormap(feature['properties']['Qualification Increase']) if feature['properties']['Qualification Increase'] else 'gray'
        },
        tooltip=folium.GeoJsonTooltip(fields=['ZIP', 'Sub-FICO EnergyScore Accuracy', 'Qualification Increase', 'FICO Accuracy', 
                                              'Total EnergyScore Accuracy', 'Increase in Accuracy'], aliases=[
                                      'ZIP', 'Sub-FICO EnergyScore Accuracy', 'Qualification Increase', 'FICO Accuracy', 
                                              'Total EnergyScore Accuracy', 'Increase in Accuracy'])
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

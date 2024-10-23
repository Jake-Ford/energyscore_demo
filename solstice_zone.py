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
import plotly.graph_objects as go


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

    # Define a function to categorize the predictions based on thresholds
    def calculate_loan_outcome(df, fico_threshold, energy_score_threshold, use_es=True):
        df['FICO_APPROVED'] = np.where(df['FICO_V9_SCORE'] >= fico_threshold, 1, 0)
        df['ES_APPROVED'] = np.where(df['WEIGHTED_ENERGYSCORE'] < energy_score_threshold, 1, 0)

        df['PAYBACK'] = df['WEIGHTED_ACTUAL_OUTPUT']  # 1 means default, 0 means pays back

        if use_es == True:
        
            # Classification categories:
            df['Category'] = np.select(
                [
                    (df['ES_APPROVED'] == 0) & (df['PAYBACK'] == 1),  # True Negative
                    (df['ES_APPROVED'] == 1) & (df['PAYBACK'] == 0),  # True Positive
                    (df['ES_APPROVED'] == 1) & (df['PAYBACK'] == 1),  # False Positive
                    (df['ES_APPROVED'] == 0) & (df['PAYBACK'] == 0),  # False Negative
                ],
                ['True Negative', 'True Positive', 'False Positive', 'False Negative'],
                default='Unknown'
            )
            return df
        else:
            df['Category'] = np.select(
                [
                    (df['FICO_APPROVED'] == 0) & (df['PAYBACK'] == 1),  # True Negative
                    (df['FICO_APPROVED'] == 1) & (df['PAYBACK'] == 0),  # True Positive
                    (df['FICO_APPROVED'] == 1) & (df['PAYBACK'] == 1),  # False Positive
                    (df['FICO_APPROVED'] == 0) & (df['PAYBACK'] == 0),  # False Negative
                ],
                ['True Negative', 'True Positive', 'False Positive', 'False Negative'],
                default='Unknown'
            )
            return df 
            return

    # Streamlit app
    st.title("Loan Threshold Simulation")
    st.write("""
             The below tool simulates loan approvals using thresholds based on FICO and EnergyScore. The two parameters are used to assume a succesful loan nets 300 dollars, and the cost of a false positive is 700 dollars. 
            
             For example, with a 650 FICO cutoff compared to a 0.5 EnergyScore cutoff, profits increase by 40%. This result will change 
             
             """)

    # Sidebar for profit and cost inputs
    true_positive_profit = st.sidebar.number_input("Profit for a successful loan (True Positive)", value=300)
    false_positive_cost = st.sidebar.number_input("Cost for a defaulted loan (False Positive)", value=-700)

    # Process the data
    df_outcome = calculate_loan_outcome(person_data, fico_threshold, energy_score_threshold, use_es=True)

    # Define the sampling percentage (e.g., 1% of the total data)
   # sampling_percentage = st.sidebar.slider("Sampling Percentage", 0.01, 1.0, 0.05)

    sampling_percentage = 0.05
    df_sampled = df_outcome.sample(frac=0.05)

    # Randomly sample the data for the plot
    df_sampled = df_outcome.sample(frac=sampling_percentage)

    # Count results for the outcome summary
    summary = df_sampled['Category'].value_counts().to_dict()

    # Create the Plotly scatter plot
    fig = px.scatter(
        df_sampled, 
        x="FICO_V9_SCORE", 
        y="WEIGHTED_ENERGYSCORE", 
        color="Category",
        color_discrete_map={
            'True Positive': 'blue',
            'False Positive': 'lightblue',
            'True Negative': 'green',
            'False Negative': 'lightgreen'
        },
        title=f"EnergyScore Decision Outcome (Sampled {sampling_percentage * 100:.0f}%)"
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title="FICO Score",
        yaxis_title="EnergyScore",
        showlegend=True,
        height=600,
        width=900,
    )

    st.plotly_chart(fig)

    # Outcome summary
    st.write("### EnergyScore Decision Summary")
    st.write(f"**True Positives**: {summary.get('True Positive', 0)}")
    st.write(f"**False Positives**: {summary.get('False Positive', 0)}")
    st.write(f"**True Negatives**: {summary.get('True Negative', 0)}")
    st.write(f"**False Negatives**: {summary.get('False Negative', 0)}")

    # Add threshold indicators and summary stats
    correct_rate = ((summary.get('True Positive', 0) + summary.get('True Negative', 0)) / len(df_sampled)) * 100

    # Calculate profit
    true_positives = summary.get('True Positive', 0)
    false_positives = summary.get('False Positive', 0)

    # Total profit: profit from True Positives + cost from False Positives
    profit = (true_positives * true_positive_profit) + (false_positives * false_positive_cost)

    # Calculate other metrics
    true_positive_rate = (summary.get('True Positive', 0) / len(df_sampled[df_sampled['PAYBACK'] == 0])) * 100
    positive_rate = (summary.get('True Positive', 0) / len(df_sampled)) * 100

    st.write(f"**Correct Rate**: {correct_rate:.2f}%")
    st.write(f"**Profit**: ${profit:,.0f}")
    st.write(f"**True Positive Rate**: {true_positive_rate:.2f}%")
    st.write(f"**Positive Rate**: {positive_rate:.2f}%")

    temp_es_profit = profit



    # ---------------------- FICO Thresholds --------------------------

    df_outcome = calculate_loan_outcome(person_data, fico_threshold, energy_score_threshold, use_es=False)
    df_sampled = df_outcome.sample(frac=sampling_percentage)

    # Count results for FICO Score
    fico_summary = df_sampled['Category'].value_counts().to_dict()

    # Create the FICO Plotly scatter plot
    fig_fico = px.scatter(
        df_sampled, 
        x="FICO_V9_SCORE", 
        y="WEIGHTED_ENERGYSCORE", 
        color="Category",
        color_discrete_map={
            'True Positive': 'blue',
            'False Positive': 'lightblue',
            'True Negative': 'green',
            'False Negative': 'lightgreen'
        },
        title=f"FICO Score Decision Outcome (Sampled {sampling_percentage * 100:.0f}%)"
    )

    # Customize the layout
    fig_fico.update_layout(
        xaxis_title="FICO Score",
        yaxis_title="EnergyScore",
        showlegend=True,
        height=600,
        width=900,
    )

    # Plot the FICO scatter plot
    st.plotly_chart(fig_fico)

    # Decision Summary for FICO
    st.write("### FICO Decision Summary")
    st.write(f"**True Positives**: {fico_summary.get('True Positive', 0)}")
    st.write(f"**False Positives**: {fico_summary.get('False Positive', 0)}")
    st.write(f"**True Negatives**: {fico_summary.get('True Negative', 0)}")
    st.write(f"**False Negatives**: {fico_summary.get('False Negative', 0)}")

    # Calculate correct rate for FICO
    correct_rate_fico = ((fico_summary.get('True Positive', 0) + fico_summary.get('True Negative', 0)) / len(df_sampled)) * 100

    # Profit Calculation for FICO
    true_positives_fico = fico_summary.get('True Positive', 0)
    false_positives_fico = fico_summary.get('False Positive', 0)
    profit_fico = (true_positives_fico * true_positive_profit) + (false_positives_fico * false_positive_cost)

    # Calculate other metrics for FICO
    true_positive_rate_fico = (fico_summary.get('True Positive', 0) / len(df_sampled[df_sampled['PAYBACK'] == 0])) * 100
    positive_rate_fico = (fico_summary.get('True Positive', 0) / len(df_sampled)) * 100

    # Display FICO metrics
    st.write(f"**Correct Rate (FICO)**: {correct_rate_fico:.2f}%")
    st.write(f"**Profit (FICO)**: ${profit_fico:,.0f}")
    st.write(f"**True Positive Rate (FICO)**: {true_positive_rate_fico:.2f}%")
    st.write(f"**Positive Rate (FICO)**: {positive_rate_fico:.2f}%")


    # temp_fico_profit = profit

    # profit_pct_increase = ((temp_es_profit - temp_fico_profit) / temp_fico_profit) * 100


    # st.sidebar.metric(label="Profit Percentage Increase: ", value = f"{profit_pct_increase:.2f}%")
        # -------------------- EnergyScore Calculation ---------------------

    # Process the data using EnergyScore (use_es=True)
    df_outcome_es = calculate_loan_outcome(person_data, fico_threshold, energy_score_threshold, use_es=True)
    df_sampled_es = df_outcome_es.sample(frac=sampling_percentage)

    # Count results for EnergyScore
    es_summary = df_sampled_es['Category'].value_counts().to_dict()

    # Calculate profit for EnergyScore
    true_positives_es = es_summary.get('True Positive', 0)
    false_positives_es = es_summary.get('False Positive', 0)

    # Total profit: profit from True Positives + cost from False Positives
    profit_es = (true_positives_es * true_positive_profit) + (false_positives_es * false_positive_cost)

    # Store EnergyScore profit for comparison
    temp_es_profit = profit_es

    # ---------------------- FICO Threshold Calculation --------------------------

    # Process the data using FICO Score (use_es=False)
    df_outcome_fico = calculate_loan_outcome(person_data, fico_threshold, energy_score_threshold, use_es=False)
    df_sampled_fico = df_outcome_fico.sample(frac=sampling_percentage)

    # Count results for FICO Score
    fico_summary = df_sampled_fico['Category'].value_counts().to_dict()

    # Calculate profit for FICO
    true_positives_fico = fico_summary.get('True Positive', 0)
    false_positives_fico = fico_summary.get('False Positive', 0)

    # Total profit: profit from True Positives + cost from False Positives
    profit_fico = (true_positives_fico * true_positive_profit) + (false_positives_fico * false_positive_cost)

    # Store FICO profit for comparison
    temp_fico_profit = profit_fico

    # ------------------- Profit Percentage Increase Calculation -----------------

    # Calculate the profit percentage increase between EnergyScore and FICO
    if temp_fico_profit != 0:
        profit_pct_increase = ((temp_es_profit - temp_fico_profit) / temp_fico_profit) * 100
    else:
        profit_pct_increase = 0  # Avoid division by zero

    # Display Profit Percentage Increase
    st.sidebar.metric(label="Profit Percentage Increase: ", value=f"{profit_pct_increase:.2f}%")

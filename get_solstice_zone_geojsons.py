import geopandas as gpd
import pandas as pd
import json
import os

# Step 1: Load the GeoJSON file containing all US ZIP codes
us_zip_geojson = gpd.read_file('/Users/jakeford/solstice/energyscore-model/us_zips.geojson')  # Adjust the file path as needed
us_zip_geojson['ZIP'] = us_zip_geojson['ZCTA5CE10'].astype(str).str.zfill(5)

# Step 2: Load the CSV file containing filter conditions for ZIP codes
# get this by running following on redash: select * from utility_zones, save locally 
df = pd.read_csv('//Users/jakeford/Downloads/82284_2024_10_16.csv')  # Load your CSV file

# Create a directory to save the output GeoJSON and CSV files
output_dir = 'filtered_geojsons'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def extract_zipcodes(json_string):
    try:
        # Replace single quotes with double quotes to make it valid JSON
        json_string = json_string.replace("'", "\"")
        # Load the string as a dictionary and extract the list
        zipcodes_list = json.loads(json_string)['list']
        return zipcodes_list
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error decoding JSON: {e}")
        return []

# Apply the function to the 'zipcodes' column
df['zipcodes_list'] = df['zipcodes'].apply(extract_zipcodes)

def save_zipcodes_csv(df, output_dir):
    for index, row in df.iterrows():
        # Get the list of zip codes for the current row
        zipcodes_list = row['zipcodes_list']  # Assuming this column now contains the Python list of zip codes
        
        # Create a DataFrame from the list of zip codes
        zip_df = pd.DataFrame(zipcodes_list, columns=['ZIP'])
        
        # Use the 'friendly_name' column for the filename
        friendly_name = row['friendly_name']
        output_filename = os.path.join(output_dir, f"{friendly_name}_zipcodes.csv")
        
        # Save the DataFrame to CSV
        zip_df.to_csv(output_filename, index=False)
        print(f"Saved ZIP codes CSV for {friendly_name} to {output_filename}")

for index, row in df.iterrows():
    # Get the list of zip codes for the current row
    zipcodes_list = row['zipcodes_list']
    
    # Filter the GeoJSON data to include only the relevant ZIP codes
    filtered_zip_geojson = us_zip_geojson[us_zip_geojson['ZIP'].isin(zipcodes_list)]
    
    # Use the 'friendly_name' column for the filename
    friendly_name = row['friendly_name']
    output_filename = os.path.join(output_dir, f"{friendly_name}.geojson")
    
    # Save the filtered GeoJSON file
    filtered_zip_geojson.to_file(output_filename, driver='GeoJSON')
    
    print(f"Saved filtered GeoJSON for {friendly_name} to {output_filename}")

# Call the function to save zip codes to CSV
save_zipcodes_csv(df, output_dir)

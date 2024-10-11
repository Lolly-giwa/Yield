import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Define a function to make predictions
def predict_yield(data):
    # Convert input data into a DataFrame
    input_data = pd.DataFrame([data])
    
    # List of all expected features based on the training data
    expected_features = [
        "Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest",
        "Crop_Barley", "Crop_Cotton", "Crop_Maize", "Crop_Rice", "Crop_Soybean", "Crop_Wheat",
        "Region_East", "Region_North", "Region_South", "Region_West",
        "Weather_Condition_Cloudy", "Weather_Condition_Rainy", "Weather_Condition_Sunny",
        "Soil_Type_Chalky", "Soil_Type_Clay", "Soil_Type_Loam", "Soil_Type_Peaty", "Soil_Type_Sandy", "Soil_Type_Silt",
        "Fertilizer_Used_False", "Fertilizer_Used_True",
        "Irrigation_Used_False", "Irrigation_Used_True"
    ]
    
    # Add any missing columns with a value of 0
    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Reorder columns to match the expected order
    input_data = input_data[expected_features]

    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app layout
st.title("Crop Yield Prediction App")
st.write("Predict crop yield based on input data.")

# Input fields for user input
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=300.0)
temperature = st.number_input("Temperature (Celsius)", min_value=-10.0, max_value=50.0, value=25.0)
days_to_harvest = st.number_input("Days to Harvest", min_value=50, max_value=200, value=100)
fertilizer = st.selectbox("Fertilizer Used", ["True", "False"])
irrigation = st.selectbox("Irrigation Used", ["True", "False"])

# Select the crop type
crop_type = st.selectbox("Select Crop", ["Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"])

# Add checkboxes for regions
region_east = st.checkbox("Region: East")
region_north = st.checkbox("Region: North")
region_south = st.checkbox("Region: South")
region_west = st.checkbox("Region: West")

# Add checkboxes for weather conditions
weather_cloudy = st.checkbox("Weather Condition: Cloudy")
weather_rainy = st.checkbox("Weather Condition: Rainy")
weather_sunny = st.checkbox("Weather Condition: Sunny")

# Add checkboxes for soil types
soil_type_chalky = st.checkbox("Soil Type: Chalky")
soil_type_clay = st.checkbox("Soil Type: Clay")
soil_type_loam = st.checkbox("Soil Type: Loam")
soil_type_peaty = st.checkbox("Soil Type: Peaty")
soil_type_sandy = st.checkbox("Soil Type: Sandy")
soil_type_silt = st.checkbox("Soil Type: Silt")

# Prepare input data for prediction
input_data = {
    "Rainfall_mm": rainfall,
    "Temperature_Celsius": temperature,
    "Days_to_Harvest": days_to_harvest,
    f"Crop_{crop_type}": 1,  # Set the selected crop type to 1
    "Region_East": int(region_east),
    "Region_North": int(region_north),
    "Region_South": int(region_south),
    "Region_West": int(region_west),
    "Weather_Condition_Cloudy": int(weather_cloudy),
    "Weather_Condition_Rainy": int(weather_rainy),
    "Weather_Condition_Sunny": int(weather_sunny),
    "Soil_Type_Chalky": int(soil_type_chalky),
    "Soil_Type_Clay": int(soil_type_clay),
    "Soil_Type_Loam": int(soil_type_loam),
    "Soil_Type_Peaty": int(soil_type_peaty),
    "Soil_Type_Sandy": int(soil_type_sandy),
    "Soil_Type_Silt": int(soil_type_silt),
    "Fertilizer_Used_False": 1 if fertilizer == "False" else 0,
    "Fertilizer_Used_True": 1 if fertilizer == "True" else 0,
    "Irrigation_Used_False": 1 if irrigation == "False" else 0,
    "Irrigation_Used_True": 1 if irrigation == "True" else 0,
}

# Predict button
if st.button("Predict Yield"):
    yield_prediction = predict_yield(input_data)
    st.success(f"Predicted Yield: {yield_prediction:.2f} tons/hectare")

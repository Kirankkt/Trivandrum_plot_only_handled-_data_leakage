import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import category_encoders as ce
import joblib

# Define function to load data
@st.cache
def load_data():
    data = pd.read_csv("standardized_locations_dataset.csv")
    return data

# Add weighted features based on proximity categories
def add_weighted_features(data):
    data['Weighted_Beachfront'] = (data['beach_proximity'] == 'Beachfront').astype(int) * 2.5
    data['Weighted_Seaview'] = (data['beach_proximity'] == 'Sea view').astype(int) * 2.0
    data['Weighted_Lakefront'] = (data['lake_proximity'] == 'Lakefront').astype(int) * 1.8
    data['Weighted_Lakeview'] = (data['lake_proximity'] == 'Lake view').astype(int) * 1.5
    return data

# Define function to compute Mean_Price_per_Cent
def add_location_mean_price(data):
    data['Price_per_cent'] = data['Price'] / data['Area']
    mean_price_per_location = (
        data.groupby("Location")['Price_per_cent']
        .mean()
        .rename("Mean_Price_per_Cent")
        .reset_index()
    )
    data = pd.merge(data, mean_price_per_location, on="Location", how="left")
    return data

# Predict function
def predict_price(model, training_data, area, location, beach_proximity, lake_proximity, density):
    # Target encoding for location
    target_encoder = ce.TargetEncoder(cols=['Location'])
    target_encoder.fit(training_data['Location'], training_data['Price'])

    # Map proximity inputs to weights
    beach_weights = {'Inland': 0, 'Sea view': 2.0, 'Beachfront': 2.5}
    lake_weights = {'Inland': 0, 'Lake view': 1.5, 'Lakefront': 1.8}

    # Calculate Mean_Price_per_Cent dynamically
    price_per_cent_mean = training_data.loc[training_data['Location'] == location, 'Price'].sum() / \
                          training_data.loc[training_data['Location'] == location, 'Area'].sum()
    if np.isnan(price_per_cent_mean):
        price_per_cent_mean = training_data['Price'].sum() / training_data['Area'].sum()

    # Calculate weights
    weighted_beachfront = beach_weights[beach_proximity]
    weighted_seaview = beach_weights[beach_proximity]
    weighted_lakefront = lake_weights[lake_proximity]
    weighted_lakeview = lake_weights[lake_proximity] * 0.75
    area_density = area * (1 if density == 'High' else 0)

    # Encode location
    encoded_location = target_encoder.transform(pd.Series([location]))

    # Create a single input DataFrame
    input_data = pd.DataFrame([{
        'Area': area,
        'Mean_Price_per_Cent': price_per_cent_mean,
        'Weighted_Beachfront': weighted_beachfront,
        'Weighted_Seaview': weighted_seaview,
        'Weighted_Lakefront': weighted_lakefront,
        'Weighted_Lakeview': weighted_lakeview,
        'Area_Density': area_density,
        'density': density,
        'Location': encoded_location.iloc[0]
    }])

    # Predict
    predicted_price = model.predict(input_data)
    return predicted_price[0]

# Streamlit UI
st.title("Real Estate Price Predictor (Advanced)")
st.write("Predict the price of plots based on features like location, proximity to amenities, and area.")

# Load data
data = load_data()
data = add_weighted_features(data)
data = add_location_mean_price(data)

# Load the trained model
model = joblib.load('xgb_price_model.pkl')  # Save this earlier using joblib

# Inputs
area = st.number_input("Enter the area in cents:", min_value=1.0, step=0.1)
location = st.selectbox("Select the location:", options=data['Location'].unique())
beach_proximity = st.selectbox("Select beach proximity:", options=['Inland', 'Sea view', 'Beachfront'])
lake_proximity = st.selectbox("Select lake proximity:", options=['Inland', 'Lake view', 'Lakefront'])
density = st.selectbox("Select density:", options=['Low', 'High'])

if st.button("Predict Price"):
    try:
        predicted_price = predict_price(model, data, area, location, beach_proximity, lake_proximity, density)
        st.success(f"Predicted Price for the plot: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

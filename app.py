import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your dataset for dropdown values
df = pd.read_csv("adverts_preprocessed.csv")

# Define features exactly as in training
numerical_features = ['mileage', 'year_of_registration']
categorical_features = [
    'standard_colour', 'standard_make', 'standard_model',
    'vehicle_condition', 'body_type', 'fuel_type'
]
# Ordering the features
all_features_ordered = ['mileage', 'standard_colour', 'standard_make', 'standard_model', 'vehicle_condition', 'year_of_registration', 'body_type', 'fuel_type']


# Get unique options
colour_options = sorted(df['standard_colour'].dropna().unique())
make_options = sorted(df['standard_make'].dropna().unique())
model_options = sorted(df['standard_model'].dropna().dropna().unique())
condition_options = sorted(df['vehicle_condition'].dropna().unique())
body_type_options = sorted(df['body_type'].dropna().unique())
fuel_type_options = sorted(df['fuel_type'].dropna().unique())

# Streamlit UI
st.title("Car Price Predictor")

# Input widgets
col1, col2 = st.columns(2)
with col1:
    # Full list of makes
    make_options = sorted(df['standard_make'].dropna().unique())
    make = st.selectbox("Make", make_options)

    # Filter the dataframe by selected make
    filtered_df = df[df['standard_make'] == make]

    # Get valid options based on selected make
    model_options = sorted(filtered_df['standard_model'].dropna().unique())
    fuel_type_options = sorted(filtered_df['fuel_type'].dropna().unique())
    body_type_options = sorted(filtered_df['body_type'].dropna().unique())

    # Now show the dependent dropdowns
    model_name = st.selectbox("Model", model_options)
    fuel = st.selectbox("Fuel Type", fuel_type_options)
    body_type = st.selectbox("Body Type", body_type_options)


with col2:
    colour = st.selectbox("Colour", colour_options)
    condition = st.selectbox("Condition", condition_options)

    # Set max mileage and year of registration based on vehicle condition
    if condition == 'NEW':
        max_mileage = 50
        reg_year = 2021
        mileage = st.number_input("mileage", min_value=0, max_value=max_mileage)
        st.number_input("year_of_registration", min_value=2021, max_value=2021, value=reg_year, disabled=True)

    else:
        max_mileage = 200000
        mileage = st.number_input("mileage", min_value=0, max_value=max_mileage)
        reg_year = st.number_input("year_of_registration", min_value=2000, max_value=2021)


# Load artifacts
try:
    # Load all model components
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('best_xgb_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    price_scaler = joblib.load('price_scaler.pkl')

except Exception as e:
    st.error(f"Error loading model artifacts: {str(e)}")
    st.stop()


user_input = pd.DataFrame([{
    'standard_colour': colour,
    'standard_make': make,
    'standard_model': model_name,
    'vehicle_condition': condition,
    'body_type': body_type,
    'fuel_type': fuel,
    'mileage': mileage,
    'year_of_registration': reg_year
}])

# Apply same preprocessing as in training
try:
    # Apply label encoding to categorical features
    for col in categorical_features:
        # Handle unseen categories by mapping to a default value
        if user_input[col].iloc[0] not in label_encoders[col].classes_:
            st.warning(f"⚠️ '{user_input[col].iloc[0]}' not seen in training for {col}, using default encoding")
            user_input[col] = -1  # Special value for unseen categories
        else:
            user_input[col] = label_encoders[col].transform(user_input[col])

    # Scale numerical features
    user_input_num = user_input[numerical_features]
    user_input_num_scaled = scaler.transform(user_input_num)
    user_input[numerical_features] = user_input_num_scaled


    # Ensure correct column order
    final_input = user_input[all_features_ordered]


except Exception as e:
    st.error(f"Preprocessing error: {str(e)}")
    st.stop()

# Predict button
if st.button("Predict Price", type="primary"):
    with st.spinner("Calculating fair price..."):
        try:
            # Predict scaled price
            prediction = model.predict(final_input)

            # ⬇️ Inverse transform to get real-world price
            y_pred_price = price_scaler.inverse_transform(prediction.reshape(-1, 1)).ravel()


            # Show result
            st.balloons()
            st.success(f"## Estimated Price: £{y_pred_price[0]:,.2f}")
            st.info("💡 This prediction is based on similar vehicles in our database")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

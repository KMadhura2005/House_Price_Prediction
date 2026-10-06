import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

location_columns = [c for c in model_columns if c.startswith("location_")]
area_columns = [c for c in model_columns if c.startswith("area_type")]
availability_columns = [c for c in model_columns if c.startswith("availability_")]

st.title("🏠 House Price Prediction")
st.write("Enter the property details below to estimate its price.")

st.sidebar.header("Project Information")
st.sidebar.write("Machine Learning Regression Project")
st.sidebar.write("Target: House Price")
st.sidebar.write("Price unit: Lakhs")

col1, col2, col3 = st.columns(3)

with col1:
    total_sqft = st.number_input(
        "Total Area (sqft)",
        min_value=300.0,
        max_value=20000.0,
        value=1200.0,
        step=50.0
    )
    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

with col2:
    bath = st.number_input(
        "Bathrooms",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=1.0
    )
    balcony = st.number_input(
        "Balconies",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=1.0
    )

with col3:
    location_display = st.selectbox(
        "Location",
        [c.replace("location_", "", 1) for c in location_columns]
    )
    area_display = st.selectbox(
        "Area Type",
        [c.replace("area_type", "", 1).strip() for c in area_columns]
    )

availability_display = st.selectbox(
    "Availability",
    ["Ready To Move"]
)

if st.button("Predict House Price", type="primary"):
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)

    input_data["bath"] = bath
    input_data["balcony"] = balcony
    input_data["total_sqft_int"] = total_sqft
    input_data["bhk"] = bhk

    location_col = "location_" + location_display
    if location_col in input_data.columns:
        input_data[location_col] = 1

    # Area type column names contain the original double-space formatting.
    for c in area_columns:
        input_data[c] = 0

    selected_area_col = next(
        (c for c in area_columns
         if c.replace("area_type", "", 1).strip() == area_display),
        None
    )
    if selected_area_col:
        input_data[selected_area_col] = 1

    for c in availability_columns:
        input_data[c] = 0

    ready_col = "availability_Ready To Move"
    if ready_col in input_data.columns:
        input_data[ready_col] = 1

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ₹ {prediction:.2f} Lakhs")
    st.info(f"Approximate value: ₹ {prediction * 100000:,.0f}")

st.markdown("---")
st.caption(
    "Note: This model is an educational prediction system. "
    "Actual property prices can vary based on market conditions and other factors."
)

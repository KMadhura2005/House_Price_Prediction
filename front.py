import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HomeValue AI",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

/* Main page */
.stApp {
    background-color: #f5f7fb;
}

/* Hero box */
.hero-box {
    background: linear-gradient(
        135deg,
        #3730a3,
        #6366f1
    );

    padding: 38px 42px;
    border-radius: 24px;
    margin-bottom: 35px;

    box-shadow: 0 12px 30px rgba(79, 70, 229, 0.25);
}

/* Hero title */
.hero-title {
    color: white !important;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 12px;
}

/* Hero subtitle */
.hero-subtitle {
    color: #eef2ff !important;
    font-size: 17px;
    line-height: 1.6;
    margin: 0;
}

/* Section headings */
h1, h2, h3 {
    color: #172033;
}

/* Prediction result */
div[data-testid="stMetric"] {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

div[data-testid="stMetricValue"] {
    color: #4f46e5;
    font-size: 36px;
    font-weight: 800;
}

div[data-testid="stMetricLabel"] {
    color: #64748b;
    font-size: 15px;
    font-weight: 600;
}

/* Predict button */
.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;

    background-color: #4f46e5;
    color: white;

    font-size: 17px;
    font-weight: 700;

    border: none;
}

.stButton > button:hover {
    background-color: #3730a3;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: white !important;
}

/* Input labels */
label {
    font-weight: 600 !important;
    color: #334155 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🏠 HomeValue AI")

    st.caption("Smart House Price Prediction")

    st.divider()

    st.subheader("📌 About")

    st.write(
        "This application uses Machine Learning "
        "to estimate the price of a house based "
        "on its property details."
    )

    st.subheader("🤖 Model")

    st.write("Random Forest Regressor")

    st.subheader("📊 Problem Type")

    st.write("Regression")

    st.subheader("🎯 Target")

    st.write("House Price")

    st.divider()

    st.caption(
        "Built with Python • Scikit-learn • Streamlit"
    )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    "# 🏠 Find Your Home's Estimated Value"
)

st.write(
    "Enter the property details below and let our "
    "Machine Learning model estimate the house price."
)

st.divider()
# =========================================================
# GET MODEL COLUMNS
# =========================================================

location_columns = [
    col for col in model_columns
    if col.startswith("location_")
]

area_columns = [
    col for col in model_columns
    if col.startswith("area_type")
]

availability_columns = [
    col for col in model_columns
    if col.startswith("availability_")
]


# =========================================================
# PROPERTY DETAILS
# =========================================================

st.header("🏡 Property Details")

st.write(
    "Provide the basic information about the property."
)


# =========================================================
# INPUTS
# =========================================================

col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# PROPERTY SIZE
# ---------------------------------------------------------

with col1:

    st.subheader("📐 Property Size")

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


# ---------------------------------------------------------
# FACILITIES
# ---------------------------------------------------------

with col2:

    st.subheader("🚿 Facilities")

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


# ---------------------------------------------------------
# LOCATION
# ---------------------------------------------------------

with col3:

    st.subheader("📍 Location")

    location = st.selectbox(
        "Location",
        [
            col.replace("location_", "")
            for col in location_columns
        ]
    )

    area_type = st.selectbox(
        "Area Type",
        [
            col.replace("area_type", "").strip()
            for col in area_columns
        ]
    )


# =========================================================
# AVAILABILITY
# =========================================================

availability = st.selectbox(
    "🏗️ Availability",
    [
        col.replace("availability_", "")
        for col in availability_columns
    ]
)


st.write("")


# =========================================================
# PREDICT BUTTON
# =========================================================

predict = st.button(
    "🔮 Predict House Price"
)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model_columns
    )


    # -----------------------------------------------------
    # NUMERICAL VALUES
    # -----------------------------------------------------

    if "total_sqft_int" in input_data.columns:
        input_data["total_sqft_int"] = total_sqft

    if "bhk" in input_data.columns:
        input_data["bhk"] = bhk

    if "bath" in input_data.columns:
        input_data["bath"] = bath

    if "balcony" in input_data.columns:
        input_data["balcony"] = balcony


    # -----------------------------------------------------
    # LOCATION
    # -----------------------------------------------------

    location_column = "location_" + location

    if location_column in input_data.columns:
        input_data[location_column] = 1


    # -----------------------------------------------------
    # AREA TYPE
    # -----------------------------------------------------

    selected_area_column = None

    for col in area_columns:

        if col.replace(
            "area_type", ""
        ).strip() == area_type:

            selected_area_column = col

            break


    if selected_area_column:

        input_data[selected_area_column] = 1


    # -----------------------------------------------------
    # AVAILABILITY
    # -----------------------------------------------------

    availability_column = (
        "availability_" + availability
    )

    if availability_column in input_data.columns:

        input_data[availability_column] = 1


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # =====================================================
    # RESULT
    # =====================================================

    st.write("")

    st.header("💰 Estimated Property Value")

    st.metric(
        label="Predicted House Price",
        value=f"₹ {prediction:.2f} Lakhs"
    )

    st.success(
        f"Approximately ₹ {prediction * 100000:,.0f}"
    )


    # =====================================================
    # PROPERTY SUMMARY
    # =====================================================

    st.write("")

    st.header("📋 Property Summary")

    st.write(
        "Details used by the model for this prediction."
    )


    s1, s2, s3, s4 = st.columns(4)


    with s1:

        st.metric(
            "📐 Area",
            f"{total_sqft:,.0f} sqft"
        )


    with s2:

        st.metric(
            "🛏️ Bedrooms",
            f"{bhk} BHK"
        )


    with s3:

        st.metric(
            "🚿 Bathrooms",
            f"{bath:.0f}"
        )


    with s4:

        st.metric(
            "📍 Location",
            location
        )


    # =====================================================
    # ADDITIONAL DETAILS
    # =====================================================

    st.write("")

    detail1, detail2 = st.columns(2)

    with detail1:

        st.info(
            f"🏠 **Area Type:** {area_type}"
        )

    with detail2:

        st.info(
            f"🏗️ **Availability:** {availability}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🏠 HomeValue AI • House Price Prediction using Machine Learning"
)
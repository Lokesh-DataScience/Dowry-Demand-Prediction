import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Change if deployed elsewhere

# Fetch options from the API
@st.cache_data
def fetch_options():
    response = requests.get(f"{API_URL}/options")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch input options from API.")
        return {}

def predict_dowry(data):
    response = requests.post(f"{API_URL}/predict", json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Prediction failed: {response.json().get('detail')}")
        return None

st.title("💍 Dowry Prediction Web App")
st.markdown("Use the form below to input details and get dowry prediction.")

GITHUB_URL = "https://github.com/Lokesh-DataScience/Dowry-Demand-Prediction.git"  # <-- Replace with your repo link
st.markdown(
    f'<a href="{GITHUB_URL}" target="_blank"><button style="background-color:#24292F;color:white;padding:8px 16px;border:none;border-radius:5px;cursor:pointer;">View Source Code on GitHub</button></a>',
    unsafe_allow_html=True
)

# Load dropdown options
options = fetch_options()

# --- Input Fields ---
year = st.number_input("Year of Marriage", min_value=2000, max_value=2030, value=2023)
womens_age = st.slider("Woman's Age", min_value=15, max_value=50, value=22)
mens_age = st.slider("Man's Age", min_value=18, max_value=60, value=28)
mohor_amount = st.number_input("Mohor Amount (Taka)", min_value=0.0, value=10000.0)

family_type = st.selectbox("Family Type", options.get("family_types", []))
area = st.selectbox("Area", options.get("areas", []))
girls_job = st.selectbox("Girl's Job", options.get("girls_jobs", []))
boys_job = st.selectbox("Boy's Job", options.get("boys_jobs", []))
marriage_type = st.selectbox("Marriage Type", options.get("marriage_types", []))
womens_marital_status = st.selectbox("Woman's Marital Status", options.get("marital_statuses", []))

# Submit Button
if st.button("Predict Dowry"):
    input_payload = {
        "year": year,
        "womens_age": womens_age,
        "mens_age": mens_age,
        "mohor_amount": mohor_amount,
        "family_type": family_type,
        "area": area,
        "girls_job": girls_job,
        "boys_job": boys_job,
        "marriage_type": marriage_type,
        "womens_marital_status": womens_marital_status
    }

    result = predict_dowry(input_payload)
    if result:
        st.success(f"### 🎯 Predicted Dowry Category: `{result['predicted_dowry']}`")
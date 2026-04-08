import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model-2.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Hospital AI Dashboard",
    layout="wide"
)

# Custom CSS (Hospital Theme)
st.markdown("""
    <style>
        .main {
            background-color: #f4f8fb;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #0a3d62;
        }
        .subtitle {
            font-size: 18px;
            color: #3c6382;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🧠 Alzheimer Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered clinical decision support</div>', unsafe_allow_html=True)

st.write("---")

# Sidebar (Patient Form)
st.sidebar.header("🧾 Patient Information")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age", 50, 100, 70)
educ = st.sidebar.slider("Education Level", 1, 5, 3)
ses = st.sidebar.slider("SES (1=High, 5=Low)", 1, 5, 3)
mmse = st.sidebar.slider("MMSE Score", 0, 30, 25)
etiv = st.sidebar.number_input("eTIV", 1000, 2000, 1400)
nwbv = st.sidebar.slider("nWBV", 0.5, 1.0, 0.7)
asf = st.sidebar.slider("ASF", 0.8, 1.5, 1.1)

gender_val = 1 if gender == "Male" else 0

input_data = pd.DataFrame([[gender_val, age, educ, ses, mmse, etiv, nwbv, asf]],
                          columns=features)

# Layout columns
col1, col2 = st.columns([2, 1])

# Patient Summary Card
with col1:
    st.markdown("### 📋 Patient Summary")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write(f"**Gender:** {gender}")
    st.write(f"**Age:** {age}")
    st.write(f"**MMSE Score:** {mmse}")
    st.write(f"**Brain Volume (nWBV):** {nwbv}")

    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Panel
with col2:
    st.markdown("### 🔍 Prediction")

    if st.button("Run Diagnosis"):

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Risk Level
        if probability > 0.6:
            risk = "HIGH"
            color = "red"
        elif probability > 0.3:
            risk = "MEDIUM"
            color = "orange"
        else:
            risk = "LOW"
            color = "green"

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if prediction == 1:
            st.error("⚠️ Alzheimer Detected")
        else:
            st.success("✅ No Alzheimer Detected")

        st.write(f"### Probability: {probability*100:.2f}%")
        st.markdown(f"### Risk Level: :{color}[{risk}]")

        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.write("---")
st.caption("⚠️ This tool is for educational purposes only. Consult a medical professional.")

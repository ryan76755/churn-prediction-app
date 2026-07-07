import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Prediksi Customer Churn", page_icon="📡", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, encoders, feature_cols

model, scaler, encoders, feature_cols = load_artifacts()

st.title("📡 Sistem Prediksi Customer Churn")
st.markdown("""
Aplikasi ini memprediksi kemungkinan seorang pelanggan **berhenti berlangganan (churn)**
berdasarkan profil dan penggunaan layanannya, menggunakan model **Random Forest**
yang dilatih pada data historis pelanggan telekomunikasi.
""")

st.divider()
st.subheader("Masukkan Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Punya Partner?", ["Yes", "No"])
    dependents = st.selectbox("Punya Tanggungan?", ["Yes", "No"])
    tenure = st.slider("Lama Berlangganan (bulan)", 0, 72, 12)
    phone_service = st.selectbox("Layanan Telepon", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Layanan Internet", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Metode Pembayaran", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

if st.button("🔍 Prediksi Churn", type="primary", use_container_width=True):
    raw = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([raw])
    for col, le in encoders.items():
        if col in input_df.columns:
            # handle unseen categories safely
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])

    input_df = input_df[feature_cols]

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Pelanggan berisiko **CHURN** — probabilitas: {proba:.1%}")
        st.markdown("**Rekomendasi:** prioritaskan pelanggan ini untuk kampanye retensi (penawaran diskon, upgrade layanan, atau kontrak jangka panjang).")
    else:
        st.success(f"✅ Pelanggan diprediksi **tetap berlangganan** — probabilitas churn: {proba:.1%}")

    st.progress(float(proba))

st.divider()
with st.expander("ℹ️ Tentang Model & Etika Penggunaan"):
    st.markdown("""
    - **Model:** Random Forest Classifier (ROC-AUC ±0.84 pada data uji)
    - **Fitur paling berpengaruh:** jenis kontrak, lama berlangganan, total biaya, biaya bulanan, layanan keamanan online
    - **Catatan etis:** hasil prediksi ini adalah alat bantu pengambilan keputusan, bukan keputusan final.
      Gunakan untuk menawarkan nilai tambah ke pelanggan berisiko, bukan untuk tindakan yang merugikan
      pelanggan secara sepihak. Model perlu dievaluasi ulang secara berkala agar tetap representatif.
    """)

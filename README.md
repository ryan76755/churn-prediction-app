# Sistem Prediksi Customer Churn

Aplikasi web deployment model machine learning untuk memprediksi churn pelanggan telekomunikasi.

## Isi Repository
- `Churn_Prediction_Project.ipynb` — notebook analisis, pelatihan, dan evaluasi model
- `app.py` — aplikasi web (Streamlit) untuk prediksi interaktif
- `train_model.py` — script pelatihan model
- `churn_model.pkl`, `scaler.pkl`, `encoders.pkl`, `feature_columns.pkl` — artefak model terlatih
- `telco_churn.csv` — dataset yang digunakan
- `requirements.txt` — daftar dependensi

## Cara Menjalankan Aplikasi Secara Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
Aplikasi akan terbuka di `http://localhost:8501`.

## Cara Deploy ke Streamlit Community Cloud (gratis, untuk mendapatkan link publik)
1. Buat repository baru di GitHub, upload seluruh isi folder ini (termasuk file `.pkl`).
2. Buka [share.streamlit.io](https://share.streamlit.io), login dengan akun GitHub.
3. Klik **New app**, pilih repository dan branch, set **Main file path** ke `app.py`.
4. Klik **Deploy**. Setelah proses build selesai, aplikasi akan memiliki link publik
   berformat `https://<nama-app>.streamlit.app` yang bisa dicantumkan di laporan.

## Ringkasan Model
- **Algoritma:** Random Forest Classifier (Supervised Learning — Klasifikasi Biner)
- **Performa:** Accuracy 76.6%, F1-Score 0.635, ROC-AUC 0.842
- **Fitur paling berpengaruh:** Contract, tenure, TotalCharges, MonthlyCharges, OnlineSecurity

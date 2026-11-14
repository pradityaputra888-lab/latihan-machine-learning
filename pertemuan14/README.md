# Prediksi Churn Pelanggan Telco: paket deploy Streamlit

Isi folder ini harus diunggah **semuanya** ke root repositori GitHub:

| File | Fungsi |
|---|---|
| `app.py` | Aplikasi Streamlit (file utama) |
| `fitur_telco.py` | Modul preprocessing yang dipakai notebook dan aplikasi |
| `model_terbaik.pkl` | Pipeline model terlatih (joblib) |
| `model_meta.json` | Ambang keputusan, opsi form, metrik, versi library |
| `feature_importance.csv` | Fitur terpenting untuk tab "Tentang model" |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset (dipakai bila model perlu dilatih ulang otomatis) |
| `requirements.txt` | Daftar library |

## Uji lokal
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Unggah semua file di atas ke repositori GitHub (root, bukan subfolder).
2. Buka share.streamlit.io, pilih Create app, pilih repositori dan cabang `main`, Main file path: `app.py`.
3. Di Advanced settings pilih Python 3.12, lalu Deploy.
4. Jika build gagal karena versi `scikit-learn` di `requirements.txt` tidak tersedia, hapus angka versinya
   (tinggalkan `scikit-learn`). Aplikasi mendeteksi perbedaan versi dan melatih ulang model otomatis
   dengan hyperparameter yang sama.

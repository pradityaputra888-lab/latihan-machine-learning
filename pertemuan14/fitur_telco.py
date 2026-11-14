"""
fitur_telco.py
Modul pendukung Mini Project Prediksi Churn Pelanggan Telco.
Dipakai oleh NOTEBOOK (saat melatih) dan oleh APLIKASI STREAMLIT (saat memprediksi),
sehingga pipeline preprocessing identik di kedua tempat (prinsip Bab 4 dan Bab 14.4).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

LAYANAN_TAMBAHAN = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]

# 18 kolom mentah yang diminta dari pengguna (TotalCharges dan customerID sengaja tidak dipakai)
FITUR_MENTAH = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                "PhoneService", "MultipleLines", "InternetService", *LAYANAN_TAMBAHAN,
                "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges"]

FITUR_NUMERIK = ["tenure", "MonthlyCharges", "jumlah_layanan_tambahan"]   # di-scaling
FITUR_BINER = ["SeniorCitizen"]                                            # sudah 0/1
FITUR_KATEGORIKAL = [c for c in FITUR_MENTAH
                     if c not in ("SeniorCitizen", "tenure", "MonthlyCharges")]


def tambah_fitur(X):
    """Feature engineering: jumlah layanan tambahan yang dilanggan (0-6)."""
    X = X.copy()
    X["jumlah_layanan_tambahan"] = (X[LAYANAN_TAMBAHAN] == "Yes").sum(axis=1)
    return X


def bangun_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), FITUR_NUMERIK),
        ("bin", "passthrough", FITUR_BINER),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), FITUR_KATEGORIKAL),
    ])


def bangun_pipeline(estimator):
    """Pipeline lengkap: rekayasa fitur -> preprocessing -> model."""
    return Pipeline([
        ("fitur", FunctionTransformer(tambah_fitur, validate=False)),
        ("prep", bangun_preprocessor()),
        ("model", estimator),
    ])


# Dipakai aplikasi untuk melatih ulang bila versi scikit-learn di server berbeda
REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    "SVC": SVC,
}
KONFIG_TETAP = {
    "LogisticRegression": dict(max_iter=2000),
    "RandomForestClassifier": dict(random_state=42),
    "HistGradientBoostingClassifier": dict(random_state=42),
    "SVC": dict(kernel="rbf"),
}


def buat_estimator(nama, params):
    return REGISTRY[nama](**{**KONFIG_TETAP.get(nama, {}), **params})

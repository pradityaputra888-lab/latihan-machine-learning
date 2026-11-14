"""Aplikasi Streamlit - Prediksi Risiko Churn Pelanggan Telco (Mini Project ML, Bab 14)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.model_selection import train_test_split

import fitur_telco as ft  # WAJIB diimpor sebelum joblib.load (pipeline memakai fungsi dari modul ini)

BASE = Path(__file__).parent
NAMA_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

st.set_page_config(page_title="Prediksi Churn Pelanggan Telco", page_icon="📞", layout="wide")


# ---------------------------------------------------------------- memuat model
@st.cache_resource(show_spinner="Memuat model...")
def muat_model():
    """Memuat model tersimpan. Jika versi scikit-learn berbeda / file rusak, model dilatih ulang
    dari dataset dengan algoritma dan hyperparameter yang sama (tersimpan di model_meta.json)."""
    meta = json.loads((BASE / "model_meta.json").read_text(encoding="utf-8"))
    catatan = None
    file_model = BASE / "model_terbaik.pkl"
    if meta["versi"]["scikit-learn"] == sklearn.__version__ and file_model.exists():
        try:
            return joblib.load(file_model), meta, None
        except Exception as e:  # noqa: BLE001
            catatan = f"Model tersimpan gagal dimuat ({type(e).__name__}); model dilatih ulang."
    else:
        catatan = (f"Versi scikit-learn server ({sklearn.__version__}) berbeda dari saat pelatihan "
                   f"({meta['versi']['scikit-learn']}); model dilatih ulang agar kompatibel.")
    df = pd.read_csv(BASE / NAMA_CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    est = ft.buat_estimator(meta["estimator"]["nama"], meta["estimator"]["params"])
    model = ft.bangun_pipeline(est).fit(X_tr[ft.FITUR_MENTAH], y_tr)
    return model, meta, catatan


model, meta, catatan_model = muat_model()
OPSI = meta["opsi"]
ASUMSI = meta["asumsi_bisnis"]
THR = meta["ambang_keputusan"]

# ------------------------------------------------------------------ nilai awal
DEFAULT = dict(gender="Female", SeniorCitizen="Tidak", Partner="No", Dependents="No", tenure=12,
               PhoneService="Yes", MultipleLines="No", InternetService="DSL",
               OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
               StreamingTV="No", StreamingMovies="No", Contract="Month-to-month",
               PaperlessBilling="Yes", PaymentMethod="Electronic check", MonthlyCharges=60.0)
PRESET_BERISIKO = dict(gender="Female", SeniorCitizen="Ya", Partner="No", Dependents="No", tenure=2,
                       PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic",
                       OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
                       StreamingTV="No", StreamingMovies="No", Contract="Month-to-month",
                       PaperlessBilling="Yes", PaymentMethod="Electronic check", MonthlyCharges=80.0)
PRESET_LOYAL = dict(gender="Male", SeniorCitizen="Tidak", Partner="Yes", Dependents="Yes", tenure=60,
                    PhoneService="Yes", MultipleLines="Yes", InternetService="DSL",
                    OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes",
                    StreamingTV="No", StreamingMovies="No", Contract="Two year",
                    PaperlessBilling="No", PaymentMethod="Credit card (automatic)", MonthlyCharges=65.0)

for k, v in DEFAULT.items():
    st.session_state.setdefault(f"k_{k}", v)


def terapkan(preset):
    for k, v in preset.items():
        st.session_state[f"k_{k}"] = v


# Ketergantungan antar-kolom (harus dijalankan SEBELUM widget dibuat)
if st.session_state["k_InternetService"] == "No":
    for c in ft.LAYANAN_TAMBAHAN:
        st.session_state[f"k_{c}"] = "No internet service"
else:
    for c in ft.LAYANAN_TAMBAHAN:
        if st.session_state[f"k_{c}"] == "No internet service":
            st.session_state[f"k_{c}"] = "No"
if st.session_state["k_PhoneService"] == "No":
    st.session_state["k_MultipleLines"] = "No phone service"
elif st.session_state["k_MultipleLines"] == "No phone service":
    st.session_state["k_MultipleLines"] = "No"

NAMA_RAMAH = {
    "tenure": "Lama berlangganan", "MonthlyCharges": "Tagihan bulanan",
    "jumlah_layanan_tambahan": "Jumlah layanan tambahan", "SeniorCitizen": "Pelanggan lansia",
}


def nama_fitur(n):
    n = n.split("__", 1)[1]
    if n in NAMA_RAMAH:
        return NAMA_RAMAH[n]
    for c in ft.FITUR_KATEGORIKAL:
        if n.startswith(c + "_"):
            return f"{c} = {n[len(c) + 1:]}"
    return n


def kontribusi_lokal(x):
    """Kontribusi tiap fitur pada log-odds (hanya untuk model linear)."""
    prep, est = model.named_steps["prep"], model.named_steps["model"]
    xt = prep.transform(model.named_steps["fitur"].transform(x))
    xt = xt.toarray() if hasattr(xt, "toarray") else np.asarray(xt)
    nama = [nama_fitur(n) for n in prep.get_feature_names_out()]
    return pd.Series(est.coef_[0] * xt[0], index=nama)


# ------------------------------------------------------------------- tampilan
st.title("📞 Prediksi Risiko Churn Pelanggan Telco")
st.caption("Mini Project Machine Learning - Bab 14 | Model: "
           f"**{meta['model_terpilih']}** | Dataset: Telco Customer Churn (IBM sample)")
if catatan_model:
    st.info(catatan_model)

with st.sidebar:
    st.header("Contoh cepat")
    st.button("Pelanggan berisiko tinggi", on_click=terapkan, args=(PRESET_BERISIKO,), use_container_width=True)
    st.button("Pelanggan loyal", on_click=terapkan, args=(PRESET_LOYAL,), use_container_width=True)
    st.button("Reset", on_click=terapkan, args=(DEFAULT,), use_container_width=True)
    st.divider()
    st.caption("Aplikasi ini alat bantu analitik untuk memprioritaskan program retensi. "
               "Hasilnya adalah estimasi statistik, bukan kepastian perilaku pelanggan.")

tab1, tab2 = st.tabs(["Prediksi", "Tentang model & data"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Profil pelanggan")
        st.selectbox("Jenis kelamin", OPSI["gender"], key="k_gender")
        st.selectbox("Pelanggan lansia", ["Tidak", "Ya"], key="k_SeniorCitizen")
        st.selectbox("Memiliki pasangan", OPSI["Partner"], key="k_Partner")
        st.selectbox("Memiliki tanggungan", OPSI["Dependents"], key="k_Dependents")
        st.slider("Lama berlangganan (bulan)", 0, 72, key="k_tenure")
    with c2:
        st.subheader("Layanan")
        st.selectbox("Layanan telepon", OPSI["PhoneService"], key="k_PhoneService")
        telp_mati = st.session_state["k_PhoneService"] == "No"
        st.selectbox("Banyak saluran telepon", ["No phone service"] if telp_mati else ["No", "Yes"],
                     key="k_MultipleLines", disabled=telp_mati)
        st.selectbox("Layanan internet", OPSI["InternetService"], key="k_InternetService")
        net_mati = st.session_state["k_InternetService"] == "No"
        for c in ft.LAYANAN_TAMBAHAN:
            st.selectbox(c, ["No internet service"] if net_mati else ["No", "Yes"],
                         key=f"k_{c}", disabled=net_mati)
    with c3:
        st.subheader("Kontrak & tagihan")
        st.selectbox("Jenis kontrak", OPSI["Contract"], key="k_Contract")
        st.selectbox("Tagihan tanpa kertas", OPSI["PaperlessBilling"], key="k_PaperlessBilling")
        st.selectbox("Metode pembayaran", OPSI["PaymentMethod"], key="k_PaymentMethod")
        st.slider("Tagihan bulanan (USD)", 15.0, 125.0, step=0.5, key="k_MonthlyCharges")

    if st.button("Prediksi risiko churn", type="primary"):
        nilai = {c: st.session_state[f"k_{c}"] for c in ft.FITUR_MENTAH}
        nilai["SeniorCitizen"] = 1 if nilai["SeniorCitizen"] == "Ya" else 0
        x = pd.DataFrame([nilai])[ft.FITUR_MENTAH]
        skor = float(model.predict_proba(x)[0, 1])

        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.metric("Skor risiko churn", f"{skor:.1%}")
        k2.metric("Ambang keputusan model", f"{THR:.1%}")
        manfaat = skor * ASUMSI["rasio_berhasil"] * ASUMSI["nilai_pelanggan"] - ASUMSI["biaya_tawaran"]
        k3.metric("Estimasi manfaat bersih penawaran retensi", f"${manfaat:,.0f}")
        st.progress(min(max(skor, 0.0), 1.0))

        if skor >= THR:
            st.error("**Risiko TINGGI.** Disarankan masuk program retensi (mis. penawaran kontrak "
                     "1-2 tahun, diskon tagihan, atau paket layanan tambahan).")
        elif skor >= THR / 2:
            st.warning("**Risiko SEDANG.** Pantau pelanggan ini dan pertimbangkan komunikasi ringan.")
        else:
            st.success("**Risiko RENDAH.** Tidak perlu intervensi khusus.")

        if hasattr(model.named_steps["model"], "coef_"):
            kontrib = kontribusi_lokal(x).sort_values()
            tampil = pd.concat([kontrib.head(4), kontrib.tail(4)]).drop_duplicates()
            st.subheader("Faktor yang memengaruhi skor pelanggan ini")
            st.bar_chart(tampil.rename("Pengaruh (log-odds)"), horizontal=True)
            st.caption("Batang positif menaikkan risiko churn; batang negatif menurunkannya.")

with tab2:
    st.subheader("Kinerja model pada data uji")
    st.dataframe(pd.DataFrame([meta["metrik_uji"]]).T.rename(columns={0: "Nilai"}).round(3))
    st.subheader("Perbandingan model")
    st.dataframe(pd.DataFrame(meta["perbandingan"]).round(3), hide_index=True)
    fi_path = BASE / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path).head(10)
        st.subheader("Fitur terpenting (permutation importance, PR-AUC)")
        st.bar_chart(fi.set_index("fitur")["penting"], horizontal=True)
    st.subheader("Asumsi bisnis (ilustratif)")
    st.write(f"Nilai pelanggan yang berhasil dipertahankan **${ASUMSI['nilai_pelanggan']}**, biaya penawaran "
             f"**${ASUMSI['biaya_tawaran']}** per pelanggan, peluang penawaran berhasil "
             f"**{ASUMSI['rasio_berhasil']:.0%}**. Angka ini asumsi dan harus diganti dengan data perusahaan.")
    st.subheader("Keterbatasan")
    st.markdown("- Data berasal dari dataset contoh (fiktif) IBM; kinerja pada perusahaan nyata bisa berbeda.\n"
                "- Model hanya menangkap pola historis dan tidak membuktikan hubungan sebab-akibat.\n"
                "- Data tidak memuat faktor eksternal (kompetitor, harga pasar, keluhan pelanggan).")

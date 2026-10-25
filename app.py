import re
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Airline Sentiment Analyzer",
    page_icon="✈️",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

@st.cache_resource
def load_resources():
    bundle = joblib.load(ROOT / "sentiment_artifacts" / "sentiment_model.joblib")
    with open(ROOT / "sentiment_artifacts" / "metadata.json", "r") as f:
        metadata = json.load(f)
    return bundle, metadata

bundle, metadata = load_resources()
pipeline = bundle["pipeline"]
classes = bundle["classes"]
top_words = bundle["top_words_per_class"]

def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Header
st.title("✈️ Airline Sentiment Analyzer (NLP)")
st.markdown(
    "Aplikasi deteksi sentimen opini pelanggan media sosial secara otomatis (**Negative**, **Neutral**, **Positive**) "
    "menggunakan representasi **TF-IDF (Unigram + Bigram)** dan model **Logistic Regression (Class-Weighted)**."
)

# Metric Badges
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Test Macro F1-Score", f"{metadata['test_macro_f1']:.4f}")
with m2:
    st.metric("Test Accuracy", f"{metadata['test_accuracy'] * 100:.2f}%")
with m3:
    st.metric("F1-Score (Negative)", "0.8242")
with m4:
    st.metric("Representasi Fitur", "TF-IDF (5.000 n-grams)")

st.markdown("---")

# Layout: Input on left, Model Explainability on right
col_input, col_info = st.columns([1.5, 1])

# Presets
presets = {
    "Pilih contoh ulasan...": "",
    "🟢 Positif: Pelayanan ramah & tepat waktu": "@VirginAmerica Thank you so much! Flight was right on time and the cabin crew was incredibly friendly and helpful.",
    "🔴 Negatif: Delay lama & bagasi hilang": "@AmericanAir Worst experience ever! My flight was delayed for 6 hours and my luggage is completely lost at the airport.",
    "🟡 Netral: Pertanyaan jadwal penerbangan": "@SouthwestAir Can you please let me know what time the boarding gate closes for flight SW104 tomorrow?"
}

with col_input:
    st.subheader("📝 Masukkan Teks Ulasan / Tweet")
    
    selected_preset = st.selectbox("Atau pilih contoh teks siap uji:", list(presets.keys()))
    
    initial_text = presets[selected_preset] if selected_preset != "Pilih contoh ulasan..." else ""
    
    user_text = st.text_area(
        "Ketik teks ulasan penerbangan di sini:",
        value=initial_text,
        height=130,
        placeholder="Contoh: The flight was delayed and customer service was unresponsive..."
    )
    
    predict_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)

with col_info:
    st.subheader("💡 Kata Kunci Indikator Utama")
    st.caption("Kata-kata dengan bobot koefisien tertinggi yang dipelajari model:")
    
    tab_neg, tab_neu, tab_pos = st.tabs(["🔴 Negatif", "🟡 Netral", "🟢 Positif"])
    with tab_neg:
        st.write(", ".join([f"`{w}`" for w in top_words["negative"][:12]]))
    with tab_neu:
        st.write(", ".join([f"`{w}`" for w in top_words["neutral"][:12]]))
    with tab_pos:
        st.write(", ".join([f"`{w}`" for w in top_words["positive"][:12]]))

# Inference Output
if user_text.strip():
    cleaned = clean_tweet(user_text)
    
    if not cleaned:
        st.warning("Teks hanya berisi karakter khusus atau tanda baca yang terhapus saat pembersihan.")
    else:
        probs = pipeline.predict_proba([cleaned])[0]
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        pred_conf = probs[pred_idx] * 100
        
        st.markdown("---")
        st.subheader("📊 Hasil Analisis Sentimen")
        
        res_col1, res_col2 = st.columns([1, 1.2])
        
        with res_col1:
            if pred_label == "positive":
                st.success(f"### Sentimen: **POSITIVE** 🟢")
            elif pred_label == "negative":
                st.error(f"### Sentimen: **NEGATIVE** 🔴")
            else:
                st.warning(f"### Sentimen: **NEUTRAL** 🟡")
                
            st.write(f"Tingkat Keyakinan: **{pred_conf:.1f}%**")
            st.caption(f"Teks bersih yang diproses: *\"{cleaned}\"*")
            
        with res_col2:
            st.markdown("##### Probabilitas Distribusi Kelas:")
            prob_df = pd.DataFrame({
                "Sentimen": ["Negative", "Neutral", "Positive"],
                "Probabilitas": [
                    probs[classes.index("negative")],
                    probs[classes.index("neutral")],
                    probs[classes.index("positive")]
                ]
            })
            
            for _, row in prob_df.iterrows():
                st.write(f"**{row['Sentimen']}**: {row['Probabilitas']*100:.1f}%")
                st.progress(float(row['Probabilitas']))

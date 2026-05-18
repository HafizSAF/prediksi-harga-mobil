import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide"
)

# ── CSS custom ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .price-box {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(26,115,232,0.3);
        margin-bottom: 20px;
    }
    .price-box h1 { font-size: 3rem; margin: 0; font-weight: 800; }
    .price-box p  { font-size: 1.1rem; margin: 4px 0 0; opacity: 0.85; }
    .spec-card {
        background: white;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        margin-bottom: 14px;
    }
    .spec-card h4 { margin: 0 0 8px; color: #1a73e8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .spec-card p  { margin: 0; font-size: 1.2rem; font-weight: 600; color: #333; }
    .author-box {
        background: #e8f0fe;
        border-radius: 12px;
        padding: 16px 22px;
        text-align: center;
        margin-top: 10px;
        border: 1px solid #c5d8fb;
    }
    .author-box p { margin: 3px 0; color: #333; }
    .badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 4px;
    }
    .header-sub {
        color: #666;
        font-size: 1rem;
        margin-bottom: 24px;
    }
    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px;
        font-size: 1.1rem;
        font-weight: 700;
        cursor: pointer;
        margin-top: 10px;
        box-shadow: 0 4px 14px rgba(26,115,232,0.35);
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #1557b0, #0a2f6b);
    }
</style>
""", unsafe_allow_html=True)


# ── Train model (cached) ──────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("Car_sales.xls")
    df = df.dropna(subset=["Price_in_thousands"])

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    le = LabelEncoder()
    df["Vehicle_type_encoded"] = le.fit_transform(df["Vehicle_type"])

    features = ["Engine_size", "Horsepower", "Wheelbase", "Width", "Length",
                "Curb_weight", "Fuel_capacity", "Fuel_efficiency", "Vehicle_type_encoded"]

    model = LinearRegression()
    model.fit(df[features], df["Price_in_thousands"])

    # Top 10 for reference
    df["Car_Name"] = df["Manufacturer"] + " " + df["Model"]
    top10 = df.nlargest(10, "Sales_in_thousands")[
        ["Car_Name", "Sales_in_thousands", "Price_in_thousands"]
    ].reset_index(drop=True)

    return model, le, top10


model, le, top10 = train_model()

# ── Header ───────────────────────────────────────────────────
st.markdown('<p class="header-title">🚗 Sistem Prediksi Harga Mobil</p>', unsafe_allow_html=True)
st.markdown('<p class="header-sub">Masukkan spesifikasi teknis kendaraan untuk mendapatkan estimasi harga pasar</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Layout: 2 columns ────────────────────────────────────────
col_input, col_output = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown("### 📋 Input Spesifikasi Mobil")

    v_type = st.selectbox("🏷️ Tipe Kendaraan", options=["Passenger", "Car"],
                          help="Passenger = Sedan/MPV, Car = Sports/Coupe")

    c1, c2 = st.columns(2)
    with c1:
        engine  = st.slider("🔧 Engine Size (Liter)", 1.0, 8.0, 2.5, 0.1)
        wheelbase = st.slider("📏 Wheelbase (inci)",  92.6, 138.7, 107.0, 0.1)
        length  = st.slider("📐 Length (inci)",      149.4, 224.5, 190.0, 0.5)
        fuel_c  = st.slider("⛽ Fuel Capacity (galon)", 10.0, 32.0, 16.0, 0.5)
    with c2:
        hp      = st.slider("⚡ Horsepower (HP)",    55, 450, 180, 5)
        width   = st.slider("↔️ Width (inci)",        62.6, 79.9, 70.0, 0.1)
        weight  = st.slider("⚖️ Curb Weight (K lbs)", 1.9, 5.6, 3.2, 0.1)
        fuel_e  = st.slider("🌿 Fuel Efficiency (MPG)", 15.0, 45.0, 27.0, 0.5)

    predict_btn = st.button("🔍 Hitung Harga Mobil")

# ── Prediction ────────────────────────────────────────────────
encoded = int(le.transform([v_type])[0])
input_df = pd.DataFrame([[engine, hp, wheelbase, width, length, weight, fuel_c, fuel_e, encoded]],
    columns=["Engine_size","Horsepower","Wheelbase","Width","Length",
             "Curb_weight","Fuel_capacity","Fuel_efficiency","Vehicle_type_encoded"])

predicted_price = model.predict(input_df)[0]
predicted_price = max(predicted_price, 1.0)  # floor

# ── Output column ─────────────────────────────────────────────
with col_output:
    st.markdown("### 💰 Hasil Prediksi")

    # Price box
    st.markdown(f"""
    <div class="price-box">
        <p>Perkiraan Harga Mobil</p>
        <h1>${predicted_price:.1f}K</h1>
        <p>≈ ${predicted_price*1000:,.0f} USD</p>
    </div>
    """, unsafe_allow_html=True)

    # Spec summary cards
    st.markdown("**Spesifikasi yang Diinputkan:**")
    specs = [
        ("🏷️ Tipe Kendaraan",     v_type),
        ("🔧 Engine Size",         f"{engine} Liter"),
        ("⚡ Horsepower",          f"{hp} HP"),
        ("📏 Wheelbase",           f"{wheelbase} inci"),
        ("↔️ Width",               f"{width} inci"),
        ("📐 Length",              f"{length} inci"),
        ("⚖️ Curb Weight",        f"{weight} K lbs"),
        ("⛽ Fuel Capacity",       f"{fuel_c} Galon"),
        ("🌿 Fuel Efficiency",     f"{fuel_e} MPG"),
    ]
    r1, r2 = st.columns(2)
    for i, (label, val) in enumerate(specs):
        col = r1 if i % 2 == 0 else r2
        col.markdown(f"""
        <div class="spec-card">
            <h4>{label}</h4>
            <p>{val}</p>
        </div>""", unsafe_allow_html=True)

    # Author box
    st.markdown("""
    <div class="author-box">
        <p style="font-weight:700; font-size:0.9rem; color:#1a73e8;">SISTEM INI DIBUAT OLEH</p>
        <p><b>Nama  :</b> &nbsp;Hafiz Syafiq Ahmad Fauzi</p>
        <p><b>NPM   :</b> &nbsp;237006050</p>
        <p><b>Kelas :</b> &nbsp;B</p>
    </div>
    """, unsafe_allow_html=True)

# ── Top 10 chart (bawah) ──────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Referensi: 10 Mobil dengan Penjualan Terbanyak")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#f8f9fa")

# Sales chart
colors_s = ["#1a73e8" if i == 0 else "#90bef5" for i in range(10)]
axes[0].barh(top10["Car_Name"], top10["Sales_in_thousands"], color=colors_s)
axes[0].invert_yaxis()
axes[0].set_xlabel("Penjualan (ribuan unit)")
axes[0].set_title("Jumlah Penjualan (K unit)", fontweight="bold")
axes[0].set_facecolor("#f8f9fa")
for spine in axes[0].spines.values(): spine.set_visible(False)

# Price chart + prediction line
colors_p = ["#f4b942" if i == 0 else "#f9d98a" for i in range(10)]
axes[1].barh(top10["Car_Name"], top10["Price_in_thousands"], color=colors_p)
axes[1].invert_yaxis()
axes[1].axvline(predicted_price, color="#e53935", linewidth=2, linestyle="--", label=f"Prediksi: ${predicted_price:.1f}K")
axes[1].set_xlabel("Harga ($000)")
axes[1].set_title("Harga (ribuan USD)", fontweight="bold")
axes[1].set_facecolor("#f8f9fa")
axes[1].legend(loc="lower right", fontsize=9)
for spine in axes[1].spines.values(): spine.set_visible(False)

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
<p style='text-align:center; color:#999; font-size:0.8rem; margin-top:20px;'>
    Garis merah putus-putus pada chart harga menunjukkan posisi harga prediksi Anda dibandingkan 10 mobil terlaris.
</p>
""", unsafe_allow_html=True)

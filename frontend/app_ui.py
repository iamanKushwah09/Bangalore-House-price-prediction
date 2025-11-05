import time
import requests
import streamlit as st

# ==========================
# BASIC PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="🏠 Bangalore House Price Prediction",
    page_icon="🏙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==========================
# BACKEND URL (secrets → fallback)
# ==========================
try:
    API_URL = st.secrets["API_URL"]  # .streamlit/secrets.toml (for cloud)
except Exception:
    API_URL = "http://127.0.0.1:8000"  # local fallback


# ==========================
# HEADER
# ==========================
st.markdown(
    """
    <h1 style='text-align:center; margin-bottom:0'>🏠 Bangalore House Price Prediction</h1>
    <p style='text-align:center; color:#6B7280; margin-top:6px'>
        FastAPI + Streamlit • Linear Regression • Price in Lakhs
    </p>
    """,
    unsafe_allow_html=True,
)
st.write("")

# Backend health check
status_box = st.empty()
try:
    ping = requests.get(f"{API_URL}/", timeout=5)
    if ping.ok:
        status_box.success(f"✅ FastAPI connected at: {API_URL}")
    else:
        status_box.warning(f"🟡 FastAPI reachable but returned {ping.status_code}")
except requests.exceptions.RequestException:
    status_box.error(f"❌ Cannot reach FastAPI at {API_URL}. Please start backend.")

st.divider()

# ==========================
# UTILS
# ==========================
def validate_inputs(bhk: int, bath: int, balcony: int, sqft: float, pps: float) -> list[str]:
    """Light client-side checks (backend has strict validation)."""
    issues = []
    if bhk < 1 or bhk > 12: issues.append("BHK must be between 1 and 12.")
    if bath < 1 or bath > 12: issues.append("Bathrooms must be between 1 and 12.")
    if balcony < 0 or balcony > 6: issues.append("Balconies must be between 0 and 6.")
    if sqft <= 200: issues.append("Total sqft must be > 200.")
    if pps <= 0: issues.append("Price per sqft must be > 0.")
    # Soft mirror of cleaning rule for better UX hint (backend enforces strictly)
    if bhk >= 1 and (sqft / bhk) < 350:
        issues.append("Sqft per BHK should be ≥ 350 (data cleaning rule).")
    if bath > bhk + 2:
        issues.append("Bathrooms should typically be ≤ (BHK + 2).")
    return issues


def predict_post(bhk, bath, balcony, sqft, pps):
    payload = {
        "bath": int(bath),
        "balcony": int(balcony),
        "total_sqft_int": float(sqft),
        "bhk": int(bhk),
        "price_per_sqft": float(pps),
    }
    with st.spinner("⏳ Predicting (POST /predict)..."):
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=12)
    return r


def predict_get(bhk, bath, balcony, sqft, pps):
    params = {
        "bath": int(bath),
        "balcony": int(balcony),
        "total_sqft_int": float(sqft),
        "price_per_sqft": float(pps),
    }
    with st.spinner("⏳ Predicting (GET /predict/{bhk})..."):
        r = requests.get(f"{API_URL}/predict/{int(bhk)}", params=params, timeout=12)
    return r


# ==========================
# TABS
# ==========================
tab_pred, tab_info, tab_help = st.tabs(["🔮 Prediction", "📊 Model Info", "❓ Help"])

with tab_pred:
    st.subheader("📋 Enter House Details")

    c1, c2 = st.columns(2)
    with c1:
        bhk = st.number_input("🏘️ BHK", min_value=1, max_value=12, value=2, step=1)
        bath = st.number_input("🛁 Bathrooms", min_value=1, max_value=12, value=2, step=1)
        balcony = st.number_input("🌅 Balconies", min_value=0, max_value=6, value=1, step=1)
    with c2:
        sqft = st.number_input("📐 Total Area (sqft)", min_value=250.0, value=1000.0, step=50.0)
        pps = st.number_input("💸 Price per Sqft (₹)", min_value=1.0, value=6000.0, step=100.0)

    st.caption("💡 Tip: `price_per_sqft` aapke trained features ka hissa tha — isliye yahan dena zaroori hai.")

    # Client-side validation hints
    issues = validate_inputs(bhk, bath, balcony, sqft, pps)
    if issues:
        with st.container():
            st.warning("Please check these before predicting:")
            for i in issues:
                st.write(f"• {i}")

    st.write("")
    a, b = st.columns(2)
    with a:
        if st.button("🚀 Predict (POST • recommended)", use_container_width=True):
            if issues:
                st.stop()
            try:
                r = predict_post(bhk, bath, balcony, sqft, pps)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"💰 Predicted Price: ₹ {data['predicted_price_lakhs']} lakh")
                    st.progress(100)
                    with st.expander("🔍 Details"):
                        st.json(data)
                    # Tiny visualization (for demo feel)
                    st.caption("Quick glance:")
                    st.bar_chart({"Sqft": [sqft], "Predicted Price (₹L)": [data["predicted_price_lakhs"]]})
                else:
                    st.error(f"⚠️ API Error {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"🚫 Connection error: {e}")

    with b:
        if st.button("🔗 Predict (GET • demo)", use_container_width=True):
            if issues:
                st.stop()
            try:
                r = predict_get(bhk, bath, balcony, sqft, pps)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"💰 Predicted Price: ₹ {data['predicted_price_lakhs']} lakh")
                    with st.expander("🔍 Details"):
                        st.json(data)
                else:
                    st.error(f"⚠️ API Error {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"🚫 Connection error: {e}")

with tab_info:
    st.subheader("📊 Model Metadata")
    st.caption("Backend se live info fetch ki ja rahi hai...")
    try:
        info = requests.get(f"{API_URL}/model/info", timeout=8).json()
        st.success("Model info loaded.")
        st.json(info)
        st.markdown(
            f"**Model Type:** `{info.get('model_type', 'Unknown')}`  \n"
            f"**Feature Order:** `{', '.join(info.get('feature_order', []))}`"
        )
    except Exception as e:
        st.error(f"Couldn't fetch model info: {e}")

with tab_help:
    st.subheader("❓ How to use")
    st.markdown(
        """
        **1.** BHK, Bathrooms, Balconies, Total Sqft, aur Price per Sqft bhar dijiye.  
        **2.** *POST Predict* dabayein (recommended).  
        **3.** Result lakhs me show hoga.  
        **4.** Validation rules:
        - Sqft per BHK ≥ **350** (clean data rule)
        - Bathrooms ≤ **BHK + 2** (typical constraint)

        **Note:** Ye UI **FastAPI** backend ko call karta hai. Agar backend nahi chal raha hai toh upar error aayega.
        """
    )

# Footer
st.write("")
st.markdown(
    """
    <hr style="margin-top:30px; margin-bottom:10px"/>
    <p style="text-align:center; color:#9CA3AF">
        © 2025 Aman Kushwah • BCA Mini Project • FastAPI × Streamlit
    </p>
    """,
    unsafe_allow_html=True,
)

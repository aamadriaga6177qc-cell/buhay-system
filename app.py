import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- 1. PAGE SETUP & UI DESIGN ---
st.set_page_config(page_title="BUHAY ADSS Dashboard", page_icon="🐟", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 40px; font-weight: 900; color: #1E3A8A; margin-bottom: -10px; }
    .sub-title { font-size: 18px; color: #64748B; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🐟 BUHAY System Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Bridging Understanding of Habitat and Aquaculture Yield | BFAR NFFTC</p>', unsafe_allow_html=True)
st.divider()

# --- 2. AI BACKEND (UPDATED PARA SA GITHUB) ---
@st.cache_resource
def train_buhay_ai():
    # Binago ko ito para basahin direkta kung saan nakasave ang file, hindi na sa Google Drive
    file_path = 'BUHAY_Cleaned_Ready.csv' 
    
    if not os.path.exists(file_path):
        return None, None, None, None
        
    df = pd.read_csv(file_path)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['temp', 'weather_condition', 'do']])
    
    look_back = 3
    X_lstm, y_lstm = [], []
    for i in range(len(scaled_data) - look_back):
        X_lstm.append(scaled_data[i:(i + look_back), :])
        y_lstm.append(scaled_data[i + look_back, 2])
        
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 3)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    with st.spinner('Sinisimulan ang BUHAY AI Engine...'):
        model.fit(X_lstm, y_lstm, epochs=30, batch_size=16, verbose=0)
        
    return model, scaler, scaled_data, df

model, scaler, scaled_data, df = train_buhay_ai()

if df is None:
    st.error("❌ HINDI MAHANAP ANG DATASET. Siguraduhing magkasama ang app.py at BUHAY_Cleaned_Ready.csv sa iisang folder/repository.")
    st.stop()

# --- 3. PRESCRIPTIVE LOGIC ---
def buhay_adss_alert(predicted_do):
    if predicted_do < 3.0:
        st.error(f"🔴 **CRITICAL RISK | Predicted DO: {predicted_do:.2f} mg/L**\n\n**Action:** Activate emergency aerators immediately and STOP all feeding.")
    elif predicted_do >= 3.0 and predicted_do < 4.0:
        st.warning(f"🟠 **WARNING | Predicted DO: {predicted_do:.2f} mg/L**\n\n**Action:** Oxygen levels dropping. Reduce feeding by 50% to minimize metabolic oxygen demand.")
    else:
        st.success(f"🟢 **SAFE | Predicted DO: {predicted_do:.2f} mg/L**\n\n**Action:** Normal environmental conditions. Proceed with regular alternate-day feeding strategy.")

# --- 4. FRONT-END TABS ---
tab1, tab2 = st.tabs(["📡 Live Monitoring", "🎛️ Interactive Simulation & Economics"])

with tab1:
    st.subheader("BFAR Real-Time Data Assessment")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("📋 **Recent Log Entries (Past 3 Days)**")
        st.dataframe(df[['temp', 'weather_condition', 'do']].tail(3), use_container_width=True)
        st.info("💡 **Weather Legend:**\n1=Rainy/Stormy, 2=Cloudy to Rainy, 3=Cloudy to Sunny, 4=Sunny")
        
    with col2:
        st.write("🤖 **LSTM Predictive Analytics**")
        st.write("Tinatasa ng AI ang nakaraang 3 araw upang hulaan ang DO level bukas.")
        if st.button("▶ GENERATE OFFICIAL FORECAST", type="primary", use_container_width=True):
            last_3_days_scaled = scaled_data[-3:]
            live_input = np.reshape(last_3_days_scaled, (1, 3, 3))
            predicted_scaled_do = model.predict(live_input, verbose=0)
            
            dummy = np.zeros((1, 3))
            dummy[0, 2] = predicted_scaled_do[0, 0]
            real_do_prediction = scaler.inverse_transform(dummy)[0, 2]
            
            st.divider()
            buhay_adss_alert(real_do_prediction)

with tab2:
    st.subheader("Simulate Environmental Triggers (What-If Analysis)")
    
    sim_col1, sim_col2 = st.columns([1, 1])
    
    with sim_col1:
        st.markdown("**🔧 Simulation Inputs**")
        sim_temp = st.slider("🌡️ Current Temperature (°C)", min_value=20.0, max_value=40.0, value=30.0, step=0.1)
        sim_do = st.slider("💧 Current Dissolved Oxygen (mg/L)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
        
        weather_map = {"Sunny (4)": 4, "Cloudy to Sunny (3)": 3, "Cloudy to Rainy (2)": 2, "Rainy/Stormy (1)": 1}
        sim_weather_label = st.selectbox("☁️ Weather Forecast", list(weather_map.keys()))
        sim_weather = weather_map[sim_weather_label]

    with sim_col2:
        st.markdown("**🧠 AI Prescriptive Output**")
        
        last_2_days = scaled_data[-3:-1] 
        sim_array = np.array([[sim_temp, sim_weather, sim_do]])
        sim_scaled = scaler.transform(sim_array)
        
        live_sim_input = np.vstack((last_2_days, sim_scaled))
        live_sim_input = np.reshape(live_sim_input, (1, 3, 3))
        
        sim_predicted_scaled_do = model.predict(live_sim_input, verbose=0)
        
        dummy_sim = np.zeros((1, 3))
        dummy_sim[0, 2] = sim_predicted_scaled_do[0, 0]
        sim_real_do_prediction = scaler.inverse_transform(dummy_sim)[0, 2]
        
        if sim_real_do_prediction < 0:
            sim_real_do_prediction = 0.0 
            
        buhay_adss_alert(sim_real_do_prediction)

    st.divider()

    st.subheader("🏢 Institutional Decision Support (BFAR Protocols)")
    inst_col1, inst_col2 = st.columns([1, 1])

    with inst_col1:
        st.markdown("#### 💰 Economic Cost-Benefit")
        fish_price = st.number_input("Market Price of Tilapia (₱/kg)", value=120.0)
        est_biomass = st.number_input("Est. Biomass in Affected Ponds (kg)", value=1000.0)
        aeration_cost = st.number_input("Cost of Aeration (₱/day)", value=500.0)

        do_val = sim_real_do_prediction
        if do_val < 3.0:
            potential_loss_peso = (est_biomass * 0.80) * fish_price
            st.error(f"⚠️ **Potential Loss:** ₱ {potential_loss_peso:,.2f}")
            st.success(f"✅ **Savings if Aerator Used:** ₱ {(potential_loss_peso - aeration_cost):,.2f}")
        elif do_val < 4.0:
            potential_loss_peso = (est_biomass * 0.20) * fish_price
            st.warning(f"⚠️ **Minor Loss Risk:** ₱ {potential_loss_peso:,.2f}")
        else:
            st.info(f"🟢 **Safe. Est. Crop Value:** ₱ {(est_biomass * fish_price):,.2f}")

    with inst_col2:
        st.markdown("#### 👥 Manpower Allocation Status")
        total_personnel = st.number_input("Total Farm Personnel Available", value=10, min_value=1)
        normal_ops = st.number_input("Personnel for Normal Ops (per pond)", value=1, min_value=1)
        emergency_ops = st.number_input("Personnel for Emergency Aeration (per pond)", value=3, min_value=1)
        affected_ponds = st.number_input("Number of Ponds at Risk", value=2, min_value=1)

        st.markdown("**Allocation Status:**")
        if do_val < 3.0:
            required_personnel = affected_ponds * emergency_ops
            status_type = "🔴 Critical Intervention"
        else:
            required_personnel = affected_ponds * normal_ops
            status_type = "🟢 Normal Operations"

        surplus = total_personnel - required_personnel

        st.write(f"- **Scenario:** {status_type}")
        st.write(f"- **Available / Required:** {total_personnel} / {required_personnel}")

        if surplus >= 0:
            st.success(f"✅ **Surplus Personnel: {surplus}**.")
        else:
            st.error(f"⚠️ **Shortfall: {abs(surplus)}**. Need immediate backup!")
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Wheat Spoilage Detector",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox div[data-baseweb="select"] {
        margin-bottom: 20px;
    }
    .stSlider {
        color: #4CAF50;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header {
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load('models/logistic_regression_model.pkl'),
        "Decision Tree": joblib.load('models/decision_tree_model.pkl'),
        "Random Forest": joblib.load('models/random_forest_model.pkl')
    }
    scaler = joblib.load('models/scaler.pkl')
    return models, scaler

models, scaler = load_models()

# Header
st.title("🌾 Wheat Spoilage Detection System")
st.markdown("""
<div class="header">
    <p>Predict wheat spoilage based on environmental conditions using machine learning models.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for model selection
with st.sidebar:
    st.header("Model Configuration")
    selected_model = st.selectbox(
        "Select Prediction Model",
        list(models.keys()),
        index=0,
        help="Choose which machine learning model to use for predictions"
    )
    
    st.markdown("---")
    st.markdown("""
    **About the Models:**
    - **Logistic Regression:** Fast linear model
    - **Decision Tree:** Simple rule-based model
    - **Random Forest:** Powerful ensemble model
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Parameters")
    
    # Interactive sliders
    humidity = st.slider(
        "Humidity (%)", 
        min_value=20.0, max_value=90.0, value=50.0, step=0.1,
        help="Optimal range: 40-55% | Spoilage risk: >60%"
    )
    
    gases = st.slider(
        "Gases Concentration (ppm)", 
        min_value=0.0, max_value=400.0, value=100.0, step=1.0,
        help="Normal air: <150ppm | Spoilage risk: >180ppm"
    )
    
    temperature = st.slider(
        "Temperature (°C)", 
        min_value=10.0, max_value=45.0, value=22.0, step=0.1,
        help="Optimal range: 15-27°C | Spoilage risk: >30°C"
    )

with col2:
    st.header("Safe Ranges")
    
    # Safe ranges visualization
    ranges_df = pd.DataFrame({
        'Feature': ['Humidity', 'Gases', 'Temperature'],
        'Safe Min': [40, 0, 15],
        'Safe Max': [55, 150, 27],
        'Current': [humidity, gases, temperature]
    })
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=ranges_df.melt(id_vars='Feature'), 
                x='value', y='Feature', hue='variable',
                palette={'Safe Min': '#4CAF50', 'Safe Max': '#4CAF50', 'Current': '#FF5722'})
    plt.title("Current Values vs Safe Ranges")
    plt.xlabel("Value")
    plt.ylabel("Feature")
    plt.legend(title="")
    st.pyplot(fig)

# Prediction button
if st.button("Predict Spoilage Status", key="predict_button"):
    # Prepare input
    features = np.array([[humidity, gases, temperature]])
    features_scaled = scaler.transform(features)
    
    # Get selected model
    model = models[selected_model]
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    
    # Display results
    with st.container():
        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ Warning: Potential spoilage detected ({proba[1]*100:.1f}% confidence)")
        else:
            st.success(f"✅ Wheat is safe ({proba[0]*100:.1f}% confidence)")
        
        # Result details
        with st.expander("View Prediction Details"):
            st.markdown(f"""
            **Selected Model:** {selected_model}  
            **Humidity:** {humidity}%  
            **Gases:** {gases} ppm  
            **Temperature:** {temperature}°C  
            
            **Probabilities:**  
            - Safe: {proba[0]*100:.1f}%  
            - Spoiled: {proba[1]*100:.1f}%
            """)
            
            # Confidence visualization
            fig2, ax2 = plt.subplots(figsize=(8, 2))
            ax2.barh(['Spoilage Probability'], [proba[1]], color='#FF5722' if prediction == 1 else '#4CAF50')
            ax2.set_xlim(0, 1)
            ax2.set_title("Spoilage Confidence Level")
            st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Wheat Spoilage Detection System | AIJNA PROJECT</p>
</div>
""", unsafe_allow_html=True)
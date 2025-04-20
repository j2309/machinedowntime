import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle, joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from io import StringIO

# Load model and preprocessor
model = pickle.load(open('rfc.pkl', 'rb'))
preprocessor = joblib.load('preprocessor.sav')

# Updated standard parameters based on your new columns
STANDARD_PARAMS = {
    'Hydraulic_Pressure(bar)': 150.0,
    'Coolant_Pressure(bar)': 5.0,
    'Air_System_Pressure(bar)': 7.0,
    'Coolant_Temperature': 25.0,
    'Hydraulic_Oil_Temperature(°C)': 45.0,
    'Spindle_Bearing_Temperature(°C)': 60.0,
    'Spindle_Vibration(µm)': 10.0,
    'Tool_Vibration(µm)': 8.0,
    'Spindle_Speed(RPM)': 8000,
    'Voltage(volts)': 400.0,
    'Torque(Nm)': 50.0,
    'Cutting(kN)': 2.5
}

def get_user_input():
    st.sidebar.header("🔧 Machine Parameters Input")
    
    inputs = {}
    for param, default_val in STANDARD_PARAMS.items():
        # Clean parameter name for display (remove units)
        display_name = param.split('(')[0].replace('_', ' ')
        
        if param in ['Spindle_Speed(RPM)', 'Torque(Nm)', 'Cutting(kN)']:  # Integer parameters
            inputs[param] = st.sidebar.number_input(
                f"{display_name} (Standard: {default_val})",
                min_value=0,
                value=int(default_val),
                key=f"sidebar_{param}"
            )
        else:  # Float parameters
            inputs[param] = st.sidebar.number_input(
                f"{display_name} (Standard: {default_val})",
                value=float(default_val),
                key=f"sidebar_{param}"
            )
    
    return pd.DataFrame([inputs])

def process_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Check if all required columns are present
        required_cols = set(STANDARD_PARAMS.keys())
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Machine Downtime Predictor",
        page_icon="🏭",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .header-style {
        font-size: 24px;
        color: #2e86ab;
        padding-bottom: 10px;
        border-bottom: 2px solid #4CAF50;
        margin-bottom: 20px;
    }
    .input-section {
        background-color: green;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
        background-color: green;
    }
    .failure {
        color: #d9534f;
        font-weight: bold;
    }
    .normal {
        color: #5cb85c;
        font-weight: bold;
    }
    .file-uploader {
        padding: 20px;
        border: 2px dashed #cccccc;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<div class="header-style">🏭 Machine Downtime Prediction System</div>', unsafe_allow_html=True)
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="input-section">📋 Input Method Selection</div>', unsafe_allow_html=True)
        
        # File uploader
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📁 Upload CSV or Excel file with machine parameters",
            type=['csv', 'xls', 'xlsx'],
            key="file_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            input_df = process_uploaded_file(uploaded_file)
            if input_df is not None:
                st.success("File successfully uploaded and processed!")
                with st.expander("👀 View Uploaded Data"):
                    st.dataframe(input_df)
        else:
            # Manual input
            input_df = get_user_input()
            
            # Show user inputs
            with st.expander("🔍 View Input Parameters"):
                st.dataframe(input_df)
    
    with col2:
        st.markdown('<div class="input-section">📊 Prediction Results</div>', unsafe_allow_html=True)
        
        # Display standard parameters
        with st.expander("⚙ Standard Operating Parameters (Reference)"):
            st.table(pd.DataFrame.from_dict(STANDARD_PARAMS, orient='index', columns=['Value']))
        
        # Predict button
        if st.button("🔮 Predict Downtime", use_container_width=True):
            if 'input_df' not in locals():
                st.warning("Please provide input parameters first")
                return
            
            try:
                # Ensure columns are in correct order
                input_df = input_df[list(STANDARD_PARAMS.keys())]
                
                # Preprocess
                processed_data = preprocessor.transform(input_df)
                
                # Predict
                prediction = model.predict(processed_data)
                proba = model.predict_proba(processed_data)[0]
                
                # Display results
                st.markdown(f"""
                <div class="result-box">
                    <h3>Prediction Result</h3>
                    <p>Machine Status: <span class="{'failure' if prediction[0] == 1 else 'normal'}">
                        {'1 (NO Machine Failure)' if prediction[0] == 1 else '0 (Machine Failure)'}
                    </span></p>
                    
                    
                </div>
                """, unsafe_allow_html=True)
                
                # Interpretation guide
                st.markdown("""
                ### 📝 Interpretation Guide
                - *0*: Machine Failure 
                - *1*: NO Machine Failure 
                """)
                
                # Show detailed results if multiple rows
                if len(input_df) > 1:
                    input_df['Prediction'] = prediction
                    input_df['Failure_Probability'] = [x[1] for x in model.predict_proba(processed_data)]
                    with st.expander("📈 Detailed Results"):
                        st.dataframe(input_df)
                        
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    main()

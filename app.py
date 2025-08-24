import pandas as pd
import streamlit as st
import joblib
from io import StringIO

# ✅ Load pipeline (preprocessor + model together)
pipeline = joblib.load("pipeline.pkl")

# Updated standard parameters
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

# -------------------------------
# Sidebar manual input
# -------------------------------
def get_user_input():
    st.sidebar.header("🔧 Machine Parameters Input")
    inputs = {}
    for param, default_val in STANDARD_PARAMS.items():
        display_name = param.split('(')[0].replace('_', ' ')
        if param in ['Spindle_Speed(RPM)', 'Torque(Nm)', 'Cutting(kN)']:  # Integer params
            inputs[param] = st.sidebar.number_input(
                f"{display_name} (Standard: {default_val})",
                min_value=0,
                value=int(default_val),
                key=f"sidebar_{param}"
            )
        else:  # Float params
            inputs[param] = st.sidebar.number_input(
                f"{display_name} (Standard: {default_val})",
                value=float(default_val),
                key=f"sidebar_{param}"
            )
    return pd.DataFrame([inputs])

# -------------------------------
# File upload processing
# -------------------------------
def process_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file format. Please upload CSV or Excel.")
            return None

        required_cols = set(STANDARD_PARAMS.keys())
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            st.error(f"⚠ Missing required columns: {', '.join(missing)}")
            return None

        return df
    except Exception as e:
        st.error(f"⚠ Error processing file: {str(e)}")
        return None

# -------------------------------
# Main app
# -------------------------------
def main():
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
        background-color: #f0fdf4;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
        background-color: #f9f9f9;
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

    st.markdown('<div class="header-style">🏭 Machine Downtime Prediction System</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    # ---------------- Left column: Input ----------------
    with col1:
        st.markdown('<div class="input-section">📋 Input Method Selection</div>', unsafe_allow_html=True)

        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📁 Upload CSV or Excel file with machine parameters",
            type=['csv', 'xls', 'xlsx'],
            key="file_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        input_df = None
        if uploaded_file is not None:
            input_df = process_uploaded_file(uploaded_file)
            if input_df is not None:
                st.success("✅ File successfully uploaded and processed!")
                with st.expander("👀 View Uploaded Data"):
                    st.dataframe(input_df)
        else:
            input_df = get_user_input()
            with st.expander("🔍 View Input Parameters"):
                st.dataframe(input_df)

    # ---------------- Right column: Prediction ----------------
    with col2:
        st.markdown('<div class="input-section">📊 Prediction Results</div>', unsafe_allow_html=True)

        with st.expander("⚙ Standard Operating Parameters (Reference)"):
            st.table(pd.DataFrame.from_dict(STANDARD_PARAMS, orient='index', columns=['Value']))

        if st.button("🔮 Predict Downtime", use_container_width=True):
            if input_df is None:
                st.warning("⚠ Please provide input parameters first")
                return
            try:
                input_df = input_df[list(STANDARD_PARAMS.keys())]
                prediction = pipeline.predict(input_df)
                proba = pipeline.predict_proba(input_df)[0]

                status = "Machine Failure" if prediction[0] == 1 else "NO Machine Failure"
                css_class = "failure" if prediction[0] == 1 else "normal"

                st.markdown(f"""
                <div class="result-box">
                    <h3>Prediction Result</h3>
                    <p>Machine Status: <span class="{css_class}">{status}</span></p>
                    <p>Failure Probability: {proba[1]*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                ### 📝 Interpretation Guide
                - *0*: NO Machine Failure  
                - *1*: Machine Failure  
                """)

                if len(input_df) > 1:
                    input_df['Prediction'] = prediction
                    with st.expander("📈 Detailed Results"):
                        st.dataframe(input_df)

            except Exception as e:
                st.error(f"⚠ Error in prediction: {str(e)}")


if __name__ == '__main__':
    main()

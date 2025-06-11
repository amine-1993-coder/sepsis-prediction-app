import streamlit as st
import pandas as pd
import json
import requests
import numpy as np
import time
from datetime import datetime
import pytz
import plotly.graph_objects as go
import yagmail
import io

# 🔧 Gmail sender setup
SENDER_EMAIL = "amine2671993@gmail.com"
RECIPIENT_EMAILS = [
    "amine2671993@gmail.com",
    #"example1@hospital.org",
    "khelifi@rowan.edu"
]

# ✅ API endpoint deployed on Render
MODEL_API_URL = "https://sepsis-model-api.onrender.com/test/v1.0/prediction/"

# 🕒 Get current EST timestamp
eastern = pytz.timezone("US/Eastern")
now_est = datetime.now(eastern)
timestamp_str = now_est.strftime("Prediction Results of %m-%d-%Y - %I-%M %p EST")

# 📤 Email sending function using yagmail
def send_email_with_csv(csv_content, subject, filename, recipient_emails):
    try:
        yag = yagmail.SMTP(SENDER_EMAIL, "pgvj kyrn ragl zqjj")
        attachment = io.StringIO(csv_content)
        attachment.name = filename
        yag.send(
            to=recipient_emails,
            subject=subject,
            contents="Hello Doctor,\n\nPlease find the attached sepsis prediction report.",
            attachments=attachment
        )
        st.success(f"📧 Report emailed to: {', '.join(recipient_emails)}")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

# ✅ Only include features your model was trained on
VALID_FEATURES = [
    'Age', 'Gender', 'HeartRate', 'Temp', 'SystolicBP', 'MeanBP', 'DiastolicBP',
    'RespRate', 'OximetrySat', 'Potassium', 'Chloride', 'Calcium', 'Hemoglobin', 'pH',
    'BaseExcess', 'Bicarbonate', 'FiO2', 'Glucose', 'BUN', 'Creatinine', 'Magnesium',
    'SGOT', 'SGPT', 'TotalBili', 'WBC', 'Platelets', 'PaCO2', 'Lactate'
]

def call_docker_model(payload: dict):
    try:
        # 🧼 Remove invalid or null features for each row
        cleaned_payload = {
            "sepsis_fv": [
                {
                    k: float(v)
                    for k, v in row.items()
                    if k in VALID_FEATURES and v is not None and not pd.isna(v)
                }
                for row in payload["sepsis_fv"]
            ]
        }

        # 🧪 DEBUG (optional)
        #st.write("📤 Cleaned payload:", json.dumps(cleaned_payload, indent=2))

        headers = {"Content-Type": "application/json"}
        response = requests.post(MODEL_API_URL, json=cleaned_payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        raw_preds = result.get("sepsis_risk", [])

        if len(raw_preds) != len(cleaned_payload["sepsis_fv"]):
            st.error("🚨 Mismatch: Model returned fewer predictions than inputs.")
            return pd.DataFrame()

        df = pd.DataFrame(cleaned_payload["sepsis_fv"])
        df.insert(0, "PatientID", [f"Patient_{i+1}" for i in range(len(df))])
        df['SepsisPrediction'] = ["Positive" if p == 1 else "Negative" for p in raw_preds]
        df['Warning'] = df['Temp'].apply(lambda t: "Temperature is low" if t is not None and t < 35 else "")
        return df

    except requests.exceptions.HTTPError as http_err:
        st.error(f"❌ API error {http_err.response.status_code}: {http_err.response.text}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return pd.DataFrame()



# 🎨 Row coloring
def style_predictions(df):
    def highlight(row):
        color = "#ffcccc" if row["SepsisPrediction"] == "Positive" else "#ccffcc"
        return ['background-color: {}'.format(color)] * len(row)
    return df.style.apply(highlight, axis=1)

# 🧬 Streamlit UI
st.set_page_config(page_title="Sepsis Prediction ML Tool", layout="centered")

# 🖼️ Image logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images.png", width=100, use_container_width=False)

st.markdown("<h1 style='text-align: center; color: black;'>🩺 Sepsis Prediction ML Tool</h1>", unsafe_allow_html=True)

# 📁 JSON Upload
uploaded_file = st.file_uploader("Upload The Hospital Lab Records", type=["json"])
json_payload = None

if uploaded_file is not None:
    try:
        json_payload = json.load(uploaded_file)
        if "sepsis_fv" not in json_payload:
            st.error("❌ JSON format invalid. Must contain top-level key 'sepsis_fv'.")
            json_payload = None
        else:
            st.success("✅ Hospital Lab Records loaded successfully.")
            st.markdown("### <center>Sepsis Prediction ?</center>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ Failed to parse JSON: {e}")

# ▶️ Proceed Button
st.markdown("<br>", unsafe_allow_html=True)
col_center = st.columns(3)
with col_center[1]:
    if st.button("▶️ Proceed", use_container_width=True, key="proceed_button", help="Run prediction", type="primary"):
        if json_payload:
            with st.spinner("🧠 Sending data to ML model..."):
                time.sleep(1)
                result_df = call_docker_model(json_payload)
                st.session_state['result_df'] = result_df
        else:
            st.warning("⚠️ Please upload a valid test.json file.")

# 📊 Results Display
if 'result_df' in st.session_state and not st.session_state['result_df'].empty:
    st.markdown(f"### {timestamp_str}")
    df_result = st.session_state['result_df']
    st.dataframe(style_predictions(df_result), use_container_width=True)

    pos_count = (df_result['SepsisPrediction'] == "Positive").sum()
    neg_count = (df_result['SepsisPrediction'] == "Negative").sum()
    total = len(df_result)
    pos_percent = round((pos_count / total) * 100, 1)

    fig = go.Figure(data=[go.Pie(
        labels=['Negative', 'Positive'],
        values=[neg_count, pos_count],
        hole=0.6,
        marker_colors=['#66ff99', '#ff9999'],
        textinfo='label+percent'
    )])

    fig.update_layout(
        title="Sepsis Prediction Distribution",
        annotations=[dict(text=f'{pos_percent}%<br>Positive', x=0.5, y=0.5, font_size=18, showarrow=False)]
    )

    st.plotly_chart(fig, use_container_width=True)

# 📥 Generate CSV + Download Button
st.markdown("<br>", unsafe_allow_html=True)
col_report = st.columns(3)
with col_report[1]:
    if st.button("📄 Generate Report", use_container_width=True, key="generate_button", help="Download CSV report", type="primary"):
        if 'result_df' in st.session_state:
            csv_data = st.session_state['result_df'].to_csv(index=False)
            st.session_state['csv_ready'] = True
            st.session_state['csv_data'] = csv_data

            send_email_with_csv(
                csv_content=csv_data,
                subject=timestamp_str,
                filename=f"{timestamp_str}.csv",
                recipient_emails=RECIPIENT_EMAILS
            )
        else:
            st.warning("⚠️ Run prediction before generating the report.")

if st.session_state.get('csv_ready'):
    col_download = st.columns(3)
    with col_download[1]:
        st.download_button(
            label="⬇️ Download CSV Report",
            data=st.session_state['csv_data'],
            file_name=f"{timestamp_str}.csv",
            mime="text/csv",
            use_container_width=True,
            key="csv_download"
        )

st.markdown("""
<div style="border: 2px solid #00AEEF; padding: 15px; border-radius: 10px; background-color: #F0F8FF;">
<b>Disclaimer:</b> This prediction is generated by an automated ML model and is not a substitute for professional medical diagnosis.
Always consult a licensed healthcare provider before taking any medical action.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
button[kind="primary"], .stDownloadButton button {
    background-color: #dabfff !important;
    color: black !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

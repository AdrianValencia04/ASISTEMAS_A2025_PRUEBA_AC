
import pandas as pd
import streamlit as st
from datetime import time

# Page setup
st.set_page_config(page_title="CAAT 1 – Validación Fuera de Horario", layout="wide")

# Title
st.title("🕒 CAAT 1 – Validación de Registros Fuera de Horario Laboral")

# Upload file
uploaded = st.file_uploader("📂 Subir archivo", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Sube un archivo para iniciar el análisis.")
    st.stop()

# Load data
df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

# Display data preview
st.subheader("📄 Vista previa")
st.dataframe(df.head())

# Parameters for the analysis
st.subheader("⚙️ Parámetros del análisis")
hora_inicio = st.time_input("Hora de inicio laboral", time(8, 0))
hora_fin = st.time_input("Hora de fin laboral", time(18, 0))
dias_habiles = st.multiselect("Días hábiles", [0, 1, 2, 3, 4], default=[0, 1, 2, 3, 4])

# Convert timestamps
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df['Hora'] = df['Timestamp'].dt.hour
df['DiaSemana'] = df['Timestamp'].dt.dayofweek

# Mark records outside of working hours
df["Fuera_Horario"] = (df["Hora"] < hora_inicio.hour) | (df["Hora"] > hora_fin.hour) | (~df["DiaSemana"].isin(dias_habiles))

# Risk scoring
def score_row(row):
    score = 0
    reasons = []
    if row["Fuera_Horario"]:
        score += 3
        reasons.append("Fuera de horario")
    return score, "; ".join(reasons)

df[["Puntaje_Riesgo", "Motivo_Riesgo"]] = df.apply(lambda r: pd.Series(score_row(r)), axis=1)

# Display results
st.subheader("📊 Resultados del análisis")
st.dataframe(df[["Usuario", "Puntaje_Riesgo", "Motivo_Riesgo"]])

# Download the report
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

st.download_button(
    label="📥 Descargar informe en Excel",
    data=to_excel(df),
    file_name="informe_CAAT_analisis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

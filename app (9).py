
# app.py
# ======================================================
# Suite de Auditoría Asistida por Computadora (CAAT 1–5)
# Formato: una sola página con secciones interactivas.
# Dependencias: streamlit, pandas, numpy (sin matplotlib)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import time, datetime

st.set_page_config(page_title="Suite de Auditoría Asistida por Computadora (CAAT 1–5)", layout="wide")

# ================= Utilidades generales =================

def read_any(file):
    if file is None:
        return None
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file, encoding="utf-8", sep=None, engine="python")
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(file)
        else:
            st.error("Formato no soportado. Sube CSV o XLSX.")
            return None
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None

def normalize_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def smart_datetime_cast(series):
    try:
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    except Exception:
        return pd.to_datetime(series, errors="coerce")

def risk_label(score):
    if score >= 80: return "Crítico"
    if score >= 60: return "Alto"
    if score >= 40: return "Medio"
    if score >= 20: return "Bajo"
    return "Muy Bajo"

def show_score(score, title="Riesgo agregado"):
    col1, col2 = st.columns([1,2])
    col1.metric(title, f"{score:.0f}/100", delta=None)
    pct = min(max(float(score),0.0),100.0)/100.0
    bar = "█" * int(pct*25) + "░" * (25-int(pct*25))
    col2.code(f"[{bar}] {score:.0f} ({risk_label(score)})")

def methodology_box(text):
    with st.expander("Metodología aplicada (cómo y por qué)"):
        st.markdown(text)

def recommendations_box(items):
    with st.expander("Recomendaciones"):
        st.markdown("\n".join([f"- {i}" for i in items]))

def explain_findings(df, empty_msg="Sin observaciones."):
    if df is None or df.empty:
        st.success(empty_msg)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.warning(f"Se encontraron **{len(df)}** observaciones.")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar hallazgos (CSV)", data=csv, file_name="hallazgos.csv")

st.title("🔍 Suite de Auditoría Asistida por Computadora (CAAT 1–5)")
st.caption("Sube tus bases (CSV/XLSX), ajusta parámetros y analiza en vivo.")

# Aquí irían todas las secciones CAAT 1 a CAAT 5 como en la versión previa...

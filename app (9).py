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

# ========================== CAAT 1 ==========================
st.header("🕒 CAAT 1 – Validación de registros fuera de horario")
f1 = st.file_uploader("Log de actividades (CSV/XLSX)", type=["csv","xlsx"], key="c1_file")
df1 = read_any(f1)

if df1 is not None:
    df1 = normalize_cols(df1)
    st.dataframe(df1.head(20), use_container_width=True)
    cols1 = df1.columns.tolist()

    c1a, c1b, c1c = st.columns(3)
    c1_user = c1a.selectbox("Columna Usuario", cols1)
    c1_dt   = c1b.selectbox("Columna Fecha/Hora", cols1)
    c1_act  = c1c.selectbox("Columna Acción (opcional)", ["(ninguna)"]+cols1)

    p1a, p1b, p1c, p1d = st.columns(4)
    start_h = p1a.time_input("Inicio jornada", value=time(8,0))
    end_h   = p1b.time_input("Fin jornada", value=time(18,0))
    weekdays_only = p1c.checkbox("Solo días hábiles (L–V)", True)
    rango = p1d.slider("Top N reincidentes", 5, 50, 10, help="Muestra top N usuarios con más eventos fuera de horario")

    work = df1[[c1_user, c1_dt] + ([] if c1_act=="(ninguna)" else [c1_act])].copy()
    work.rename(columns={c1_user:"user", c1_dt:"dt", c1_act:"action" if c1_act!="(ninguna)" else c1_act}, inplace=True)
    work["dt"] = smart_datetime_cast(work["dt"])
    work["weekday"] = work["dt"].dt.weekday

    # Comparación en minutos desde medianoche (evita TypeError)
    work["dt_mins"] = (work["dt"].dt.hour.fillna(-1)*60 + work["dt"].dt.minute.fillna(0)).astype(int)
    start_m = start_h.hour*60 + start_h.minute
    end_m   = end_h.hour*60 + end_h.minute
    if end_m >= start_m:
        in_schedule = (work["dt_mins"] >= start_m) & (work["dt_mins"] <= end_m)
    else:
        in_schedule = (work["dt_mins"] >= start_m) | (work["dt_mins"] <= end_m)

    if weekdays_only:
        in_schedule &= work["weekday"].between(0,4)

    work["fuera_horario"] = ~in_schedule
    out = work[work["fuera_horario"]].copy().sort_values("dt")

    total = len(work)
    fuera = len(out)
    pct = (fuera/total*100) if total else 0.0
    sc = min(100.0, pct*1.2)

    m1, m2, m3 = st.columns(3)
    m1.metric("Eventos totales", total)
    m2.metric("Fuera de horario", fuera)
    m3.metric("% fuera de horario", f"{pct:.2f}%")
    show_score(sc, "Riesgo agregado CAAT 1")

    st.subheader("Hallazgos")
    explain_findings(out, "No se detectaron eventos fuera de horario con los parámetros configurados.")

    if not out.empty:
        st.subheader("Usuarios reincidentes (Top N)")
        top = out.groupby("user").size().reset_index(name="eventos_fuera").sort_values("eventos_fuera", ascending=False).head(rango)
        st.dataframe(top, use_container_width=True, hide_index=True)

    methodology_box("""
**Objetivo:** Identificar actividades fuera del horario laboral/turno.  
**Procedimiento:** Conversión a minutos desde medianoche; filtro por días hábiles; listado de eventos fuera de horario y reincidencia por usuario.  
**Riesgo:** Operaciones no autorizadas, uso indebido de credenciales.
""")
    recommendations_box([
        "Validar permisos especiales/turnos con RRHH.",
        "Configurar alertas automáticas por accesos fuera de horario.",
        "Aplicar MFA y cierre automático de sesiones inactivas."
    ])

# ……………………………
# (Aquí siguen las secciones CAAT 2, 3, 4 y 5 con la misma lógica
# que ya te pasé en la versión completa. Puedes copiarlas de ahí
# y pegarlas debajo de CAAT 1 para tener la app entera.)
# ……………………………

st.markdown("---")
st.caption("© 2025 – Proyecto académico. Esta app muestra el análisis dentro de la web con controles interactivos.")

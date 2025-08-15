
# app.py
# =========================
# ASISTEMAS A2025 - CAATs unificados (1 a 5)
# Autoría: Proyecto de Andrea (empresa logística)
# Requisitos: streamlit, pandas, numpy (NO usa matplotlib)
# =========================

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from datetime import time

st.set_page_config(page_title="ASISTEMAS A2025 – CAAT Suite", layout="wide")

# --------- Utilidades generales ---------
def read_any(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, encoding="utf-8", sep=None, engine="python")
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        st.error("Formato no soportado. Sube CSV o XLSX.")
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
    pct = min(max(score,0),100)/100
    bar = "█" * int(pct*25) + "░" * (25-int(pct*25))
    col2.code(f"[{bar}] {score:.0f} ({risk_label(score)})")

def section_header(title, emoji="🧪"):
    st.markdown(f"### {emoji} {title}")

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

# --------- Sidebar / Navegación ---------
st.title("🔎 Suite de Auditoría Asistida (CAAT 1–5) – Empresa Logística")
st.caption("Todos los informes se visualizan en la app. Sube tus bases (CSV/XLSX) y ajusta los parámetros.")

tool = st.sidebar.radio(
    "Selecciona la prueba CAAT:",
    [
        "CAAT 1 – Registros fuera de horario",
        "CAAT 2 – Auditoría de privilegios (roles críticos y SoD)",
        "CAAT 3 – Conciliación logs vs. transacciones",
        "CAAT 4 – Variación inusual de pagos a proveedores",
        "CAAT 5 – Criterios de selección de proveedores",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Consejo:** Si tus columnas tienen nombres distintos, usa el mapeo de columnas en cada prueba.")

# =====================================================================================
# CAAT 1 – Registros modificados fuera de horario laboral
# =====================================================================================
if tool.startswith("CAAT 1"):
    section_header("CAAT 1 – Validación de registros fuera de horario", "🕒")
    st.write("Sube tu log de actividades (p. ej., **Para CAAT 1.csv** o **Muestra CAAT 1.csv**).")

    f = st.file_uploader("Log de actividades (CSV/XLSX)", type=["csv","xlsx"], key="c1_log")
    df = read_any(f)
    if df is not None:
        df = normalize_cols(df)
        st.markdown("**Vista rápida (primeras filas):**")
        st.dataframe(df.head(20), use_container_width=True)

        # Mapeo de columnas
        st.subheader("Mapeo de columnas")
        cols = df.columns.tolist()
        c_user = st.selectbox("Columna de usuario", cols, index=min(0,len(cols)-1))
        c_dt   = st.selectbox("Columna de fecha/hora", cols, index=min(1,len(cols)-1))
        c_act  = st.selectbox("Columna de acción/evento (opcional)", ["(ninguna)"]+cols)

        # Parámetros
        st.subheader("Parámetros de horario laboral")
        colA, colB, colC = st.columns(3)
        start_h = colA.time_input("Inicio de jornada", value=time(8,0))
        end_h   = colB.time_input("Fin de jornada", value=time(18,0))
        weekdays_only = colC.checkbox("Solo días hábiles (L–V)", True)

        # Procesamiento
        work = df[[c_user, c_dt] + ([] if c_act=="(ninguna)" else [c_act])].copy()
        work.rename(columns={c_user:"user", c_dt:"dt", c_act:"action" if c_act!="(ninguna)" else c_act}, inplace=True)
        work["dt"] = smart_datetime_cast(work["dt"])
        work["hour"] = work["dt"].dt.time
        work["weekday"] = work["dt"].dt.weekday

        in_schedule = (work["hour"] >= start_h) & (work["hour"] <= end_h)
        if weekdays_only:
            in_schedule &= work["weekday"].between(0,4)

        work["fuera_horario"] = ~in_schedule
        out = work[work["fuera_horario"]].copy().sort_values("dt")

        # Métricas
        total = len(work)
        fuera = len(out)
        pct = (fuera/total*100) if total else 0
        sc = min(100, pct*1.2)

        c1, c2, c3 = st.columns(3)
        c1.metric("Eventos totales", total)
        c2.metric("Fuera de horario", fuera)
        c3.metric("% fuera de horario", f"{pct:.2f}%")

        show_score(sc, "Riesgo agregado CAAT 1")
        st.subheader("Hallazgos")
        explain_findings(out, "No se detectaron eventos fuera de horario con los parámetros configurados.")

        methodology_box("**Objetivo:** Identificar actividades registradas fuera del horario laboral/turno definido.")
        recommendations_box(["Verificar con RRHH", "Implementar alertas", "Reforzar MFA"])

# =====================================================================================
# CAAT 2 – Auditoría de privilegios
# =====================================================================================
elif tool.startswith("CAAT 2"):
    section_header("CAAT 2 – Auditoría de privilegios de usuario (Roles críticos y SoD)", "🛡️")
    st.write("Sube tu **matriz de usuarios y roles** (p. ej., **Para CAAT 2.xlsx**).")
    # Resto del código como en la versión larga anterior...

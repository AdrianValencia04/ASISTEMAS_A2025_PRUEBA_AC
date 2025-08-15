import pandas as pd
import streamlit as st
import numpy as np
from io import BytesIO

st.set_page_config(page_title="CAATs – Auditoría Asistida por Computadora", layout="wide")
st.title("📝 Herramienta de Auditoría Asistida por Computadora (CAAT)")

uploaded_file = st.file_uploader("📂 Subir archivo CSV o Excel", type=["csv", "xlsx"])

def to_excel(df_):
    """Exportar DataFrame a Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_.to_excel(writer, index=False, sheet_name="Hallazgos")
    return output.getvalue()

if uploaded_file is not None:
    # Leer archivo CSV o Excel
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.subheader("📄 Vista previa de los datos")
    st.dataframe(df.head())

    # -------------------
    # CAAT 1 – Fuera de horario
    # -------------------
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Hora"] = df["Timestamp"].dt.hour
        df["DiaSemana"] = df["Timestamp"].dt.dayofweek
        df["Fuera_Horario"] = (df["Hora"] < 8) | (df["Hora"] > 18) | (~df["DiaSemana"].isin(range(5)))
        st.subheader("🕒 Registros fuera de horario")
        st.dataframe(df[["Usuario","Timestamp","Fuera_Horario"]])

    # -------------------
    # CAAT 2 – Privilegios conflictivos
    # -------------------
    if "Usuario" in df.columns and "Módulo" in df.columns:
        df_conf = df.groupby("Usuario")["Módulo"].apply(set).reset_index()
        df_conf["Conflicto"] = df_conf["Módulo"].apply(lambda x: "Compras" in x and "Tesorería" in x)
        st.subheader("⚠️ Usuarios con accesos conflictivos")
        st.dataframe(df_conf)

    # -------------------
    # CAAT 3 – Conciliación de registros
    # -------------------
    if "N_OC" in df.columns and "Timestamp" in df.columns:
        df["_merge"] = np.where(df.duplicated(subset="N_OC", keep=False), "Duplicado","Unico")
        st.subheader("🔄 Conciliación de registros")
        st.dataframe(df[["N_OC","_merge"]])

    # -------------------
    # CAAT 4 – Variación de pagos
    # -------------------
    if "Proveedor" in df.columns and "Monto" in df.columns and "Fecha_Pago" in df.columns:
        df["Fecha_Pago"] = pd.to_datetime(df["Fecha_Pago"])
        df["Mes"] = df["Fecha_Pago"].dt.to_period("M")
        pagos_mes = df.groupby(["Proveedor","Mes"])["Monto"].sum().reset_index()
        pagos_mes["Var_Porcentual"] = pagos_mes.groupby("Proveedor")["Monto"].pct_change()
        df_sospechoso = pagos_mes[abs(pagos_mes["Var_Porcentual"]) > 1.0]
        st.subheader("💰 Pagos inusuales detectados")
        st.dataframe(df_sospechoso)

    # -------------------
    # CAAT 5 – Evaluación de proveedores
    # -------------------
    if "Evaluación_Técnica" in df.columns and "Evaluación_Financiera" in df.columns and "Evaluación_Cumplimiento" in df.columns:
        df["Total"] = df[["Evaluación_Técnica","Evaluación_Financiera","Evaluación_Cumplimiento"]].sum(axis=1)
        df_no_aprobado = df[(df["Total"] < 70) & (df["Seleccionado"] == True)]
        st.subheader("❌ Proveedores no aprobados")
        st.dataframe(df_no_aprobado)

    # -------------------
    # Descargar informe completo
    # -------------------
    st.download_button(
        "📥 Descargar Informe",
        data=to_excel(df),
        file_name="Informe_CAAT.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Por favor, sube un archivo CSV o Excel para iniciar el análisis.")

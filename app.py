
import streamlit as st
import pandas as pd
from caat_tool import detectar_facturas_duplicadas, detectar_montos_inusuales, detectar_datos_faltantes

st.set_page_config(page_title="CAAT - Auditoría Asistida", layout="wide")

st.title("CAAT - Herramienta de Auditoría Asistida por Computadora")
st.markdown("Sube un archivo Excel y ejecuta las pruebas de auditoría para detectar duplicados, montos inusuales y datos faltantes.")

# Cargar archivo Excel
uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Vista Previa de los Datos")
    st.dataframe(df)

    # Prueba 1: Facturas duplicadas
    st.subheader("1. Facturas Duplicadas")
    duplicadas = detectar_facturas_duplicadas(df, columna='Factura')
    if duplicadas.empty:
        st.success("No se encontraron facturas duplicadas.")
    else:
        st.warning("Se encontraron facturas duplicadas:")
        st.dataframe(duplicadas)

    # Prueba 2: Montos inusuales
    st.subheader("2. Montos Inusuales (>10,000)")
    inusuales = detectar_montos_inusuales(df, columna='Monto', limite=10000)
    if inusuales.empty:
        st.success("No se encontraron montos inusuales.")
    else:
        st.warning("Se encontraron montos inusuales:")
        st.dataframe(inusuales)

    # Prueba 3: Datos faltantes
    st.subheader("3. Datos Faltantes")
    faltantes = detectar_datos_faltantes(df)
    if faltantes.empty:
        st.success("No se encontraron datos faltantes.")
    else:
        st.warning("Se encontraron datos faltantes:")
        st.dataframe(faltantes)

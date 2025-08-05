
import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="CAAT 2 – Auditoría de Privilegios de Usuario", layout="wide")

st.title("🔐 CAAT 2 – Auditoría de Privilegios de Usuario (Roles Críticos)")

st.markdown("""
Esta herramienta analiza los privilegios asignados a cada usuario para detectar accesos conflictivos,
como tener permisos en **Compras** y **Tesorería** simultáneamente.
""")

# Subir archivo
uploaded_file = st.file_uploader("📂 Cargar archivo Excel/CSV con datos de usuarios y roles", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    else:
        df = pd.read_excel(uploaded_file)
    
    st.subheader("📄 Vista previa de datos")
    st.dataframe(df.head(), use_container_width=True)

    # Validación de conflictos
    usuarios_modulos = df.groupby('Usuario')['Módulo'].apply(set).reset_index()
    usuarios_modulos['Tiene_Conflicto'] = usuarios_modulos['Módulo'].apply(
        lambda x: 'Compras' in x and 'Tesorería' in x
    )

    conflictos = usuarios_modulos[usuarios_modulos['Tiene_Conflicto']]
    df_conflictos = df[df['Usuario'].isin(conflictos['Usuario'])].copy()

    # KPIs
    total_usuarios = df['Usuario'].nunique()
    total_conflictos = conflictos.shape[0]
    porcentaje_conflictos = (total_conflictos / total_usuarios) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total usuarios", total_usuarios)
    col2.metric("Usuarios con conflicto", total_conflictos)
    col3.metric("% con conflicto", f"{porcentaje_conflictos:.1f}%")

    # Detalle de usuarios con conflicto
    st.subheader("🚨 Usuarios con Accesos Conflictivos")
    st.dataframe(df_conflictos, use_container_width=True)

    # Clasificación por criticidad y permisos
    st.subheader("📊 Resumen por Criticidad del Módulo")
    resumen_criticidad = df_conflictos.groupby(['Usuario', 'Criticidad_Módulo'])['Módulo'].count().reset_index()
    st.dataframe(resumen_criticidad, use_container_width=True)

    # Recomendaciones automáticas
    st.subheader("📝 Recomendaciones del Auditor")
    if total_conflictos > 0:
        st.warning("Se detectaron usuarios con acceso simultáneo a Compras y Tesorería. Se recomienda revisar y segregar funciones.")
        st.info("Priorizar la revocación de accesos con criticidad Alta y permisos de 'Total' o 'Aprobación'.")
    else:
        st.success("No se detectaron accesos conflictivos. Mantener la política de segregación de funciones.")

    # Descargar informe
    def to_excel(dataframe):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            dataframe.to_excel(writer, index=False, sheet_name="Conflictos")
        return output.getvalue()

    st.download_button(
        label="📥 Descargar Informe de Conflictos",
        data=to_excel(df_conflictos),
        file_name="Informe_Caat2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Por favor carga un archivo Excel o CSV para iniciar el análisis.")


import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="CAAT 2 – Auditoría de Privilegios de Usuario", layout="wide")

st.title("🔍 CAAT 2 – Auditoría de Privilegios y Segregación de Funciones")

st.markdown("""
Esta herramienta analiza los **roles y permisos de usuarios** en un sistema para detectar posibles conflictos,
riesgos de acceso y violaciones a la segregación de funciones.
""")

uploaded_file = st.file_uploader("📂 Cargar archivo Excel con los datos de usuarios", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("📄 Vista previa de datos")
        st.dataframe(df.head(), use_container_width=True)

        # --- Análisis de accesos conflictivos ---
        conflictos_definidos = [
            ("Compras", "Financiero"),
            ("Recursos Humanos", "Financiero"),
            ("Inventarios", "Comercial"),
        ]

        conflictos_detectados = []
        for usuario in df['Nombre'].unique():
            areas_usuario = set(df[df['Nombre'] == usuario]['Área Funcional'])
            for conflicto in conflictos_definidos:
                if conflicto[0] in areas_usuario and conflicto[1] in areas_usuario:
                    conflictos_detectados.append(usuario)

        df_conflictos = df[df['Nombre'].isin(conflictos_detectados)]

        # --- Usuarios con permisos de alto riesgo ---
        permisos_riesgo = ["Total", "Aprobación"]
        df_permisos_riesgo = df[df['Permisos'].isin(permisos_riesgo)]

        # --- Accesos recientes (últimos 30 días) ---
        fecha_hoy = pd.Timestamp.today().normalize()
        df['Fecha Asignación'] = pd.to_datetime(df['Fecha Asignación'], errors='coerce')
        df_recientes = df[(fecha_hoy - df['Fecha Asignación']).dt.days <= 30]

        # --- Matriz de riesgos ---
        matriz_riesgos = []
        for usuario in df['Nombre'].unique():
            riesgo = "Bajo"
            justificacion = "Accesos dentro de parámetros normales."
            if usuario in conflictos_detectados:
                riesgo = "Alto"
                justificacion = "Posee accesos conflictivos entre áreas críticas."
            elif usuario in df_permisos_riesgo['Nombre'].unique():
                riesgo = "Medio"
                justificacion = "Tiene permisos de alto riesgo en módulos críticos."
            matriz_riesgos.append([usuario, riesgo, justificacion])

        df_matriz = pd.DataFrame(matriz_riesgos, columns=["Usuario", "Nivel de Riesgo", "Observación"])

        # --- Observaciones del auditor ---
        st.subheader("📝 Observaciones del Auditor")
        for _, row in df_matriz.iterrows():
            if row["Nivel de Riesgo"] == "Alto":
                st.error(f"⚠️ {row['Usuario']} es de ALTO riesgo: {row['Observación']}")
            elif row["Nivel de Riesgo"] == "Medio":
                st.warning(f"🔶 {row['Usuario']} es de riesgo MEDIO: {row['Observación']}")
            else:
                st.success(f"✅ {row['Usuario']} es de riesgo BAJO: {row['Observación']}")

        # --- Mostrar matrices ---
        st.subheader("📊 Matriz de Riesgos")
        st.dataframe(df_matriz, use_container_width=True)

        st.subheader("⚠️ Usuarios con Accesos Conflictivos")
        st.dataframe(df_conflictos, use_container_width=True)

        st.subheader("🔐 Permisos de Alto Riesgo")
        st.dataframe(df_permisos_riesgo, use_container_width=True)

        st.subheader("🆕 Accesos Recientes (últimos 30 días)")
        st.dataframe(df_recientes, use_container_width=True)

        # --- Generar informe Excel ---
        def generar_informe():
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name="Base de Datos", index=False)
                df_matriz.to_excel(writer, sheet_name="Matriz de Riesgos", index=False)
                df_conflictos.to_excel(writer, sheet_name="Accesos Conflictivos", index=False)
                df_permisos_riesgo.to_excel(writer, sheet_name="Permisos Alto Riesgo", index=False)
                df_recientes.to_excel(writer, sheet_name="Accesos Recientes", index=False)
            return output.getvalue()

        st.download_button(
            label="📥 Descargar Informe en Excel",
            data=generar_informe(),
            file_name="Informe_Auditoria_Caat2.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

else:
    st.info("Por favor carga un archivo para iniciar el análisis.")

st.markdown("---")
st.caption("Desarrollado por Andrea – Proyecto CAAT Auditoría de Sistemas")

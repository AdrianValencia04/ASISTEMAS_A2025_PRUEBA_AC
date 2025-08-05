
import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Auditoría de Sistemas – Registros Fuera de Horario", layout="wide")

st.title("🖥️ Informe de Auditoría de Sistemas – Registros Fuera de Horario Laboral")

st.markdown("""
Esta herramienta realiza una **auditoría de sistemas** sobre un archivo Excel con registros de actividad para identificar operaciones realizadas fuera del horario laboral.
""")

# Subida de archivo Excel
uploaded_file = st.file_uploader("📂 Cargar archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Leer Excel
        df = pd.read_excel(uploaded_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Hora'] = df['Timestamp'].dt.hour
        df['Día'] = df['Timestamp'].dt.dayofweek  # 0 = Lunes, 6 = Domingo

        # Filtrar fuera de horario
        fuera_horario = df[(df['Hora'] < 7) | (df['Hora'] > 18) | (df['Día'] >= 5)]

        # Conteo por usuario
        conteo_usuario = fuera_horario['Usuario'].value_counts().reset_index()
        conteo_usuario.columns = ['Usuario', 'Registros_Fuera_Horario']

        # Resumen general
        total_registros = len(df)
        total_fuera = len(fuera_horario)
        porcentaje_fuera = round((total_fuera / total_registros) * 100, 2)

        # Mostrar análisis en pantalla
        st.subheader("📊 Resumen Ejecutivo")
        st.write(f"**Fecha del Informe:** {datetime.now().strftime('%d/%m/%Y')}")
        st.write(f"- Total de registros analizados: **{total_registros}**")
        st.write(f"- Total de registros fuera de horario: **{total_fuera}**")
        st.write(f"- Porcentaje fuera de horario: **{porcentaje_fuera}%**")

        st.subheader("📌 Hallazgos Relevantes")
        if total_fuera > 0:
            st.write("- Se identificaron registros fuera del horario laboral establecido.")
            st.write("- Existe riesgo potencial de accesos no autorizados o manipulación de datos fuera de supervisión.")
            st.write("- Se detectaron usuarios con múltiples incidencias recurrentes.")
        else:
            st.success("No se encontraron registros fuera de horario. Cumplimiento satisfactorio.")

        st.subheader("⚠️ Usuarios con mayor cantidad de incidencias")
        st.dataframe(conteo_usuario, use_container_width=True)

        st.subheader("🧠 Análisis y Recomendaciones")
        if total_fuera > 0:
            st.write("""
            - **Control de accesos:** Implementar restricciones de acceso en horarios no laborales.
            - **Monitoreo continuo:** Configurar alertas automáticas cuando se registren operaciones fuera de horario.
            - **Segregación de funciones:** Revisar que las operaciones críticas no sean ejecutadas por usuarios con múltiples permisos.
            - **Registro y trazabilidad:** Asegurar que todos los eventos queden registrados en logs inalterables.
            """)
        else:
            st.write("- Se recomienda mantener las políticas de control actuales y continuar con auditorías periódicas.")

        # Generar informe Excel
        def generar_informe_excel():
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name="Datos Originales", index=False)
                fuera_horario.to_excel(writer, sheet_name="Registros Fuera Horario", index=False)
                conteo_usuario.to_excel(writer, sheet_name="Conteo por Usuario", index=False)
                pd.DataFrame({
                    "Fecha del Informe": [datetime.now().strftime('%d/%m/%Y')],
                    "Total Registros": [total_registros],
                    "Registros Fuera de Horario": [total_fuera],
                    "Porcentaje Fuera de Horario": [f"{porcentaje_fuera}%"]
                }).to_excel(writer, sheet_name="Resumen Ejecutivo", index=False)
            return output.getvalue()

        st.download_button(
            label="📥 Descargar Informe en Excel",
            data=generar_informe_excel(),
            file_name="Informe_Auditoria_Sistemas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

else:
    st.info("Por favor carga un archivo para iniciar la auditoría.")

st.markdown("---")
st.caption("Desarrollado por Andrea – Proyecto CAAT Auditoría de Sistemas")

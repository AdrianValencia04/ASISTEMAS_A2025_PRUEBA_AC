
import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="CAAT 1 – Auditoría de Registros Fuera de Horario", layout="wide")

st.title("🕒 CAAT 1 – Auditoría de Registros Fuera de Horario Laboral")

st.markdown("""
Esta herramienta analiza un archivo Excel con registros de actividad para detectar
operaciones realizadas fuera del horario laboral (antes de las 07:00, después de las 18:00 o fines de semana).

**Formato esperado del archivo:**
- Columnas: `Usuario`, `Acción`, `Timestamp`, `Tipo_Modificación`
- Archivo Excel `.xlsx`
""")

uploaded_file = st.file_uploader("📂 Carga el archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Hora'] = df['Timestamp'].dt.hour
        df['Día'] = df['Timestamp'].dt.dayofweek

        fuera_horario = df[(df['Hora'] < 7) | (df['Hora'] > 18) | (df['Día'] >= 5)]
        conteo_por_usuario = fuera_horario['Usuario'].value_counts().reset_index()
        conteo_por_usuario.columns = ['Usuario', 'Registros_Fuera_Horario']

        resumen = {
            "Total Registros": len(df),
            "Total Fuera de Horario": len(fuera_horario),
            "Porcentaje Fuera de Horario": round((len(fuera_horario) / len(df)) * 100, 2)
        }

        st.subheader("📊 Resumen General")
        st.write(resumen)

        st.subheader("📄 Registros Fuera de Horario")
        st.dataframe(fuera_horario, use_container_width=True)

        st.subheader("👤 Conteo por Usuario")
        st.dataframe(conteo_por_usuario, use_container_width=True)

        def generar_informe():
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name="Datos Originales", index=False)
                fuera_horario.to_excel(writer, sheet_name="Registros Fuera Horario", index=False)
                conteo_por_usuario.to_excel(writer, sheet_name="Conteo por Usuario", index=False)
                pd.DataFrame([resumen]).to_excel(writer, sheet_name="Resumen", index=False)
            return output.getvalue()

        st.download_button(
            label="📥 Descargar Informe en Excel",
            data=generar_informe(),
            file_name="Informe_Auditoria_Prueba1.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"⚠️ Error al procesar el archivo: {e}")

else:
    st.info("Sube un archivo para iniciar la auditoría.")

st.markdown("---")
st.caption("Desarrollado por Andrea – Proyecto CAAT Auditoría de Sistemas")

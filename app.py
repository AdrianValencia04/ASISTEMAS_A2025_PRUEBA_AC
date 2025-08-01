
import pandas as pd
import streamlit as st
from datetime import time

st.set_page_config(page_title="CAAT 1 – Registros fuera de horario", layout="wide")

st.title("🕒 CAAT 1 – Validación de registros modificados fuera de horario laboral")

st.markdown("""
Este módulo identifica operaciones (INSERT, UPDATE, DELETE) realizadas fuera del horario laboral definido.

- **Horario permitido:** 07:00 a 18:00 de lunes a viernes.
- **Datos esperados:** `registro_actividades.csv` con las columnas:
    - `Usuario`
    - `Acción`
    - `Timestamp`
    - `Tipo_Modificación`
""")

uploaded_file = st.file_uploader("📁 Carga el archivo `registro_actividades.csv`", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hora'] = df['Timestamp'].dt.hour
        df['Día'] = df['Timestamp'].dt.dayofweek  # 0=Lunes, 6=Domingo

        # Condiciones fuera de horario: antes de 07h o después de 18h o fines de semana
        fuera_horario = df[(df['Hora'] < 7) | (df['Hora'] > 18) | (df['Día'] >= 5)]

        st.success(f"✅ Se encontraron {len(fuera_horario)} registros fuera de horario laboral.")
        st.dataframe(fuera_horario, use_container_width=True)

        csv = fuera_horario.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar registros sospechosos", csv, file_name="registros_fuera_horario.csv", mime='text/csv')

    except Exception as e:
        st.error(f"⚠️ Error al procesar el archivo: {e}")

else:
    st.warning("Por favor carga un archivo para iniciar la auditoría.")

st.markdown("---")
st.caption("Desarrollado como parte del proyecto CAAT de auditoría de sistemas – Andrea")

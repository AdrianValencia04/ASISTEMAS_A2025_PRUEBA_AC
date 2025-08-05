
import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="CAAT 1 – Validación Fuera de Horario", layout="wide")

st.title("🕒 CAAT 1 – Validación de Registros Fuera de Horario Laboral")

st.markdown("""
Esta herramienta analiza las operaciones realizadas por los usuarios y detecta aquellas que
se registraron fuera del horario laboral o del turno asignado, evaluando el riesgo asociado.
""")

# Configuración de horario laboral
hora_inicio = 8
hora_fin = 18
dias_habiles = [0, 1, 2, 3, 4]  # Lunes a Viernes

# Subir archivo
uploaded_file = st.file_uploader("📂 Cargar archivo CSV con registros de actividad", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="utf-8")
    st.subheader("📄 Vista previa de datos")
    st.dataframe(df.head(), use_container_width=True)

    # Conversión de fechas
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Hora'] = df['Timestamp'].dt.hour
    df['DiaSemana'] = df['Timestamp'].dt.dayofweek

    # Detectar fuera de horario laboral
    df['Fuera_Horario'] = ((df['Hora'] < hora_inicio) | (df['Hora'] > hora_fin) | (~df['DiaSemana'].isin(dias_habiles)))

    # Comparar con turno asignado
    def validar_turno(row):
        if row['Turno_Asignado'] == 'Mañana':
            return 7 <= row['Hora'] <= 14
        elif row['Turno_Asignado'] == 'Tarde':
            return 14 <= row['Hora'] <= 22
        elif row['Turno_Asignado'] == 'Noche':
            return (row['Hora'] >= 22 or row['Hora'] <= 6)
        return False

    df['Coincide_Turno'] = df.apply(validar_turno, axis=1)
    df['Fuera_Turno'] = ~df['Coincide_Turno']

    # Comparar dispositivo y ubicación
    df['Cambio_Dispositivo'] = df['Dispositivo'] != df['Dispositivo_Habitual']
    # Si en la base hay ubicación habitual se podría comparar

    # Índice de riesgo
    def calcular_riesgo(row):
        score = 0
        if row['Fuera_Horario']: score += 3
        if row['Fuera_Turno']: score += 3
        if row['Severidad'] == 'Alto': score += 4
        if row['Acción'] in ['Eliminación', 'Aprobación']: score += 5
        if row['Cambio_Dispositivo']: score += 2
        if "sin justificación" in str(row['Motivo']).lower(): score += 2
        return score

    df['Puntaje_Riesgo'] = df.apply(calcular_riesgo, axis=1)

    # Clasificación
    def clasificar_riesgo(score):
        if score >= 10:
            return "Alto"
        elif score >= 6:
            return "Medio"
        return "Bajo"

    df['Nivel_Riesgo'] = df['Puntaje_Riesgo'].apply(clasificar_riesgo)

    # KPIs
    total_registros = len(df)
    fuera_horario = df['Fuera_Horario'].sum()
    fuera_tur = df['Fuera_Turno'].sum()
    st.metric("Total registros", total_registros)
    st.metric("Fuera de horario", fuera_horario)
    st.metric("Fuera de turno", fuera_tur)

    # Top reincidencias
    st.subheader("👥 Top usuarios reincidentes")
    reincidencias = df[df['Fuera_Turno']].groupby('Usuario').size().reset_index(name='Incidentes').sort_values(by='Incidentes', ascending=False)
    st.dataframe(reincidencias.head(10), use_container_width=True)

    # Matriz de hallazgos
    st.subheader("📊 Matriz de Hallazgos")
    st.dataframe(df[['Usuario', 'Acción', 'Timestamp', 'Tipo_Modificación', 'Severidad', 'Fuera_Horario', 'Fuera_Turno', 'Cambio_Dispositivo', 'Puntaje_Riesgo', 'Nivel_Riesgo']], use_container_width=True)

    # Sugerencias automáticas
    st.subheader("📝 Sugerencias del Auditor")
    if fuera_horario > 0:
        st.warning("Se detectaron operaciones fuera del horario laboral. Revisar permisos de acceso y políticas de control.")
    if fuera_tur > 0:
        st.warning("Existen acciones realizadas fuera del turno asignado. Evaluar si fueron autorizadas.")
    if df['Nivel_Riesgo'].eq('Alto').sum() > 0:
        st.error("Usuarios con riesgo alto detectados. Requiere atención inmediata.")

    # Descargar informe
    def to_excel(dataframe):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            dataframe.to_excel(writer, index=False, sheet_name="Hallazgos")
        return output.getvalue()

    st.download_button(
        label="📥 Descargar Informe en Excel",
        data=to_excel(df),
        file_name="Informe_Caat1.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Por favor carga un archivo CSV para iniciar el análisis.")

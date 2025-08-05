
import pandas as pd
import streamlit as st
from io import BytesIO

st.set_page_config(page_title="CAAT 1 – Validación Fuera de Horario", layout="wide")

st.title("🕒 CAAT 1 – Validación de Registros Fuera de Horario Laboral")

st.markdown("""
Esta herramienta analiza registros de actividad para detectar operaciones realizadas fuera del horario laboral
o del turno asignado, evaluando el riesgo y ofreciendo recomendaciones.
""")

# Configuración de horario laboral fijo
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

    # Detección fuera de horario
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

    # Cambios de dispositivo
    df['Cambio_Dispositivo'] = df['Dispositivo'] != df['Dispositivo_Habitual']

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

    def clasificar_riesgo(score):
        if score >= 10:
            return "Alto"
        elif score >= 6:
            return "Medio"
        return "Bajo"

    df['Nivel_Riesgo'] = df['Puntaje_Riesgo'].apply(clasificar_riesgo)

    # KPIs con porcentaje
    total_registros = len(df)
    fuera_horario = df['Fuera_Horario'].sum()
    fuera_turno = df['Fuera_Turno'].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total registros", total_registros)
    col2.metric("Fuera de horario", f"{fuera_horario} ({(fuera_horario/total_registros)*100:.1f}%)")
    col3.metric("Fuera de turno", f"{fuera_turno} ({(fuera_turno/total_registros)*100:.1f}%)")

    # Resumen Ejecutivo
    st.subheader("📌 Resumen Ejecutivo")
    resumen = f'''
    - Se analizaron **{total_registros}** registros de actividad.
    - **{fuera_horario}** ({(fuera_horario/total_registros)*100:.1f}%) ocurrieron fuera del horario laboral estándar.
    - **{fuera_turno}** ({(fuera_turno/total_registros)*100:.1f}%) ocurrieron fuera del turno asignado.
    - Usuarios con riesgo **alto** detectados: {df['Nivel_Riesgo'].eq('Alto').sum()}.
    '''
    st.markdown(resumen)

    # Top reincidencias
    st.subheader("👥 Top usuarios reincidentes")
    reincidencias = df[df['Fuera_Turno']].groupby('Usuario').size().reset_index(name='Incidentes').sort_values(by='Incidentes', ascending=False)
    st.dataframe(reincidencias.head(10), use_container_width=True)

    # Matriz de hallazgos
    st.subheader("📊 Matriz de Hallazgos")
    st.dataframe(df[['Usuario', 'Acción', 'Timestamp', 'Tipo_Modificación', 'Severidad',
                     'Fuera_Horario', 'Fuera_Turno', 'Cambio_Dispositivo',
                     'Puntaje_Riesgo', 'Nivel_Riesgo']], use_container_width=True)

    # Observaciones automáticas
    st.subheader("📝 Observaciones del Auditor")
    if fuera_horario > 0:
        st.warning("Se detectaron operaciones fuera del horario laboral. Podrían indicar actividades no autorizadas.")
    if fuera_turno > 0:
        st.warning("Existen registros fuera del turno asignado, revisar políticas de asignación y cumplimiento.")
    if df['Cambio_Dispositivo'].sum() > 0:
        st.info("Se han detectado cambios de dispositivo, lo que puede indicar accesos desde ubicaciones no habituales.")
    if df['Nivel_Riesgo'].eq('Alto').sum() > 0:
        st.error("Usuarios con riesgo alto requieren revisión inmediata y posible bloqueo preventivo.")

    # Conclusiones y recomendaciones
    st.subheader("✅ Conclusiones y Recomendaciones")
    st.markdown("""
    - Reforzar el control de accesos fuera de horario y turno.
    - Revisar permisos para acciones críticas como eliminaciones y aprobaciones.
    - Implementar alertas automáticas ante cambios de dispositivo o ubicación.
    - Capacitar a los usuarios sobre políticas de uso aceptable de sistemas.
    """)

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

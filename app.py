
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="CAAT 2 – Auditoría de Privilegios de Usuario", layout="wide")

st.title("🔐 CAAT 2 – Auditoría de Privilegios de Usuario (Roles Críticos)")

st.markdown("""
Esta herramienta analiza los privilegios asignados a cada usuario para detectar accesos conflictivos,
evaluar su nivel de riesgo y generar recomendaciones.
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

    # Selección de módulos conflictivos
    modulos_unicos = sorted(df['Módulo'].unique())
    modulos_conflictivos = st.multiselect(
        "Seleccione los módulos que considera conflictivos",
        options=modulos_unicos,
        default=["Compras", "Tesorería"]
    )

    # Identificación de conflictos
    usuarios_modulos = df.groupby('Usuario')['Módulo'].apply(set).reset_index()
    usuarios_modulos['Tiene_Conflicto'] = usuarios_modulos['Módulo'].apply(
        lambda x: all(mod in x for mod in modulos_conflictivos)
    )

    conflictos = usuarios_modulos[usuarios_modulos['Tiene_Conflicto']]
    df_conflictos = df[df['Usuario'].isin(conflictos['Usuario'])].copy()

    # Cálculo de puntaje de riesgo
    def calcular_riesgo(row):
        score = 0
        if all(mod in df[df['Usuario'] == row['Usuario']]['Módulo'].unique() for mod in modulos_conflictivos):
            score += 5
        if row['Criticidad_Módulo'] == 'Alta':
            score += 3
        if row['Permisos'] in ['Total', 'Aprobación']:
            score += 2
        if row['En_Revisión'] == 'Sí':
            score += 1
        return score

    df['Puntaje_Riesgo'] = df.apply(calcular_riesgo, axis=1)

    def clasificar_riesgo(score):
        if score >= 8:
            return "Alto"
        elif score >= 5:
            return "Medio"
        return "Bajo"

    df['Nivel_Riesgo'] = df['Puntaje_Riesgo'].apply(clasificar_riesgo)

    # KPIs
    total_usuarios = df['Usuario'].nunique()
    total_conflictos = conflictos.shape[0]
    porcentaje_conflictos = (total_conflictos / total_usuarios) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total usuarios", total_usuarios)
    col2.metric("Usuarios con conflicto", total_conflictos)
    col3.metric("% con conflicto", f"{porcentaje_conflictos:.1f}%")

    # Filtro por nivel de riesgo
    nivel_filtro = st.selectbox("Filtrar por nivel de riesgo", options=["Todos", "Alto", "Medio", "Bajo"])
    if nivel_filtro != "Todos":
        df_filtrado = df[df['Nivel_Riesgo'] == nivel_filtro]
    else:
        df_filtrado = df

    st.subheader("📊 Datos filtrados por nivel de riesgo")
    st.dataframe(df_filtrado, use_container_width=True)

    # Gráficos
    st.subheader("📊 Distribución por criticidad (solo conflictos)")
    criticidad_counts = df_conflictos['Criticidad_Módulo'].value_counts()
    if not criticidad_counts.empty:
        fig1, ax1 = plt.subplots()
        ax1.pie(criticidad_counts, labels=criticidad_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

    st.subheader("📊 Top usuarios por puntaje de riesgo")
    top_riesgo = df.groupby('Usuario')['Puntaje_Riesgo'].max().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots()
    top_riesgo.plot(kind='bar', ax=ax2)
    ax2.set_ylabel('Puntaje de Riesgo')
    ax2.set_xlabel('Usuario')
    ax2.set_title('Top 10 Usuarios con Mayor Puntaje de Riesgo')
    st.pyplot(fig2)

    # Observaciones dinámicas
    st.subheader("📝 Observaciones del Auditor")
    if total_conflictos > 0:
        usuarios_altos = df[df['Nivel_Riesgo'] == 'Alto']['Usuario'].unique()
        st.warning(f"Se detectaron usuarios con acceso simultáneo a {', '.join(modulos_conflictivos)}. Esto representa un riesgo elevado de fraude.")
        if len(usuarios_altos) > 0:
            st.error(f"Usuarios de riesgo alto: {', '.join(usuarios_altos)}. Se recomienda su revisión inmediata.")
    else:
        st.success("No se detectaron accesos conflictivos. Mantener las políticas de segregación de funciones.")

    # Recomendaciones
    st.subheader("✅ Recomendaciones")
    st.markdown("""
    - Revocar accesos conflictivos inmediatamente.
    - Segregar funciones críticas entre áreas.
    - Revisar usuarios en riesgo alto de forma prioritaria.
    - Documentar justificaciones para accesos excepcionales.
    """)

    # Función para exportar a Excel
    def to_excel(dataframe):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            dataframe.to_excel(writer, index=False, sheet_name="Datos")
        return output.getvalue()

    # Botones de descarga
    st.download_button(
        label="📥 Descargar solo conflictos",
        data=to_excel(df_conflictos),
        file_name="Conflictos_Caat2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.download_button(
        label="📥 Descargar todos los usuarios con evaluación",
        data=to_excel(df),
        file_name="Evaluacion_Completa_Caat2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Por favor carga un archivo Excel o CSV para iniciar el análisis.")

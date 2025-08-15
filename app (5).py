
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Intentar importar matplotlib y otras librerías para verificar que se hayan instalado
try:
    import matplotlib.pyplot as plt
except ImportError:
    st.error("No se pudo importar matplotlib. Por favor, asegúrese de que todas las dependencias estén instaladas correctamente.")
    st.stop()

# Función para CAAT 1 - Validación de registros fuera de horario laboral
def caat_1(df):
    df['Hora'] = pd.to_datetime(df['Timestamp']).dt.hour
    df['Día'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
    fuera_de_horario = df[(df['Hora'] < 7) | (df['Hora'] > 18) | (df['Día'] >= 5)]
    return fuera_de_horario

# Función para CAAT 2 - Auditoría de privilegios de usuario
def caat_2(df):
    usuarios_con_acceso_dual = df.groupby('Usuario')['Módulo'].apply(set).reset_index()
    usuarios_con_acceso_dual['Tiene_conflicto'] = usuarios_con_acceso_dual['Módulo'].apply(
        lambda x: 'Compras' in x and 'Tesorería' in x)
    conflictos = usuarios_con_acceso_dual[usuarios_con_acceso_dual['Tiene_conflicto']]
    return conflictos

# Función para CAAT 3 - Conciliación entre logs del sistema y transacciones registradas
def caat_3(df_oc, df_logs):
    merged = df_oc.merge(df_logs, on='N_OC', how='left', indicator=True)
    sin_log = merged[merged['_merge'] == 'left_only']
    return sin_log

# Función para CAAT 4 - Análisis de variación inusual de pagos a proveedores
def caat_4(df):
    df['Fecha_Pago'] = pd.to_datetime(df['Fecha_Pago'])
    df['Mes'] = df['Fecha_Pago'].dt.to_period('M')
    pagos_mes = df.groupby(['Proveedor', 'Mes'])['Monto'].sum().reset_index()
    pagos_mes['Var_Porcentual'] = pagos_mes.groupby('Proveedor')['Monto'].pct_change()
    sospechosos = pagos_mes[abs(pagos_mes['Var_Porcentual']) > 1.0]
    return sospechosos

# Función para CAAT 5 - Verificación de criterios de selección de proveedores
def caat_5(df):
    df['Total'] = df[['Evaluación_Técnica', 'Evaluación_Financiera', 'Evaluación_Cumplimiento']].sum(axis=1)
    proveedores_no_aprobados = df[(df['Total'] < 70) & (df['Seleccionado'] == True)]
    return proveedores_no_aprobados

# Streamlit layout setup
st.title("Herramienta de Auditoría Asistida por Computadora (CAAT)")

# Cargar archivos
uploaded_file = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx"])

# Función para leer cualquier archivo
def read_file(uploaded_file):
    if uploaded_file is not None:
        # Detectar el tipo de archivo (CSV o Excel)
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Tipo de archivo no soportado. Solo se aceptan archivos CSV o Excel.")
            return None
    return None

# Leer el archivo cargado por el usuario
df = read_file(uploaded_file)

# Si el archivo se cargó y es válido, ejecutar las pruebas CAAT
if df is not None:
    # Crear pestañas para cada CAAT
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["CAAT 1", "CAAT 2", "CAAT 3", "CAAT 4", "CAAT 5"])

    with tab1:
        # Verificar si el archivo tiene los campos necesarios para CAAT 1
        if 'Timestamp' in df.columns:
            df_caat1_result = caat_1(df)
            st.write("Registros fuera de horario laboral:")
            st.dataframe(df_caat1_result)

            # Graficar los resultados
            st.subheader("Gráfico de registros fuera de horario")
            fig, ax = plt.subplots()
            df_caat1_result['Hora'].value_counts().sort_index().plot(kind='bar', ax=ax)
            ax.set_xlabel('Hora')
            ax.set_ylabel('Número de registros')
            st.pyplot(fig)

    with tab2:
        # Verificar si el archivo tiene los campos necesarios para CAAT 2
        if 'Usuario' in df.columns and 'Módulo' in df.columns:
            df_caat2_result = caat_2(df)
            st.write("Usuarios con acceso conflictivo:")
            st.dataframe(df_caat2_result)

            # Graficar los resultados
            st.subheader("Gráfico de accesos conflictivos")
            fig, ax = plt.subplots()
            df_caat2_result['Usuario'].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel('Usuario')
            ax.set_ylabel('Número de conflictos')
            st.pyplot(fig)

    with tab3:
        # Verificar si el archivo tiene los campos necesarios para CAAT 3
        if 'N_OC' in df.columns and 'Timestamp' in df.columns:
            df_caat3_result = caat_3(df, df)  # Suponiendo que ambos archivos se suben al mismo tiempo
            st.write("Órdenes de compra sin respaldo en logs:")
            st.dataframe(df_caat3_result)

            # Graficar los resultados
            st.subheader("Gráfico de conciliación de logs")
            fig, ax = plt.subplots()
            df_caat3_result['N_OC'].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel('Órdenes de compra sin log')
            ax.set_ylabel('Frecuencia')
            st.pyplot(fig)

    with tab4:
        # Verificar si el archivo tiene los campos necesarios para CAAT 4
        if 'Fecha_Pago' in df.columns and 'Monto' in df.columns:
            df_caat4_result = caat_4(df)
            st.write("Pagos inusuales detectados:")
            st.dataframe(df_caat4_result)

            # Graficar los resultados
            st.subheader("Gráfico de variaciones de pagos")
            fig, ax = plt.subplots()
            df_caat4_result['Var_Porcentual'].plot(kind='bar', ax=ax)
            ax.set_xlabel('Proveedor')
            ax.set_ylabel('Variación porcentual de pagos')
            st.pyplot(fig)

    with tab5:
        # Verificar si el archivo tiene los campos necesarios para CAAT 5
        if 'Evaluación_Técnica' in df.columns and 'Evaluación_Financiera' in df.columns:
            df_caat5_result = caat_5(df)
            st.write("Proveedores no aprobados:")
            st.dataframe(df_caat5_result)

            # Graficar los resultados
            st.subheader("Gráfico de proveedores no aprobados")
            fig, ax = plt.subplots()
            df_caat5_result['Proveedor'].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel('Proveedor')
            ax.set_ylabel('Número de no aprobados')
            st.pyplot(fig)

    # Exportar los resultados
    st.sidebar.subheader("Exportar resultados")
    if st.sidebar.button("Generar Reporte"):
        df.to_csv("reporte_completo.csv", index=False)
        st.sidebar.download_button("Descargar Reporte", "reporte_completo.csv")

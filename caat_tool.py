
import pandas as pd

def detectar_facturas_duplicadas(df, columna='Factura'):
    """Detecta facturas duplicadas en el DataFrame."""
    duplicadas = df[df.duplicated(subset=[columna], keep=False)]
    return duplicadas

def detectar_montos_inusuales(df, columna='Monto', limite=10000):
    """Identifica montos mayores al límite establecido."""
    inusuales = df[df[columna] > limite]
    return inusuales

def conciliacion_reportes(df1, df2, columna='Factura'):
    """Compara dos DataFrames para encontrar facturas que no están en ambos."""
    facturas_1 = set(df1[columna])
    facturas_2 = set(df2[columna])
    solo_en_1 = facturas_1 - facturas_2
    solo_en_2 = facturas_2 - facturas_1
    return solo_en_1, solo_en_2

def revisar_horarios_fuera_rango(df, columna='Hora', inicio='08:00', fin='18:00'):
    """Detecta registros fuera del horario laboral."""
    df['Hora'] = pd.to_datetime(df[columna], format='%H:%M')
    fuera_rango = df[(df['Hora'] < pd.to_datetime(inicio, format='%H:%M')) |
                     (df['Hora'] > pd.to_datetime(fin, format='%H:%M'))]
    return fuera_rango

def detectar_datos_faltantes(df):
    """Muestra registros con datos vacíos."""
    faltantes = df[df.isnull().any(axis=1)]
    return faltantes

# Ejemplo de uso:
if __name__ == "__main__":
    archivo = 'datos_prueba.xlsx'  # Cambiar por el archivo que uses
    try:
        df = pd.read_excel(archivo)
        print("Archivo cargado con éxito. Primeras filas:")
        print(df.head())

        print("\nFacturas duplicadas:")
        print(detectar_facturas_duplicadas(df))

        print("\nMontos inusuales (>10,000):")
        print(detectar_montos_inusuales(df))

        print("\nDatos faltantes:")
        print(detectar_datos_faltantes(df))

    except FileNotFoundError:
        print(f"El archivo {archivo} no fue encontrado.")

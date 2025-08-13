
# CAAT 1 – Validación de Registros Fuera de Horario Laboral

## Descripción
Esta herramienta permite analizar cualquier archivo de actividad (CSV, Excel, TXT, JSON) para detectar operaciones fuera del horario laboral o del turno asignado, priorizarlas por riesgo y generar un informe textual.

## Requisitos
1. Python 3.6+
2. Librerías:
   - pandas
   - streamlit
   - openpyxl
   - xlrd
   - xlsxwriter

## Instalación
1. Clona este repositorio o descarga los archivos.
2. Crea un entorno virtual en Python:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En MacOS/Linux
    .env\Scriptsctivate  # En Windows
    ```
3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso
1. Ejecuta la aplicación con Streamlit:
    ```bash
    streamlit run app.py
    ```
2. Sube un archivo CSV, Excel o JSON con los registros de actividades.
3. Establece los parámetros de análisis, como el horario laboral y los días hábiles.
4. Visualiza el análisis y descarga el informe en formato Excel.

## Contribuciones
Las contribuciones son bienvenidas. Si encuentras algún problema o deseas añadir una nueva característica, abre un issue o un pull request.

## Licencia
Este proyecto es de código abierto bajo la licencia MIT.


# CAAT 1 – Validación de Registros Fuera de Horario

Esta herramienta de auditoría asistida por computadora (CAAT) analiza registros de actividad para detectar operaciones realizadas fuera del horario laboral o del turno asignado.

## Funcionalidades
- Carga de archivo CSV con registros.
- Detección de operaciones fuera de horario y turno.
- Comparación de dispositivo y ubicación.
- Cálculo de índice de riesgo y clasificación.
- Matriz de hallazgos y sugerencias automáticas.
- Descarga de informe en Excel.

## Uso
1. Subir el archivo CSV con los campos requeridos.
2. Revisar los KPIs, tablas y sugerencias en pantalla.
3. Descargar el informe si es necesario.

## Requisitos
Ver `requirements.txt`.

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```


# CAAT 1 – Validación de Registros Fuera de Horario

Esta herramienta CAAT analiza registros de actividad para detectar operaciones realizadas fuera del horario laboral
o del turno asignado, calculando el riesgo y ofreciendo recomendaciones.

## Funcionalidades
- Carga de archivo CSV con registros.
- Detección de operaciones fuera de horario y turno.
- Comparación de dispositivos y ubicación.
- Cálculo de índice de riesgo y clasificación.
- Resumen ejecutivo y observaciones automáticas.
- Matriz de hallazgos y recomendaciones.
- Descarga de informe en Excel.

## Campos requeridos
- Usuario
- Acción
- Timestamp
- Tipo_Modificación
- Severidad
- Dispositivo
- Ubicación
- Motivo
- Turno_Asignado
- Dispositivo_Habitual

## Instalación y uso
```bash
pip install -r requirements.txt
streamlit run app.py
```

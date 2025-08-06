
# CAAT 2 – Auditoría de Privilegios de Usuario (Versión Final Mejorada)

Esta herramienta CAAT detecta accesos conflictivos en usuarios (por defecto Compras y Tesorería),
permite elegir otros módulos, evalúa el riesgo y genera informes descargables.

## Funcionalidades
- Selección de módulos conflictivos desde la interfaz.
- Filtro interactivo por nivel de riesgo.
- Observaciones dinámicas que nombran a usuarios críticos.
- Exportación de dos informes: solo conflictos y evaluación completa.
- Gráficos de distribución por criticidad y top usuarios por riesgo.

## Campos requeridos
- Usuario
- Rol
- Módulo
- Permisos
- Fecha_Asignación
- Criticidad_Módulo
- En_Revisión

## Instalación y uso
```bash
pip install -r requirements.txt
streamlit run app.py
```

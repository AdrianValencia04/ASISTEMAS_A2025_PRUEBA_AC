
# CAAT 2 – Auditoría de Privilegios de Usuario (Roles Críticos)

Esta herramienta de auditoría asistida por computadora (CAAT) permite analizar y detectar accesos conflictivos en los privilegios asignados a usuarios.

## Funcionalidades
- Carga de archivo Excel o CSV con datos de usuarios, roles, módulos y permisos.
- Identificación de usuarios con acceso simultáneo a módulos críticos (Compras y Tesorería).
- KPIs y métricas clave.
- Resumen por criticidad del módulo.
- Recomendaciones automáticas.
- Descarga de informe en Excel.

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

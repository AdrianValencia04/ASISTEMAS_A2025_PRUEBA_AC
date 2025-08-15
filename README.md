
# Herramienta de Auditoría Asistida por Computadora (CAAT)

Esta herramienta está diseñada para realizar auditorías utilizando el enfoque **Computer-Assisted Audit Tool (CAAT)**. Permite analizar archivos de auditoría y generar reportes detallados sobre los procesos de la empresa logística.

## Requisitos

Para ejecutar la app, asegúrese de tener las siguientes dependencias instaladas:

- streamlit==1.9.0
- pandas==1.5.0
- numpy==1.23.2
- matplotlib==3.6.0
- openpyxl==3.0.10

Puede instalar las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

## Uso

1. **Ejecutar la app**: 
   ```bash
   streamlit run app.py
   ```

2. **Subir los archivos**:
   En la página de la app, podrá subir archivos en formato **CSV** o **Excel** para realizar las auditorías de los 5 CAATs.

3. **Resultados**:
   La app generará un reporte y gráficos interactivos que le ayudarán a visualizar los resultados de las auditorías, permitiendo la verificación de registros fuera de horario, privilegios de usuarios, conciliación de logs, pagos inusuales y la verificación de proveedores.

4. **Exportación**:
   Podrá descargar los resultados generados en formato CSV.

## Estructura del Proyecto

El proyecto se divide en cinco pruebas CAAT:
1. **CAAT 1**: Validación de registros modificados fuera de horario laboral.
2. **CAAT 2**: Auditoría de privilegios de usuario (Roles críticos).
3. **CAAT 3**: Conciliación entre logs del sistema y transacciones registradas.
4. **CAAT 4**: Análisis de variación inusual de pagos a proveedores.
5. **CAAT 5**: Verificación de criterios de selección de proveedores.

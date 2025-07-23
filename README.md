
# CAAT – Herramienta de Auditoría Asistida por Computadora

Este proyecto consiste en el desarrollo de una **Herramienta de Auditoría Asistida por Computadora (CAAT)** en Python, diseñada para realizar análisis automáticos sobre datos contables o transaccionales con el objetivo de detectar posibles errores, fraudes o inconsistencias.

---

## **Funciones Implementadas**
La herramienta `caat_tool.py` incluye las siguientes pruebas de auditoría:
1. **Detección de Facturas Duplicadas** – Busca registros con números de factura repetidos.
2. **Detección de Montos Inusuales** – Identifica registros con montos superiores a un límite (por defecto $10.000).
3. **Conciliación de Reportes** – Compara dos archivos (por ejemplo, facturación vs. contabilidad) y muestra las diferencias.
4. **Revisión de Horarios Fuera de Rango** – Detecta registros realizados fuera del horario laboral (por defecto 08:00 - 18:00).
5. **Detección de Datos Faltantes** – Localiza registros con valores vacíos o nulos.

---

## **Estructura del Proyecto**
- `caat_tool.py` – Código principal con las pruebas de auditoría.
- `datos_prueba.xlsx` – Archivo Excel con datos de ejemplo para las pruebas.
- `requirements.txt` – Lista de librerías necesarias (`pandas`, `openpyxl`).
- `README.md` – Este archivo de documentación.

---

## **Requisitos**
- Python 3.8 o superior.
- Librerías:
  ```bash
  pip install pandas openpyxl
  ```

---

## **Uso**
1. Clonar el repositorio o descargar los archivos `caat_tool.py` y `datos_prueba.xlsx`.
2. Colocar ambos archivos en la misma carpeta.
3. Ejecutar el script con:
   ```bash
   python caat_tool.py
   ```
4. Revisar en la terminal la salida de:
   - Facturas duplicadas.
   - Montos inusuales.
   - Datos faltantes.
5. Para la conciliación de reportes, carga dos DataFrames en Python y usa:
   ```python
   from caat_tool import conciliacion_reportes
   conciliacion_reportes(df1, df2, columna='Factura')
   ```

---

## **Ejemplo de Datos de Prueba**
El archivo `datos_prueba.xlsx` contiene datos como:
| Factura | Cliente    | Monto  | Hora  |
|---------|------------|--------|-------|
| 1001    | Empresa A  | 9500   | 07:45 |
| 1002    | Empresa B  | 15000  | 09:30 |
| 1002    | Empresa B  | 15000  | 19:10 |
| 1010    | **NULO**   | 3000   | 20:15 |

---

## **Autores**
- Proyecto académico desarrollado como parte del curso de Auditoría de Sistemas.

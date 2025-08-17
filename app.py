# ------------------------------------------------------------
# Aprendizaje Colaborativo y Práctico – 2do Parcial
# Suite de Auditoría Asistida por Computadora (CAAT 1–5)
# + Modo Libre para archivos arbitrarios
# ------------------------------------------------------------
# Características:
# - Carga robusta de CSV/XLSX (con mensajes claros si falla).
# - Selectbox con keys únicos (evita StreamlitDuplicateElementId).
# - Normalización de columnas y autodetección de sinónimos.
# - Validaciones previas y ayudas inline (tooltips).
# - Reportes descargables en todos los módulos.
# - Pestaña "📦 Modo Libre" para subir cualquier archivo y analizar
#   con cualquiera de los CAAT (asistente de mapeo).
# ------------------------------------------------------------

import io
import sys
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, time, timedelta

# ==========================
# Configuración general UI
# ==========================
st.set_page_config(
    page_title="Aprendizaje Colaborativo y Práctico – 2do Parcial",
    page_icon="🕵️",
    layout="wide"
)

APP_TITLE = "Aprendizaje Colaborativo y Práctico – 2do Parcial"
APP_SUBTITLE = "Suite de Auditoría Asistida por Computadora (CAAT 1–5)"

# Paletas breves para estados
OK = "🟢"
WARN = "🟡"
ERR = "🔴"

# =====================================
# Utilidades: limpiar y lectura robusta
# =====================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas: minúsculas, sin espacios extras ni tildes."""
    import unicodedata
    def _norm(s):
        s = str(s).strip().lower()
        s = " ".join(s.split())
        s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
        s = s.replace(" ", "_")
        return s
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df

def try_read_csv(file) -> pd.DataFrame | None:
    """Intenta leer CSV con encodings comunes."""
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        try:
            return pd.read_csv(file, encoding="latin-1")
        except Exception:
            return None

def try_read_excel(file, sheet=None) -> tuple[pd.DataFrame | None, list[str] | None, str | None]:
    """
    Intenta preparar ExcelFile con openpyxl.
    Devuelve (df, sheet_names, error_msg). Si sheet es None, no lee data sino solo las hojas.
    """
    try:
        import openpyxl  # noqa: F401 (solo forzar import)
    except Exception:
        return None, None, "Falta el motor de Excel (openpyxl). Instálalo o sube CSV."

    try:
        xls = pd.ExcelFile(file, engine="openpyxl")
    except Exception:
        return None, None, (
            "No se pudo leer el Excel. Verifica que no esté protegido, "
            "que no tenga contraseña y que sea .xlsx válido. También puedes subir CSV."
        )

    if sheet is None:
        return None, xls.sheet_names, None

    try:
        df = xls.parse(sheet_name=sheet)
        return df, xls.sheet_names, None
    except Exception:
        return None, xls.sheet_names, "No se pudo cargar la hoja seleccionada."

def read_table_uploader(label: str, key_prefix: str, help_txt: str = "", accept_multiple: bool = False):
    """
    Cargador robusto universal:
    - Acepta CSV y XLSX.
    - Si es XLSX y tiene varias hojas, pedimos elegir con un selectbox (con key único).
    - Devuelve (dataframes, filenames). Para accept_multiple=False, devuelve una lista de largo 1.
    """
    files = st.file_uploader(
        label, type=["csv", "xlsx"], accept_multiple_files=accept_multiple,
        help=help_txt, key=f"{key_prefix}_uploader"
    )
    if not files:
        return [], []

    results = []
    names = []

    # Coherencia de claves únicas en selectboxes
    for idx, file in enumerate(files):
        name = file.name
        ext = name.split(".")[-1].lower()

        if ext == "csv":
            df = try_read_csv(file)
            if df is None:
                st.error(f"{ERR} No se pudo leer el CSV '{name}'. Revisa el delimitador/encoding o sube otro archivo.")
                continue
            names.append(name)
            results.append(normalize_columns(df))

        elif ext == "xlsx":
            # listar hojas primero
            _, sheets, err = try_read_excel(file, sheet=None)
            if err:
                st.error(f"{ERR} {err}\nArchivo: {name}")
                continue
            if not sheets:
                st.error(f"{ERR} El Excel '{name}' no tiene hojas visibles.")
                continue

            sheet = st.selectbox(
                "Hoja de Excel",
                options=sheets,
                index=0,
                key=f"{key_prefix}_sheet_{idx}",  # <- clave única
                help="Selecciona la hoja que contiene tus datos"
            )
            file.seek(0)
            df, _, err2 = try_read_excel(file, sheet=sheet)
            if err2:
                st.error(f"{ERR} {err2}\nArchivo: {name}")
                continue

            names.append(f"{name} :: {sheet}")
            results.append(normalize_columns(df))

        else:
            st.warning(f"{WARN} Formato no soportado: {name}. Usa CSV o XLSX.")
            continue

    return results, names

def ensure_datetime(series: pd.Series) -> pd.Series:
    """Convierte a datetime con coerción y muestra advertencia si hay nulos."""
    out = pd.to_datetime(series, errors="coerce")
    nulls = out.isna().mean()
    if nulls > 0:
        st.warning(
            f"{WARN} Algunas fechas no pudieron convertirse (nulos: {nulls:.1%}). "
            "Revisa el formato o mapea otra columna."
        )
    return out

def num_from_any(series: pd.Series) -> pd.Series:
    """Limpia strings de dinero y convierte a número (coerce)."""
    s = series.astype(str).str.replace(r"[^\d,.\-]", "", regex=True)
    # Intento simple: si hay coma y punto, intentamos heurística; si solo coma, se asume decimal con coma.
    # Luego a float con coerce.
    def _to_float(x):
        if x.count(",") > 0 and x.count(".") > 0:
            # heurística: asume separador de miles con punto y decimal con coma
            x = x.replace(".", "").replace(",", ".")
        elif x.count(",") > 0 and x.count(".") == 0:
            x = x.replace(",", ".")
        try:
            return float(x)
        except Exception:
            return np.nan
    return s.map(_to_float)

def robust_zscore(x: pd.Series) -> pd.Series:
    """Z-score robusto usando MAD."""
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series([0]*len(x), index=x.index)
    z = 0.6745 * (x - med) / mad
    return z

# =====================================================
# Autodetección de columnas (sinónimos por cada CAAT)
# =====================================================

SYN = {
    "usuario": ["usuario", "user", "empleado", "login", "account", "id_usuario", "usuario_id", "nombre_usuario"],
    "fecha":   ["timestamp", "fecha", "datetime", "fh_evento", "fecha_registro", "fecha_hora"],
    "accion":  ["accion", "evento", "operacion", "actividad", "actividad_desc", "severidad"],
    "rol":     ["rol", "modulo", "perfil", "permiso", "grupo", "funcion"],
    "critico": ["critico", "es_critico", "critical", "flag_critico", "criticidad", "nivel", "criticidad_modulo"],
    "id":      ["id", "id_transaccion", "documento", "nro_doc", "referencia"],
    "proveedor": ["proveedor", "vendor", "tercero", "ruc", "nit"],
    "monto":   ["monto", "importe", "total", "valor", "debe", "haber", "pago"],
}

def suggest_col(df: pd.DataFrame, targets: list[str]) -> str | None:
    """Devuelve el primer match de sinónimos que exista en el df."""
    cols = list(df.columns)
    for t in targets:
        for syn in SYN.get(t, []):
            if syn in cols:
                return syn
    return None

def map_to_bool(series: pd.Series) -> pd.Series:
    """Mapeo amplio a booleano (sí/true/1/alto/critico...)."""
    s = series.astype(str).str.strip().str.lower()
    true_vals = {"1", "true", "t", "si", "sí", "y", "yes", "ok", "alto", "critico", "crítico"}
    false_vals = {"0", "false", "f", "no", "n", "low", "bajo"}
    out = pd.Series([np.nan]*len(s), index=s.index)
    out[s.isin(true_vals)] = True
    out[s.isin(false_vals)] = False
    # Todo lo demás queda NaN → el usuario puede proveer lista de críticos
    return out

# ============================
# Descargas helper
# ============================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="hallazgos")
    buffer.seek(0)
    return buffer.read()

# =====================================================
# CAAT 1 – Registros fuera de horario
# =====================================================
def module_caat1(df: pd.DataFrame, mapping: dict):
    st.subheader("CAAT 1 – Validación de registros fuera de horario")

    # Validaciones
    user_col = mapping.get("usuario") or suggest_col(df, ["usuario"])
    dt_col   = mapping.get("fecha") or suggest_col(df, ["fecha"])
    action_col = mapping.get("accion") or None

    if not user_col or not dt_col:
        st.error(f"{ERR} Se requieren al menos las columnas de **usuario** y **fecha/hora**.")
        st.info("Usa el mapeo manual si la autodetección no acierta.")
        return

    # Parámetros
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    start_h = c1.selectbox("Inicio jornada", [f"{h:02d}:{m:02d}" for h in range(0,24) for m in (0,15,30,45)], index=32)
    end_h   = c2.selectbox("Fin jornada",    [f"{h:02d}:{m:02d}" for h in range(0,24) for m in (0,15,30,45)], index=72)
    only_workdays = c3.checkbox("Solo días hábiles (L–V)", value=True)
    top_n = c4.slider("Top N reincidentes", 5, 100, 20)

    # Preparación
    work = df[[user_col, dt_col] + ([action_col] if action_col in df.columns else [])].copy()
    work["dt"] = ensure_datetime(work[dt_col])
    work = work.dropna(subset=["dt"])
    work["hour"] = work["dt"].dt.hour + work["dt"].dt.minute/60
    work["weekday"] = work["dt"].dt.weekday  # 0=Lunes

    sh = int(start_h.split(":")[0]) + int(start_h.split(":")[1])/60
    eh = int(end_h.split(":")[0]) + int(end_h.split(":")[1])/60

    in_schedule = (work["hour"] >= sh) & (work["hour"] <= eh)
    if only_workdays:
        in_schedule = in_schedule & (work["weekday"] < 5)

    work["fuera_horario"] = ~in_schedule

    total = len(work)
    fuera = work["fuera_horario"].sum()
    pct = 0 if total == 0 else (fuera/total)*100

    st.metric("Eventos totales", f"{total}")
    st.metric("Fuera de horario", f"{fuera}")
    st.metric("% fuera de horario", f"{pct:.2f}%")

    hall = work[work["fuera_horario"]].copy()
    if not hall.empty:
        st.write("### Hallazgos")
        st.dataframe(hall.head(500), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Descargar CSV (hallazgos)", df_to_csv_bytes(hall), "CAAT1_hallazgos.csv", mime="text/csv")
        c2.download_button("⬇️ Descargar Excel (hallazgos)", df_to_excel_bytes(hall), "CAAT1_hallazgos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("No se detectaron eventos fuera de horario con los parámetros seleccionados.")

# =====================================================
# CAAT 2 – Privilegios (roles críticos) y SoD
# =====================================================
def module_caat2(df: pd.DataFrame, mapping: dict):
    st.subheader("CAAT 2 – Auditoría de privilegios (roles críticos y SoD)")

    user_col = mapping.get("usuario") or suggest_col(df, ["usuario"])
    role_col = mapping.get("rol")     or suggest_col(df, ["rol"])
    crit_col = mapping.get("critico") or suggest_col(df, ["critico"])

    if not user_col or not role_col:
        st.error(f"{ERR} Se requieren al menos las columnas de **usuario** y **rol**.")
        return

    c1, c2 = st.columns([2,1])
    crit_list_txt = c1.text_input(
        "Lista de roles críticos (separados por coma, opcional)",
        placeholder="ADMIN, SUPERUSER, APROBADOR, TESORERIA"
    )
    sod_rules_txt = c2.text_area(
        "Reglas SoD (una por línea, formato ROL_A -> ROL_B)",
        placeholder="REGISTRO_PROVEEDOR -> APROBACION_PAGO"
    )

    base = df[[user_col, role_col] + ([crit_col] if crit_col in df.columns else [])].copy()
    base.columns = ["user", "role"] + (["crit_src"] if crit_col in df.columns else [])
    base["user"] = base["user"].astype(str).str.strip()
    base["role"] = base["role"].astype(str).str.strip()

    # Marcar críticos
    base["is_critical"] = False
    if "crit_src" in base.columns:
        base["is_critical"] = map_to_bool(base["crit_src"])
        base["is_critical"] = base["is_critical"].fillna(False)

    if crit_list_txt.strip():
        cl = [c.strip().lower() for c in crit_list_txt.split(",") if c.strip()]
        base["is_critical"] |= base["role"].str.lower().isin(cl)

    crits = base[base["is_critical"]].copy()
    st.metric("Usuarios únicos", base["user"].nunique())
    st.metric("Roles distintos", base["role"].nunique())
    st.metric("Asignaciones críticas", len(crits))

    if not crits.empty:
        st.write("### Roles críticos detectados")
        st.dataframe(crits.head(500), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ CSV (críticos)", df_to_csv_bytes(crits), "CAAT2_criticos.csv", mime="text/csv")
        c2.download_button("⬇️ Excel (críticos)", df_to_excel_bytes(crits), "CAAT2_criticos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("No se encontraron asignaciones críticas con los criterios actuales.")

    # Reglas SoD
    if sod_rules_txt.strip():
        rules = []
        for line in sod_rules_txt.splitlines():
            if "->" in line:
                a, b = line.split("->", 1)
                a = a.strip()
                b = b.strip()
                if a and b:
                    rules.append((a, b))
        if rules:
            violaciones = []
            # conjunto de roles por usuario
            roles_user = base.groupby("user")["role"].apply(set).reset_index()
            for _, row in roles_user.iterrows():
                user = row["user"]
                rs = {r.strip().lower() for r in row["role"]}
                for a, b in rules:
                    if a.lower() in rs and b.lower() in rs:
                        violaciones.append({"user": user, "rule": f"{a} -> {b}"})
            sod_df = pd.DataFrame(violaciones)
            st.metric("Violaciones SoD", len(sod_df))
            if not sod_df.empty:
                st.write("### Violaciones SoD")
                st.dataframe(sod_df.head(500), use_container_width=True)
                c1, c2 = st.columns(2)
                c1.download_button("⬇️ CSV (SoD)", df_to_csv_bytes(sod_df), "CAAT2_sod.csv", mime="text/csv")
                c2.download_button("⬇️ Excel (SoD)", df_to_excel_bytes(sod_df), "CAAT2_sod.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning(f"{WARN} No se detectaron reglas válidas en el textarea (usa formato ROL_A -> ROL_B).")

# =====================================================
# CAAT 3 – Conciliación de logs vs transacciones
# =====================================================
def simple_time_match(logs: pd.DataFrame, tx: pd.DataFrame, id_col_logs, id_col_tx, dt_col_logs, dt_col_tx, tol_minutes=5):
    """Conciliación por ID exacto y diferencia de tiempo dentro de tolerancia."""
    logs = logs.copy()
    tx   = tx.copy()
    logs["dt"] = ensure_datetime(logs[dt_col_logs])
    tx["dt"]   = ensure_datetime(tx[dt_col_tx])
    logs = logs.dropna(subset=["dt"])
    tx   = tx.dropna(subset=["dt"])

    # Un merge por ID y luego filtrar por delta
    merged = logs.merge(tx, left_on=id_col_logs, right_on=id_col_tx, how="left", suffixes=("_log", "_tx"))
    merged["dt_diff_min"] = np.abs((merged["dt_log"] - merged["dt_tx"]).dt.total_seconds())/60
    matched = merged[merged["dt_diff_min"] <= tol_minutes].copy()
    unmatched_logs = merged[merged["dt_tx"].isna()].copy()

    return matched, unmatched_logs

def module_caat3():
    st.subheader("CAAT 3 – Conciliación de logs vs transacciones")

    st.caption("Sube **dos** archivos: uno de Logs y otro de Transacciones. "
               "Si cada archivo tiene varias hojas, elige la correspondiente. "
               "Se concilia por **ID** y **Fecha/Hora** con una tolerancia en minutos.")

    st.write("#### Logs (CSV/XLSX)")
    logs_list, logs_names = read_table_uploader(
        "Drag and drop file here",
        key_prefix="caat3_logs",
        help_txt="Sube el archivo de LOGS (CSV o XLSX).",
        accept_multiple=False
    )
    st.write("#### Transacciones (CSV/XLSX)")
    tx_list, tx_names = read_table_uploader(
        "Drag and drop file here",
        key_prefix="caat3_tx",
        help_txt="Sube el archivo de TRANSACCIONES (CSV o XLSX).",
        accept_multiple=False
    )

    if not logs_list or not tx_list:
        st.info("Sube ambos archivos para continuar.")
        return

    logs = logs_list[0]
    tx   = tx_list[0]

    # Mapeo sugerido
    id_col_logs = suggest_col(logs, ["id"]) or st.selectbox("ID en Logs", logs.columns, key="caat3_id_logs_fallback")
    id_col_tx   = suggest_col(tx, ["id"])   or st.selectbox("ID en Transacciones", tx.columns, key="caat3_id_tx_fallback")
    dt_col_logs = suggest_col(logs, ["fecha"]) or st.selectbox("Fecha/Hora en Logs", logs.columns, key="caat3_dt_logs_fb")
    dt_col_tx   = suggest_col(tx, ["fecha"])   or st.selectbox("Fecha/Hora en Transacciones", tx.columns, key="caat3_dt_tx_fb")

    c1, _ = st.columns([1,3])
    tol = c1.slider("Tolerancia (minutos)", 0, 120, 5)

    # Validaciones
    for name, col, df in [
        ("ID Logs", id_col_logs, logs), ("ID Transacciones", id_col_tx, tx),
        ("Fecha Logs", dt_col_logs, logs), ("Fecha Transacciones", dt_col_tx, tx)
    ]:
        if col not in df.columns:
            st.error(f"{ERR} La columna **{name}** no existe en el archivo mapeado.")
            return

    matched, unmatched = simple_time_match(logs, tx, id_col_logs, id_col_tx, dt_col_logs, dt_col_tx, tol_minutes=tol)

    st.metric("Coincidencias", len(matched))
    st.metric("Logs sin match", len(unmatched))

    if not matched.empty:
        st.write("### Coincidencias")
        st.dataframe(matched.head(500), use_container_width=True)
        st.download_button("⬇️ CSV (coincidencias)", df_to_csv_bytes(matched), "CAAT3_coincidencias.csv", mime="text/csv")

    if not unmatched.empty:
        st.write("### Logs sin match")
        st.dataframe(unmatched.head(500), use_container_width=True)
        st.download_button("⬇️ CSV (sin_match)", df_to_csv_bytes(unmatched), "CAAT3_sin_match.csv", mime="text/csv")

# =====================================================
# CAAT 4 – Variación inusual de pagos (outliers)
# =====================================================
def module_caat4(df: pd.DataFrame, mapping: dict):
    st.subheader("CAAT 4 – Variación inusual de pagos (outliers)")
    prov_col  = mapping.get("proveedor") or suggest_col(df, ["proveedor"])
    dt_col    = mapping.get("fecha") or suggest_col(df, ["fecha"])
    monto_col = mapping.get("monto") or suggest_col(df, ["monto"])

    if not prov_col or not dt_col or not monto_col:
        st.error(f"{ERR} Se requieren **proveedor**, **fecha** y **monto**.")
        return

    c1, c2 = st.columns(2)
    z_thr = c1.slider("Ajusta el umbral de outliers (|z| robusto)", 2.0, 6.0, 3.5, 0.5)
    top_n = c2.slider("Top N por proveedor (para vista rápida)", 5, 100, 10)

    pay = df[[prov_col, dt_col, monto_col]].copy()
    pay["fecha"] = ensure_datetime(pay[dt_col])
    pay["monto"] = num_from_any(pay[monto_col])
    pay = pay.dropna(subset=["fecha", "monto"])

    pay["z"] = robust_zscore(pay["monto"])
    out = pay[np.abs(pay["z"]) >= float(z_thr)].copy()
    st.metric("Registros", len(pay))
    st.metric("Outliers detectados", len(out))

    if not out.empty:
        st.write("### Outliers detectados")
        st.dataframe(out.head(500), use_container_width=True)
        st.download_button("⬇️ CSV (outliers)", df_to_csv_bytes(out), "CAAT4_outliers.csv", mime="text/csv")
    else:
        st.info("No se detectaron outliers con el umbral actual.")

# =====================================================
# CAAT 5 – Top N de acciones / rarezas
# =====================================================
def module_caat5(df: pd.DataFrame, mapping: dict):
    st.subheader("CAAT 5 – Frecuencias / rarezas")
    user_col   = mapping.get("usuario") or suggest_col(df, ["usuario"])
    action_col = mapping.get("accion")  or suggest_col(df, ["accion"])
    dt_col     = mapping.get("fecha")   or suggest_col(df, ["fecha"])

    if not user_col or not action_col:
        st.error(f"{ERR} Se requieren **usuario** y **acción/evento**.")
        return

    c1, c2 = st.columns(2)
    top_n = c1.slider("Top N", 5, 100, 20)
    if dt_col and dt_col in df.columns:
        # opcionalmente filtrar por rango de fechas
        # aquí podríamos agregar un date_input, lo omitimos para simplicidad
        pass

    agg = df.groupby([user_col, action_col]).size().reset_index(name="conteo")
    top = agg.sort_values("conteo", ascending=False).head(top_n).copy()

    st.metric("Usuarios únicos", agg[user_col].nunique())
    st.metric("Acciones distintas", agg[action_col].nunique())

    st.write("### Top N combinaciones Usuario–Acción")
    st.dataframe(top, use_container_width=True)
    st.download_button("⬇️ CSV (Top N)", df_to_csv_bytes(top), "CAAT5_topN.csv", mime="text/csv")

# =====================================================
# Modo Libre – Archivo arbitrario
# =====================================================
def modo_libre():
    st.subheader("📦 Modo Libre — Sube cualquier archivo")

    st.markdown(
        """
**¿Qué hace esta sección?**  
Aquí puedes subir **cualquier archivo** (CSV o Excel) y la app intentará **reconocer automáticamente**
las columnas necesarias para ejecutar **cualquiera** de los módulos CAAT (1–5).  
Si algo no coincide, te pediremos **mapear** manualmente. Luego podrás **correr el análisis** y
**descargar un reporte** con los hallazgos.

**Pasos:**
1) Sube tu archivo (CSV/XLSX).  
2) Elige el **módulo (CAAT)** que quieres probar.  
3) Ajusta o confirma el **mapeo** de columnas sugerido.  
4) Presiona **“Ejecutar análisis”** para ver métricas y **descargar** evidencias.
        """
    )

    dfs, names = read_table_uploader(
        "Sube tu archivo (CSV/XLSX)",
        key_prefix="free_mode_file",
        help_txt="Si tu Excel tiene varias hojas, elige la correcta.",
        accept_multiple=False
    )
    if not dfs:
        st.info("Sube un archivo para continuar.")
        return

    df = dfs[0]
    st.write("**Vista rápida de columnas:**", ", ".join(list(df.columns)[:30]))

    caat_choice = st.selectbox(
        "¿Qué módulo quieres ejecutar sobre este archivo?",
        ["CAAT 1 – Fuera de horario", "CAAT 2 – Privilegios/SoD",
         "CAAT 3 – Conciliación (requiere segundo archivo)", "CAAT 4 – Outliers pagos", "CAAT 5 – Top N / rarezas"],
        index=0, key="free_mode_caat"
    )

    mapping = {}
    # Armamos UI de mapeo según módulo
    if "CAAT 1" in caat_choice:
        c1, c2, c3 = st.columns(3)
        mapping["usuario"] = c1.selectbox("Columna Usuario", df.columns, index=(df.columns.tolist().index(suggest_col(df, ["usuario"])) if suggest_col(df, ["usuario"]) in df.columns else 0))
        mapping["fecha"]   = c2.selectbox("Columna Fecha/Hora", df.columns, index=(df.columns.tolist().index(suggest_col(df, ["fecha"])) if suggest_col(df, ["fecha"]) in df.columns else 0))
        mapping["accion"]  = c3.selectbox("Columna Acción (opcional)", ["(ninguna)"] + df.columns.tolist())
        if mapping["accion"] == "(ninguna)":
            mapping["accion"] = None
        if st.button("Ejecutar análisis", key="free_mode_run1"):
            module_caat1(df, mapping)

    elif "CAAT 2" in caat_choice:
        c1, c2, c3 = st.columns(3)
        mapping["usuario"] = c1.selectbox("Columna Usuario", df.columns, index=(df.columns.tolist().index(suggest_col(df, ["usuario"])) if suggest_col(df, ["usuario"]) in df.columns else 0))
        mapping["rol"]     = c2.selectbox("Columna Rol", df.columns, index=(df.columns.tolist().index(suggest_col(df, ["rol"])) if suggest_col(df, ["rol"]) in df.columns else 0))
        mapping["critico"] = c3.selectbox("Columna es_crítico (opcional)", ["(ninguna)"] + df.columns.tolist())
        if mapping["critico"] == "(ninguna)":
            mapping["critico"] = None
        if st.button("Ejecutar análisis", key="free_mode_run2"):
            module_caat2(df, mapping)

    elif "CAAT 3" in caat_choice:
        st.info("Para CAAT 3 necesitas **un segundo archivo** (Transacciones) además del que ya subiste como Logs.")
        tx_list, tx_names = read_table_uploader(
            "Sube el archivo de Transacciones",
            key_prefix="free_mode_caat3_tx",
            accept_multiple=False
        )
        if not tx_list:
            return
        logs = df
        tx   = tx_list[0]

        id_col_logs = suggest_col(logs, ["id"]) or st.selectbox("ID en Logs", logs.columns, key="free_id_logs_fb")
        id_col_tx   = suggest_col(tx, ["id"])   or st.selectbox("ID en Transacciones", tx.columns, key="free_id_tx_fb")
        dt_col_logs = suggest_col(logs, ["fecha"]) or st.selectbox("Fecha/Hora en Logs", logs.columns, key="free_dt_logs_fb")
        dt_col_tx   = suggest_col(tx, ["fecha"])   or st.selectbox("Fecha/Hora en Transacciones", tx.columns, key="free_dt_tx_fb")
        tol = st.slider("Tolerancia (minutos)", 0, 120, 5, key="free_tol")
        if st.button("Ejecutar análisis", key="free_mode_run3"):
            matched, unmatched = simple_time_match(logs, tx, id_col_logs, id_col_tx, dt_col_logs, dt_col_tx, tol_minutes=tol)
            st.metric("Coincidencias", len(matched))
            st.metric("Logs sin match", len(unmatched))
            if not matched.empty:
                st.dataframe(matched.head(500), use_container_width=True)
                st.download_button("⬇️ CSV (coincidencias)", df_to_csv_bytes(matched), "CAAT3_coincidencias.csv", mime="text/csv")
            if not unmatched.empty:
                st.dataframe(unmatched.head(500), use_container_width=True)
                st.download_button("⬇️ CSV (sin_match)", df_to_csv_bytes(unmatched), "CAAT3_sin_match.csv", mime="text/csv")

    elif "CAAT 4" in caat_choice:
        c1, c2, c3 = st.columns(3)
        mapping["proveedor"] = c1.selectbox("Columna Proveedor", df.columns, index=(df.columns.tolist().index(suggest_col(df, ["proveedor"])) if suggest_col(df, ["proveedor"]) in df.columns else 0))
        mapping["fecha"]     = c2.selectbox("Columna Fecha", df.columns, index=(df.columns.tolist().index(suggest_col(df, ["fecha"])) if suggest_col(df, ["fecha"]) in df.columns else 0))
        mapping["monto"]     = c3.selectbox("Columna Monto", df.columns, index=(df.columns.tolist().index(suggest_col(df, ["monto"])) if suggest_col(df, ["monto"]) in df.columns else 0))
        if st.button("Ejecutar análisis", key="free_mode_run4"):
            module_caat4(df, mapping)

    else:  # CAAT 5
        c1, c2, c3 = st.columns(3)
        mapping["usuario"] = c1.selectbox("Columna Usuario", df.columns, index=(df.columns.tolist().index(suggest_col(df, ["usuario"])) if suggest_col(df, ["usuario"]) in df.columns else 0))
        mapping["accion"]  = c2.selectbox("Columna Acción/Evento", df.columns, index=(df.columns.tolist().index(suggest_col(df, ["accion"])) if suggest_col(df, ["accion"]) in df.columns else 0))
        mapping["fecha"]   = c3.selectbox("Columna Fecha (opcional)", ["(ninguna)"] + df.columns.tolist())
        if mapping["fecha"] == "(ninguna)":
            mapping["fecha"] = None
        if st.button("Ejecutar análisis", key="free_mode_run5"):
            module_caat5(df, mapping)

# ============================
# Página principal
# ============================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

tabs = st.tabs([
    "🏠 Inicio",
    "⏰ CAAT 1",
    "🛡️ CAAT 2",
    "🔗 CAAT 3",
    "💸 CAAT 4",
    "📊 CAAT 5",
    "📦 Modo Libre"
])

# ------------------
# Tab: Inicio
# ------------------
with tabs[0]:
    st.markdown(
        """
### Bienvenido/a
Esta aplicación te permite **subir tus bases (CSV/XLSX)**, ajustar parámetros y **analizar en vivo** usando
**CAAT 1–5**.  
Activa la casilla de **Mostrar ayudas** cuando tengas dudas en los filtros.

**Consejos para evitar errores:**
- Si subes **Excel**, asegúrate de que no tenga contraseña, que esté cerrado en tu PC y que sea **.xlsx** válido.
- Si aparece *“Falta openpyxl”*, instálalo o sube un **CSV**.
- Si el error dice *“elementos duplicados”*, es porque dos selectbox eran iguales — ya lo evitamos con **keys únicos**.

### ¿Qué hace cada módulo?
- **CAAT 1:** Detecta eventos **fuera del horario** laboral.  
- **CAAT 2:** Revisa **roles críticos** y **violaciones SoD**.  
- **CAAT 3:** Concilia **Logs** vs **Transacciones** por **ID** y **Fecha** con tolerancia.  
- **CAAT 4:** Busca **outliers** en pagos por proveedor con **z robusto**.  
- **CAAT 5:** Muestra **Top N** de combinaciones Usuario–Acción (rareza / frecuencia).

Si tienes un archivo con formato desconocido, usa **📦 Modo Libre**.
        """
    )

# ------------------
# Tab: CAAT 1
# ------------------
with tabs[1]:
    st.markdown("#### Sube tu log de actividades (CSV/XLSX)")
    lst, names = read_table_uploader(
        "Drag and drop file here",
        key_prefix="caat1",
        help_txt="Sube tu base con columnas de usuario y fecha/hora.",
        accept_multiple=False
    )
    if lst:
        df = lst[0]
        module_caat1(df, mapping={})
    else:
        st.info("Sube un archivo para comenzar.")

# ------------------
# Tab: CAAT 2
# ------------------
with tabs[2]:
    st.markdown("#### Usuarios/Roles (CSV/XLSX)")
    lst, names = read_table_uploader(
        "Drag and drop file here",
        key_prefix="caat2",
        help_txt="Sube tu maestro de usuarios/roles.",
        accept_multiple=False
    )
    if lst:
        df = lst[0]
        module_caat2(df, mapping={})
    else:
        st.info("Sube un archivo para comenzar.")

# ------------------
# Tab: CAAT 3
# ------------------
with tabs[3]:
    module_caat3()

# ------------------
# Tab: CAAT 4
# ------------------
with tabs[4]:
    st.markdown("#### Historial de pagos (CSV/XLSX)")
    lst, names = read_table_uploader(
        "Drag and drop file here",
        key_prefix="caat4",
        help_txt="Se requieren columnas de proveedor, fecha y monto.",
        accept_multiple=False
    )
    if lst:
        df = lst[0]
        module_caat4(df, mapping={})
    else:
        st.info("Sube un archivo para comenzar.")

# ------------------
# Tab: CAAT 5
# ------------------
with tabs[5]:
    st.markdown("#### Log / Eventos (CSV/XLSX)")
    lst, names = read_table_uploader(
        "Drag and drop file here",
        key_prefix="caat5",
        help_txt="Se requieren columnas de usuario y acción/evento.",
        accept_multiple=False
    )
    if lst:
        df = lst[0]
        module_caat5(df, mapping={})
    else:
        st.info("Sube un archivo para comenzar.")

# ------------------
# Tab: Modo Libre
# ------------------
with tabs[6]:
    modo_libre()

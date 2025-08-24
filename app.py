# ------------------------------------------------------------
# Aprendizaje Colaborativo y Práctico – 2do Parcial
# Suite CAAT (1–5) + Modo libre
# ------------------------------------------------------------
# ✔ Robustez de cargas: CSV/XLSX (openpyxl), cache en sesión,
#   manejo de errores y mensajes guiados.
# ✔ Claves únicas para widgets → sin StreamlitDuplicateElementId.
# ✔ Autodetección de columnas y fallback manual.
# ✔ Reportes XLSX descargables en todos los módulos.
# ✔ KPIs y pequeñas visualizaciones.
# ✔ CAAT 1: dispositivo/ubicación (asignado vs usado/log) + 08:00–17:30 por defecto.
# ✔ CAAT 2: roles críticos por multiselect + constructor SoD visual.
# ✔ CAAT 3: conciliación bancaria corregida (cálculo de deltas vectorizado).
# ------------------------------------------------------------

import io
from datetime import time
from typing import Union, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------
# Estilo y utilitarios generales
# ------------------------------

st.set_page_config(
    page_title="Aprendizaje Colaborativo y Práctico – 2do Parcial",
    layout="wide",
)

THEME_NOTE = """
<style>
.small-note {font-size:12px; color:#6b7280;}
.badge {background:#f4f6f8; color:#111827; padding:2px 8px; border-radius:6px; font-size:12px; border:1px solid #e5e7eb;}
.badge-red {background:#fee2e2; color:#991b1b;}
.badge-green {background:#e7f9ed; color:#065f46;}
.kpi {background:#f9fafb; padding:10px 14px; border-radius:10px; border:1px solid #e5e7eb;}
.section {padding:8px 12px; border-radius:10px; border:1px dashed #e5e7eb; background:#fcfcfd;}
h3 {margin-top:6px;}
</style>
"""
st.markdown(THEME_NOTE, unsafe_allow_html=True)

# ------------------------------
# Utilidad: exportar XLSX (1 o varias hojas)
# ------------------------------

def make_xlsx(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], sheet_name: str = "Datos") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if isinstance(data, pd.DataFrame):
            data.to_excel(writer, index=False, sheet_name=(sheet_name[:31] or "Datos"))
        else:
            for name, df in data.items():
                safe = str(name)[:31].replace("/", "-").replace("\\", "-").replace(":", "-")
                df.to_excel(writer, index=False, sheet_name=(safe or "Hoja"))
    buffer.seek(0)
    return buffer.getvalue()

# ------------------------------
# Sesión y lectura de archivos
# ------------------------------

def _save_in_session(key: str, file) -> None:
    if file is None:
        return
    st.session_state[key] = {
        "name": file.name,
        "bytes": file.getvalue() if hasattr(file, "getvalue") else file.read(),
    }

def _read_from_session(key: str):
    if key not in st.session_state:
        return None
    info = st.session_state[key]
    name = info["name"]
    data = io.BytesIO(info["bytes"])
    if name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(data, encoding_errors="ignore")
        except Exception:
            data.seek(0)
            df = pd.read_csv(data, sep=";", encoding_errors="ignore")
        return df
    elif name.lower().endswith(".xlsx"):
        with pd.ExcelFile(data, engine="openpyxl") as xls:
            sheet_names = xls.sheet_names
            if len(sheet_names) == 1:
                return pd.read_excel(xls, sheet_name=sheet_names[0], engine="openpyxl")
            else:
                sheet = st.selectbox(
                    "Hoja de Excel",
                    sheet_names,
                    key=f"sheet_{key}",
                    help="Selecciona la hoja que contiene tus datos.",
                )
                return pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
    else:
        return None

def file_uploader_block(title: str, key: str, help_text: str = "", types=("csv", "xlsx")):
    with st.container():
        st.markdown(f"**{title}**")
        file = st.file_uploader(
            "Arrastra y suelta o examina…",
            type=list(types),
            key=f"uploader_{key}",
            help=help_text,
        )
        if file is not None:
            _save_in_session(key, file)

        df = _read_from_session(key)
        if df is None:
            st.info("Sube un archivo para comenzar. Formatos aceptados: CSV o XLSX.")
            return None
        st.success("Archivo listo ✅ (se conserva en memoria para no perderlo tras el rerun).")
        with st.expander("Vista rápida (primeras filas)", expanded=False):
            st.dataframe(df.head(50), use_container_width=True)
        return df

# ------------------------------
# Utilidades de columnas
# ------------------------------

def guess_column(cols, candidates):
    clower = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in clower:
            return cols[clower.index(cand.lower())]
    return None

def choose_column(df, label, candidates, key_suffix, help_text=""):
    cols = list(df.columns)
    guess = guess_column(cols, candidates)
    idx = cols.index(guess) if (guess in cols) else 0
    col = st.selectbox(
        label,
        cols,
        index=idx if idx < len(cols) else 0,
        key=f"sel_{key_suffix}",
        help=help_text,
    )
    return col

def ensure_datetime(series: pd.Series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")

# ------------------------------
# Métricas pequeñas
# ------------------------------

def robust_zscore(series: pd.Series):
    x = series.astype(float).values
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - median) / mad

# ------------------------------
# CAAT 1 – Registros fuera de horario + inconsistencias
# ------------------------------

def module_caat1():
    st.subheader("CAAT 1 – Registros fuera de horario")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. **Sube tu log** (CSV/XLSX).  
2. Selecciona **Usuario** y **Fecha/Hora**.  
3. Define **inicio/fin de jornada** y si deseas **solo días hábiles (L–V)**.  
4. (Opcional) Mapea **dispositivo** y **ubicación** para detectar inconsistencias.  
5. Revisa KPIs y descarga los **hallazgos**.
""")

    df = file_uploader_block("Log de actividades", key="caat1")
    if df is None:
        return

    # Selección de columnas básicas
    c1, c2, c3 = st.columns(3)
    with c1:
        user_col = choose_column(
            df, "Columna Usuario",
            ["usuario", "user", "empleado", "usuario_id", "id_usuario"],
            "caat1_user",
        )
    with c2:
        dt_col = choose_column(
            df, "Columna Fecha/Hora",
            ["timestamp", "fecha_hora", "fecha", "datetime", "dt"],
            "caat1_dt",
        )
    with c3:
        action_col = st.selectbox(
            "Columna Acción (opcional)",
            ["(ninguna)"] + list(df.columns),
            key="caat1_action",
            help="Si la eliges, el reporte la incluirá para contexto.",
        )
        if action_col == "(ninguna)":
            action_col = None

    # Campos opcionales para inconsistencias
    with st.expander("Campos opcionales para inconsistencias (si existen en tu archivo)", expanded=False):
        cD1, cD2 = st.columns(2)
        with cD1:
            dev_assigned = st.selectbox(
                "Dispositivo asignado",
                ["(ninguna)"] + list(df.columns),
                key="caat1_dev_assigned",
                help="Ej.: dispositivo_asignado / equipo_asignado",
            )
        with cD2:
            dev_used = st.selectbox(
                "Dispositivo usado (log)",
                ["(ninguna)"] + list(df.columns),
                key="caat1_dev_used",
                help="Ej.: dispositivo_usado / dispositivo",
            )
        cL1, cL2 = st.columns(2)
        with cL1:
            loc_assigned = st.selectbox(
                "Ubicación asignada (o de trabajo)",
                ["(ninguna)"] + list(df.columns),
                key="caat1_loc_assigned",
                help="Ej.: ubicacion_asignada / ubicacion_de_trabajo",
            )
        with cL2:
            loc_log = st.selectbox(
                "Ubicación del log",
                ["(ninguna)"] + list(df.columns),
                key="caat1_loc_log",
                help="Ej.: ubicacion_log / ubicacion",
            )

    # Parámetros de horario
    c4, c5, c6, c7 = st.columns([1, 1, 1, 2])
    with c4:
        start_h = st.time_input("Inicio de jornada", time(8, 0), key="caat1_start")
    with c5:
        end_h = st.time_input("Fin de jornada", time(17, 30), key="caat1_end")  # 17:30 por defecto
    with c6:
        only_weekdays = st.checkbox("Solo días hábiles (L–V)", value=True, key="caat1_week")
    with c7:
        topn = st.slider("Top N reincidentes", 5, 100, 20, key="caat1_topn")

    # Procesamiento base
    base_cols = [user_col, dt_col] + ([action_col] if action_col else [])
    work = df[base_cols].copy()
    work.rename(columns={user_col: "user", dt_col: "dt"}, inplace=True)
    work["dt"] = ensure_datetime(work["dt"])
    work = work.dropna(subset=["dt"])
    work["weekday"] = work["dt"].dt.weekday  # 0=Lunes
    work["hour"] = work["dt"].dt.hour + work["dt"].dt.minute / 60

    # Nombre del día y flags L–V
    DIAS = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
    work["weekday_name"] = work["weekday"].map(DIAS)
    work["is_weekday"] = work["weekday"].between(0, 4)

    # Añadir columnas opcionales (si el usuario las mapeó)
    extra_cols = []
    if dev_assigned != "(ninguna)" and dev_used != "(ninguna)":
        work["dispositivo_asignado"] = df[dev_assigned].astype(str)
        work["dispositivo_usado"] = df[dev_used].astype(str)
        work["mismatch_dispositivo"] = work["dispositivo_asignado"].ne(work["dispositivo_usado"])
        extra_cols += ["dispositivo_asignado", "dispositivo_usado", "mismatch_dispositivo"]

    if loc_assigned != "(ninguna)" and loc_log != "(ninguna)":
        work["ubicacion_asignada"] = df[loc_assigned].astype(str)
        work["ubicacion_log"] = df[loc_log].astype(str)
        work["mismatch_ubicacion"] = work["ubicacion_asignada"].ne(work["ubicacion_log"])
        extra_cols += ["ubicacion_asignada", "ubicacion_log", "mismatch_ubicacion"]

    # Fuera de horario con rango 08:00–17:30
    in_schedule = (work["hour"] >= (start_h.hour + start_h.minute/60)) & (work["hour"] <= (end_h.hour + end_h.minute/60))
    if only_weekdays:
        in_schedule &= work["is_weekday"]
    work["fuera_horario"] = ~in_schedule

    # KPIs base
    total_events = len(work)
    out_of_hours = int(work["fuera_horario"].sum())
    pct = (out_of_hours / total_events * 100) if total_events else 0

    kcol1, kcol2, kcol3 = st.columns(3)
    kcol1.metric("Eventos totales", f"{total_events:,}")
    kcol2.metric("Fuera de horario", f"{out_of_hours:,}")
    kcol3.metric("% fuera de horario", f"{pct:.2f}%")

    # Hallazgos: fuera de horario
    findings = work[work["fuera_horario"]].copy()
    if not findings.empty:
        top = (
            findings.groupby("user", as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(topn)
        )
        with st.expander("Top reincidentes", expanded=True):
            st.dataframe(top, use_container_width=True)

        cols_show = ["user", "dt", "weekday", "weekday_name"] + ([action_col] if action_col else []) + extra_cols + ["fuera_horario"]
        cols_show = [c for c in cols_show if c in findings.columns]
        with st.expander("Hallazgos (detallado)", expanded=False):
            st.dataframe(findings[cols_show], use_container_width=True)

        st.download_button(
            "⬇️ Descargar hallazgos fuera de horario (XLSX)",
            data=make_xlsx(findings[cols_show], sheet_name="Hallazgos"),
            file_name="CAAT1_fuera_horario.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "⬇️ Descargar Top (XLSX)",
            data=make_xlsx(top, sheet_name="Top"),
            file_name="CAAT1_top_reincidentes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("No se detectaron registros fuera de horario con los parámetros actuales.")

    # Reportes de inconsistencias de dispositivo/ubicación
    st.markdown("---")
    st.markdown("### 🔎 Inconsistencias de dispositivo / ubicación")

    if "mismatch_dispositivo" in work.columns:
        dev_bad = work[work["mismatch_dispositivo"]].copy()
        k1, k2 = st.columns(2)
        with k1:
            st.metric("Registros con dispositivo distinto al asignado", f"{len(dev_bad):,}")
        with k2:
            st.metric("Usuarios únicos con mismatch de dispositivo", f"{dev_bad['user'].nunique():,}")
        with st.expander("Detalle – Dispositivo no asignado", expanded=False):
            cols_dev = ["user", "dt", "dispositivo_asignado", "dispositivo_usado"] + ([action_col] if action_col else [])
            cols_dev = [c for c in cols_dev if c in dev_bad.columns]
            st.dataframe(dev_bad[cols_dev], use_container_width=True)
        st.download_button(
            "⬇️ Descargar inconsistencias de dispositivo (XLSX)",
            data=make_xlsx(dev_bad[cols_dev], sheet_name="Dispositivo"),
            file_name="CAAT1_inconsistencias_dispositivo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Mapea columnas de **dispositivo asignado/usado** para habilitar este reporte.")

    if "mismatch_ubicacion" in work.columns:
        loc_bad = work[work["mismatch_ubicacion"]].copy()
        k1, k2 = st.columns(2)
        with k1:
            st.metric("Registros con ubicación distinta a la asignada", f"{len(loc_bad):,}")
        with k2:
            st.metric("Usuarios únicos con mismatch de ubicación", f"{loc_bad['user'].nunique():,}")
        with st.expander("Detalle – Ubicación distinta", expanded=False):
            cols_loc = ["user", "dt", "ubicacion_asignada", "ubicacion_log"] + ([action_col] if action_col else [])
            cols_loc = [c for c in cols_loc if c in loc_bad.columns]
            st.dataframe(loc_bad[cols_loc], use_container_width=True)
        st.download_button(
            "⬇️ Descargar inconsistencias de ubicación (XLSX)",
            data=make_xlsx(loc_bad[cols_loc], sheet_name="Ubicacion"),
            file_name="CAAT1_inconsistencias_ubicacion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Mapea columnas de **ubicación asignada/log** para habilitar este reporte.")

# ------------------------------
# CAAT 2 – Auditoría de privilegios (roles críticos y SoD)
# ------------------------------

def module_caat2():
    st.subheader("CAAT 2 – Auditoría de privilegios (roles críticos y SoD)")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. Sube el **maestro de Usuarios/Roles**.  
2. Elige **Usuario** y **Rol**.  
3. Marca **Roles críticos** desde la lista (multiselect).  
4. Construye **Reglas SoD** con selectores (ROL_A -> ROL_B).  
""")

    df = file_uploader_block("Usuarios/Roles", key="caat2")
    if df is None:
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        user_col = choose_column(df, "Columna Usuario", ["usuario", "user", "empleado"], "caat2_user")
    with c2:
        role_col = choose_column(df, "Columna Rol", ["rol", "role", "módulo"], "caat2_role")
    with c3:
        crit_col = st.selectbox(
            "Columna es_crítico (opcional)",
            ["(ninguna)"] + list(df.columns),
            key="caat2_crit",
        )
        if crit_col == "(ninguna)":
            crit_col = None

    missing = [c for c in [user_col, role_col] if c not in df.columns]
    if missing:
        st.error(f"No encuentro columnas requeridas: {', '.join(missing)}. Corrige la selección o sube otra base.")
        return

    base = df[[user_col, role_col] + ([crit_col] if crit_col else [])].copy()
    base.rename(columns={user_col: "user", role_col: "role"}, inplace=True)
    if crit_col:
        base["is_critical"] = base[crit_col].astype(str).str.lower().isin(
            ["true", "1", "si", "sí", "x", "y", "yes", "critical", "critico", "crítico"]
        )
    else:
        base["is_critical"] = False

    st.markdown("**Roles críticos (selección múltiple)**")
    all_roles = sorted(base["role"].astype(str).unique())
    extra_crit = st.multiselect(
        "Selecciona roles críticos",
        options=all_roles,
        default=[],
        key="caat2_manualcrit",
        help="Elige uno o más roles que quieras marcar como críticos."
    )
    if extra_crit:
        base["is_critical"] = base["is_critical"] | base["role"].astype(str).isin(extra_crit)

    st.markdown("**Reglas SoD (construye las combinaciones prohibidas)**")
    if "sod_rules" not in st.session_state:
        st.session_state["sod_rules"] = []

    cA, cB, cBtn = st.columns([3, 3, 1])
    with cA:
        rol_a = st.selectbox("ROL_A", options=all_roles, key="caat2_sod_a")
    with cB:
        rol_b = st.selectbox("ROL_B", options=all_roles, key="caat2_sod_b")
    with cBtn:
        if st.button("➕ Agregar", key="caat2_add_rule"):
            if rol_a and rol_b and rol_a != rol_b:
                rule = f"{rol_a} -> {rol_b}"
                if rule not in st.session_state["sod_rules"]:
                    st.session_state["sod_rules"].append(rule)

    if st.session_state["sod_rules"]:
        st.markdown("**Reglas definidas:**")
        for i, r in enumerate(st.session_state["sod_rules"], start=1):
            st.markdown(f"{i}. {r}")

    rules = []
    for line in st.session_state["sod_rules"]:
        if "->" in line:
            a, b = [x.strip().upper() for x in line.split("->", 1)]
            rules.append((a, b))

    by_user = base.groupby("user")["role"].apply(lambda x: set(map(lambda y: str(y).upper(), x))).reset_index()
    conflicts = []
    for _, r in by_user.iterrows():
        roles = r["role"]
        for a, b in rules:
            if a in roles and b in roles:
                conflicts.append({"user": r["user"], "regla": f"{a} -> {b}"})
    conflicts_df = pd.DataFrame(conflicts)

    total_users = base["user"].nunique()
    total_roles = base["role"].nunique()
    crit_roles = int(base[base["is_critical"]]["role"].nunique())

    k1, k2, k3 = st.columns(3)
    k1.metric("Usuarios únicos", f"{total_users:,}")
    k2.metric("Roles únicos", f"{total_roles:,}")
    k3.metric("Roles críticos", f"{crit_roles:,}")

    if not conflicts_df.empty:
        with st.expander("Conflictos SoD detectados", expanded=True):
            st.dataframe(conflicts_df, use_container_width=True)
        st.download_button(
            "⬇️ Descargar conflictos SoD (XLSX)",
            data=make_xlsx(conflicts_df, sheet_name="Conflictos_SoD"),
            file_name="CAAT2_conflictos_SoD.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("No se detectaron conflictos SoD con las reglas actuales.")

    with st.expander("Roles críticos (detalle)", expanded=False):
        crit_df = base[base["is_critical"]].sort_values(["user", "role"])
        st.dataframe(crit_df, use_container_width=True)
        if not crit_df.empty:
            st.download_button(
                "⬇️ Descargar roles críticos (XLSX)",
                data=make_xlsx(crit_df, sheet_name="Roles_criticos"),
                file_name="CAAT2_roles_criticos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ------------------------------
# CAAT 3 – Conciliación de logs vs transacciones (+ bancaria)
# ------------------------------

def module_caat3():
    st.subheader("CAAT 3 – Conciliación de logs vs transacciones")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. Sube **Logs** y **Transacciones**.  
2. En ambos: selecciona **ID** y **Fecha/Hora**.  
3. Define **tolerancia** (minutos) para marcar desfase.  
4. Descarga los **no conciliados** y los **fuera de tolerancia**.

**Opcional:** Conciliación bancaria simple (extracto bancario vs libro).
""")

    c = st.columns(2)
    with c[0]:
        logs = file_uploader_block("Logs (CSV/XLSX)", key="caat3_logs")
    with c[1]:
        txs = file_uploader_block("Transacciones (CSV/XLSX)", key="caat3_txs")

    if logs is None or txs is None:
        return

    st.markdown("### Mapeo de columnas")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        id_logs = choose_column(logs, "ID Logs", ["id", "id_log", "id_transaccion", "folio"], "caat3_idlogs")
    with c2:
        dt_logs = choose_column(logs, "Fecha/Hora Logs", ["fecha", "timestamp", "fecha_hora", "dt", "datetime"], "caat3_dtlogs")
    with c3:
        id_txs = choose_column(txs, "ID Tx", ["id", "id_tx", "id_transaccion", "folio"], "caat3_idtxs")
    with c4:
        dt_txs = choose_column(txs, "Fecha/Hora Tx", ["fecha", "timestamp", "fecha_hora", "dt", "datetime"], "caat3_dttxs")

    tol_min = st.slider("Tolerancia (minutos)", 0, 180, 30, key="caat3_tol")

    logs2 = logs[[id_logs, dt_logs]].copy()
    logs2.rename(columns={id_logs: "id", dt_logs: "dt"}, inplace=True)
    logs2["dt"] = ensure_datetime(logs2["dt"])

    txs2 = txs[[id_txs, dt_txs]].copy()
    txs2.rename(columns={id_txs: "id", dt_txs: "dt"}, inplace=True)
    txs2["dt"] = ensure_datetime(txs2["dt"])

    merged = pd.merge(logs2, txs2, on="id", how="outer", suffixes=("_log", "_tx"))
    missing_log = merged[merged["dt_log"].isna()]
    missing_tx = merged[merged["dt_tx"].isna()]

    both = merged.dropna(subset=["dt_log", "dt_tx"]).copy()
    both["delta_min"] = (both["dt_tx"] - both["dt_log"]).dt.total_seconds().div(60).abs()
    out_tol = both[both["delta_min"] > tol_min]

    k1, k2, k3 = st.columns(3)
    k1.metric("IDs solo en Transacciones", f"{len(missing_log):,}")
    k2.metric("IDs solo en Logs", f"{len(missing_tx):,}")
    k3.metric("Fuera de tolerancia", f"{len(out_tol):,}")

    with st.expander("IDs solo en Transacciones", expanded=False):
        st.dataframe(missing_log, use_container_width=True)
        if not missing_log.empty:
            st.download_button(
                "⬇️ Descargar (XLSX)",
                data=make_xlsx(missing_log, "Solo_en_Tx"),
                file_name="CAAT3_solo_en_tx.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with st.expander("IDs solo en Logs", expanded=False):
        st.dataframe(missing_tx, use_container_width=True)
        if not missing_tx.empty:
            st.download_button(
                "⬇️ Descargar (XLSX)",
                data=make_xlsx(missing_tx, "Solo_en_Logs"),
                file_name="CAAT3_solo_en_logs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with st.expander("Fuera de tolerancia", expanded=True):
        st.dataframe(out_tol, use_container_width=True)
        if not out_tol.empty:
            st.download_button(
                "⬇️ Descargar (XLSX)",
                data=make_xlsx(out_tol, "Fuera_de_Tolerancia"),
                file_name="CAAT3_fuera_tolerancia.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.markdown("---")
    st.markdown("### 🏦 Conciliación bancaria (opcional)")
    st.caption("Paralela y simple: por **monto** y **fecha** con tolerancia en días.")

    cc = st.columns(2)
    with cc[0]:
        bank = file_uploader_block("Extracto bancario", key="caat3_bank")
    with cc[1]:
        book = file_uploader_block("Libro (contabilidad/ERP)", key="caat3_book")

    if bank is not None and book is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            amt_b = choose_column(bank, "Monto (banco)", ["monto", "importe", "amount", "valor"], "caat3_amtb")
        with c2:
            date_b = choose_column(bank, "Fecha (banco)", ["fecha", "date", "f_operacion"], "caat3_dateb")
        with c3:
            amt_c = choose_column(book, "Monto (libro)", ["monto", "importe", "amount", "valor"], "caat3_amtc")

        date_c = choose_column(book, "Fecha (libro)", ["fecha", "date", "f_registro"], "caat3_datec")

        tol_days = st.slider("Tolerancia de fecha (días)", 0, 15, 3, key="caat3_bank_tol")

        b2 = bank[[amt_b, date_b]].copy().rename(columns={amt_b: "amount", date_b: "date"})
        c2_ = book[[amt_c, date_c]].copy().rename(columns={amt_c: "amount", date_c: "date"})
        b2["date"] = pd.to_datetime(b2["date"], errors="coerce").dt.date
        c2_["date"] = pd.to_datetime(c2_["date"], errors="coerce").dt.date
        b2["amount_r"] = b2["amount"].astype(float).round(2)
        c2_["amount_r"] = c2_["amount"].astype(float).round(2)

        matches = []
        used_idx = set()
        for i, br in b2.dropna(subset=["amount_r", "date"]).iterrows():
            # candidatos por monto
            cands = c2_[(c2_["amount_r"] == br["amount_r"]) & ~c2_.index.isin(used_idx)]
            if cands.empty:
                continue
            # diferencia en días (vectorizada y robusta)
            deltas = (pd.to_datetime(cands["date"]) - pd.to_datetime(br["date"])).abs().dt.days
            j = deltas.idxmin()
            if deltas.loc[j] <= tol_days:
                matches.append({"bank_idx": i, "book_idx": j})
                used_idx.add(j)

        matched_b = set([m["bank_idx"] for m in matches])
        matched_c = set([m["book_idx"] for m in matches])

        bank_unmatched = b2[~b2.index.isin(matched_b)]
        book_unmatched = c2_[~c2_.index.isin(matched_c)]

        u1, u2 = st.columns(2)
        with u1:
            st.markdown("**No conciliados (Banco)**")
            st.dataframe(bank_unmatched, use_container_width=True)
            if not bank_unmatched.empty:
                st.download_button(
                    "⬇️ Descargar banco no conciliado (XLSX)",
                    data=make_xlsx(bank_unmatched, "Banco_no_conciliado"),
                    file_name="CAAT3_banco_no_conciliado.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        with u2:
            st.markdown("**No conciliados (Libro)**")
            st.dataframe(book_unmatched, use_container_width=True)
            if not book_unmatched.empty:
                st.download_button(
                    "⬇️ Descargar libro no conciliado (XLSX)",
                    data=make_xlsx(book_unmatched, "Libro_no_conciliado"),
                    file_name="CAAT3_libro_no_conciliado.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

# ------------------------------
# CAAT 4 – Variación inusual de pagos (outliers)
# ------------------------------

def module_caat4():
    st.subheader("CAAT 4 – Variación inusual de pagos (outliers)")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. Sube tu **histórico de pagos**.  
2. Selecciona **Proveedor**, **Fecha** y **Monto**.  
3. Ajusta el **umbral de outliers (|z| robusto)** y descarga hallazgos.
""")

    df = file_uploader_block("Histórico de pagos", key="caat4")
    if df is None:
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        vendor_col = choose_column(df, "Proveedor", ["proveedor", "supplier", "vendor", "cliente"], "caat4_vendor")
    with c2:
        date_col = choose_column(df, "Fecha", ["fecha", "f_pago", "date"], "caat4_date")
    with c3:
        amount_col = choose_column(df, "Monto", ["monto", "importe", "amount", "total"], "caat4_amount")

    work = df[[vendor_col, date_col, amount_col]].copy()
    work.rename(columns={vendor_col: "vendor", date_col: "date", amount_col: "amount"}, inplace=True)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work = work.dropna(subset=["date", "amount"])

    out_list = []
    for v, g in work.groupby("vendor"):
        z = robust_zscore(g["amount"])
        tmp = g.copy()
        tmp["z_robusto"] = z
        out_list.append(tmp)
    workz = pd.concat(out_list).reset_index(drop=True)

    thr = st.slider("Umbral |z| robusto", 2.0, 6.0, 3.5, 0.5, key="caat4_thr")
    outliers = workz[workz["z_robusto"].abs() >= thr]

    k1, k2 = st.columns(2)
    k1.metric("Registros", f"{len(work):,}")
    k2.metric("Outliers", f"{len(outliers):,}")

    with st.expander("Outliers detectados", expanded=True):
        st.dataframe(outliers.sort_values(["vendor", "date"]), use_container_width=True)
        if not outliers.empty:
            st.download_button(
                "⬇️ Descargar outliers (XLSX)",
                data=make_xlsx(outliers, "Outliers"),
                file_name="CAAT4_outliers.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ------------------------------
# CAAT 5 – Duplicados / near-duplicados
# ------------------------------

def module_caat5():
    st.subheader("CAAT 5 – Duplicados / near-duplicados")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. Sube tu **tabla** (pagos, registros, etc.).  
2. Selecciona **columnas clave** y opcionalmente **Fecha** y **Monto**.  
3. Descarga **duplicados** y, si quieres, **near-duplicados** (mismo monto y fecha cercana).
""")

    df = file_uploader_block("Base", key="caat5")
    if df is None:
        return

    st.markdown("**Columnas clave**")
    key_cols = st.multiselect(
        "Selecciona columnas para detectar duplicados exactos",
        list(df.columns),
        default=[c for c in df.columns if c.lower() in ("id", "id_transaccion")],
        key="caat5_keys",
    )

    date_col = st.selectbox("(Opcional) Fecha", ["(ninguna)"] + list(df.columns), key="caat5_date")
    if date_col == "(ninguna)":
        date_col = None
    amt_col = st.selectbox("(Opcional) Monto", ["(ninguna)"] + list(df.columns), key="caat5_amt")
    if amt_col == "(ninguna)":
        amt_col = None

    dups = pd.DataFrame()
    if key_cols:
        dups = (
            df[df.duplicated(subset=key_cols, keep=False)]
            .sort_values(key_cols)
            .copy()
        )
    k1, k2 = st.columns(2)
    k1.metric("Registros", f"{len(df):,}")
    k2.metric("Duplicados exactos", f"{len(dups):,}")

    with st.expander("Duplicados", expanded=not dups.empty):
        if dups.empty:
            st.info("No se detectaron duplicados exactos con las columnas seleccionadas.")
        else:
            st.dataframe(dups, use_container_width=True)
            st.download_button(
                "⬇️ Descargar duplicados (XLSX)",
                data=make_xlsx(dups, "Duplicados"),
                file_name="CAAT5_duplicados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.markdown("**Near-duplicados (opcional)**")
    if date_col and amt_col:
        tol_days = st.slider("Tolerancia de fecha (días)", 0, 10, 2, key="caat5_tol")
        w = df[[date_col, amt_col] + key_cols].copy()
        w["date"] = pd.to_datetime(w[date_col], errors="coerce").dt.date
        w["amount"] = pd.to_numeric(w[amt_col], errors="coerce").round(2)
        w = w.dropna(subset=["date", "amount"])

        near = []
        by_amount = w.groupby("amount")
        for amount, grp in by_amount:
            dates = grp["date"].sort_values().tolist()
            if len(dates) < 2:
                continue
            for i in range(len(dates) - 1):
                if abs((dates[i + 1] - dates[i]).days) <= tol_days:
                    near.append(amount)
                    break
        near_df = w[w["amount"].isin(near)].sort_values(["amount", "date"])
        with st.expander("Near-duplicados", expanded=not near_df.empty):
            if near_df.empty:
                st.info("No se detectaron near-duplicados con los parámetros actuales.")
            else:
                st.dataframe(near_df, use_container_width=True)
                st.download_button(
                    "⬇️ Descargar near-duplicados (XLSX)",
                    data=make_xlsx(near_df, "Near_duplicados"),
                    file_name="CAAT5_near_duplicados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

# ------------------------------
# Modo libre – EDA guiado
# ------------------------------

def module_free_mode():
    st.subheader("Modo libre – Sube cualquier archivo y exploramos")
    st.markdown("""
Este modo acepta **cualquier CSV/XLSX**. Intentamos detectar tipos de columna y brindamos:
- Conteo de filas/columnas, valores faltantes, tipos
- Filtros dinámicos por columna
- KPIs básicos (suma, promedio si aplica) y **descarga** del resultado filtrado (XLSX)
""")

    df = file_uploader_block("Archivo libre", key="free")
    if df is None:
        return

    st.markdown("### Información rápida")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{len(df):,}")
    c2.metric("Columnas", f"{df.shape[1]:,}")
    missing = int(df.isna().sum().sum())
    c3.metric("Celdas vacías", f"{missing:,}")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    c4.metric("Numéricas", f"{len(num_cols):,}")

    with st.expander("Tipos y nulos por columna", expanded=False):
        info = pd.DataFrame({
            "columna": df.columns,
            "dtype": [str(t) for t in df.dtypes.values],
            "nulos": df.isna().sum().values
        })
        st.dataframe(info, use_container_width=True)

    st.markdown("### Filtros rápidos")
    filtered = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            rng = st.slider(f"{c} (rango)", float(df[c].min()), float(df[c].max()),
                            (float(df[c].min()), float(df[c].max())), key=f"free_rng_{c}")
            filtered = filtered[filtered[c].between(rng[0], rng[1])]
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            d1 = pd.to_datetime(df[c].min())
            d2 = pd.to_datetime(df[c].max())
            start, end = st.date_input(f"{c} (fecha)", value=(d1.date(), d2.date()), key=f"free_date_{c}")
            filtered = filtered[(pd.to_datetime(filtered[c]).dt.date >= start) &
                                (pd.to_datetime(filtered[c]).dt.date <= end)]
        else:
            unique_vals = df[c].dropna().astype(str).unique().tolist()
            if 0 < len(unique_vals) <= 100:
                vals = st.multiselect(f"{c} (valores)", unique_vals, default=unique_vals, key=f"free_ms_{c}")
                filtered = filtered[filtered[c].astype(str).isin(vals)]

    st.markdown("### Resultado filtrado")
    st.dataframe(filtered.head(1000), use_container_width=True)
    st.download_button(
        "⬇️ Descargar filtrado (XLSX)",
        data=make_xlsx(filtered, "Filtrado"),
        file_name="modo_libre_filtrado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if num_cols:
        st.markdown("### KPIs rápidos")
        n1, n2, n3 = st.columns(3)
        sel = st.selectbox("Columna numérica", num_cols, key="free_kpi_col")
        n1.metric("Suma", f"{filtered[sel].sum():,.2f}")
        n2.metric("Promedio", f"{filtered[sel].mean():,.2f}")
        n3.metric("Desv. estándar", f"{filtered[sel].std(ddof=0):,.2f}")

# ------------------------------
# App – estructura
# ------------------------------

def main():
    st.title("Aprendizaje Colaborativo y Práctico – 2do Parcial")
    st.caption("Suite de herramientas CAAT para auditoría asistida.")

    tabs = st.tabs(["CAAT 1–5", "Modo libre"])

    with tabs[0]:
        with st.expander("Ayuda general (antes de empezar)", expanded=True):
            st.markdown("""
<span class="badge">¿Qué hace esta app?</span> Corre **5 CAAT** comunes de auditoría y un **Modo libre** para explorar cualquier archivo.

<span class="badge-red">Errores comunes</span>  
- **No se pudo leer el Excel**: verifica que el archivo **no** esté protegido y sea un **.xlsx válido**.  
- **Fechas vacías/NaT**: verifica el **formato** o elige la **columna correcta**.  
- **Selectbox duplicado**: aquí cada select tiene **claves únicas**, no deberías ver este error.  

<span class="badge-green">Descargas</span>  
Todos los módulos generan **XLSX** de hallazgos.
            """, unsafe_allow_html=True)

        st.markdown("---")
        module_caat1()
        st.markdown("---")
        module_caat2()
        st.markdown("---")
        module_caat3()
        st.markdown("---")
        module_caat4()
        st.markdown("---")
        module_caat5()

        st.markdown("---")
        st.markdown("#### Conclusión general (sugerida)")
        st.markdown("""
- Usa los KPIs para priorizar: **concentración** de outliers por proveedor (CAAT 4),  
  **reincidencia** fuera de horario (CAAT 1) y **conflictos** SoD (CAAT 2).
- En conciliación (CAAT 3), **mayores gaps** de tolerancia y **no conciliados** son foco.
- Descarga los XLSX y documenta tu evidencia.
        """)

    with tabs[1]:
        module_free_mode()

if __name__ == "__main__":
    main()

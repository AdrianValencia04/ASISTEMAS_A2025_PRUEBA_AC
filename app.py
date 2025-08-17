# -*- coding: utf-8 -*-
# APP: Aprendizaje Colaborativo y Práctico – 2do Parcial
# Suite de CAAT (1–5) + Modo libre
#
# Notas:
# - Soporta CSV y XLSX (usa openpyxl).
# - Soluciona el error: file_uploader (único vs múltiple) y claves únicas en selectbox().
# - Cada módulo tiene: explicación breve, guía de uso, tips de error, análisis y descarga de reporte.
# - "Modo libre" permite subir cualquier archivo y explorar/analizar rápidamente.
#
# Autoría: tú :)

import io
from datetime import datetime, timedelta, time as dtime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# ===================== UTILIDADES ========================
# =========================================================

OK = "✅"
WARN = "⚠️"
ERR = "❌"

st.set_page_config(
    page_title="Aprendizaje Colaborativo y Práctico – 2do Parcial",
    layout="wide",
    page_icon="🧭",
)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza header a minúsculas y sin espacios/símbolos que den problemas."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
        .str.lower()
    )
    return df

def try_read_csv(file, sample_rows: int = 2000) -> Optional[pd.DataFrame]:
    """Intenta leer CSV detectando separador; retorna None si falla."""
    try:
        raw = file.read()
        if isinstance(raw, bytes):
            data = io.BytesIO(raw)
        else:
            data = io.StringIO(raw)
        # Detect línea y delimitador
        sniffer = pd.io.common.get_handle
        # Lo más robusto es probar ; , y tab:
        for sep in [",", ";", "\t", "|"]:
            data.seek(0) if hasattr(data, "seek") else None
            try:
                df = pd.read_csv(data, sep=sep, engine="python", nrows=sample_rows)
                # Si tiene más de 1 columna, parece válido. Releer completo:
                if df.shape[1] >= 1:
                    data.seek(0) if hasattr(data, "seek") else None
                    full = pd.read_csv(data, sep=sep, engine="python")
                    return full
            except Exception:
                continue
    except Exception:
        pass
    return None

def try_read_excel(file, sheet=None) -> Tuple[Optional[pd.DataFrame], List[str], Optional[str]]:
    """Lee un Excel (XLSX). Retorna (df, lista_hojas, error)."""
    try:
        # Para poder reusar el file dos veces, lo copio en buffer
        content = file.read()
        bio = io.BytesIO(content)
        xls = pd.ExcelFile(bio, engine="openpyxl")
        sheets = list(xls.sheet_names)

        if sheet is None:
            return None, sheets, None

        bio.seek(0)
        df = pd.read_excel(bio, sheet_name=sheet, engine="openpyxl")
        return df, sheets, None
    except Exception as e:
        return None, [], f"No se pudo leer el Excel (.xlsx): {e}"

def ensure_datetime_series(series: pd.Series) -> pd.Series:
    """Convierte a datetime, si falla deja NaT."""
    return pd.to_datetime(series, errors="coerce")

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def badge(text: str, color: str = "blue") -> str:
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:6px;font-size:12px">{text}</span>'

def section_title(icon: str, title: str):
    st.markdown(f"### {icon} {title}")

def tiny(text: str):
    st.caption(text)

def metric_card(label: str, value: str):
    st.markdown(f"""
    <div style="display:inline-block;min-width:160px;margin-right:10px;padding:10px 12px;border:1px solid #eee;border-radius:10px">
        <div style="font-size:13px;color:#666">{label}</div>
        <div style="font-size:20px;font-weight:700">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# ============ CARGA DE ARCHIVOS (ROBUSTA) ================
# =========================================================

def read_table_uploader(
    label: str,
    key_prefix: str,
    help_txt: str = "",
    accept_multiple: bool = False,
) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Subida de archivos CSV/XLSX.
    - Maneja el caso único (UploadedFile) y múltiple (lista de UploadedFile).
    - Pide hoja para XLSX (key único por archivo).
    """
    files = st.file_uploader(
        label,
        type=["csv", "xlsx"],
        accept_multiple_files=accept_multiple,
        help=help_txt,
        key=f"{key_prefix}_uploader"
    )

    if not files:
        return [], []

    # Normalizar a lista si es único.
    if not accept_multiple:
        files = [files]

    results = []
    names = []

    for idx, file in enumerate(files):
        name = file.name
        ext = name.split(".")[-1].lower()

        if ext == "csv":
            df = try_read_csv(file)
            if df is None:
                st.error(f"{ERR} No se pudo leer el CSV '{name}'. Revisa delimitador/encoding o sube otro archivo.")
                continue

            df = normalize_columns(df)
            results.append(df)
            names.append(name)

        elif ext == "xlsx":
            # Primero listamos hojas
            file.seek(0)
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
                key=f"{key_prefix}_sheet_{idx}",   # 🔑 clave única
                help="Selecciona la hoja que contiene tus datos"
            )

            # Leer la hoja escogida
            file.seek(0)
            df, _, err2 = try_read_excel(file, sheet=sheet)
            if err2:
                st.error(f"{ERR} {err2}\nArchivo: {name}")
                continue

            df = normalize_columns(df)
            results.append(df)
            names.append(f"{name} :: {sheet}")

        else:
            st.warning(f"{WARN} Formato no soportado: {name}. Usa CSV o XLSX.")
            continue

    return results, names

# =========================================================
# ===================== MÓDULOS CAAT ======================
# =========================================================

def module_caat1():
    """
    CAAT 1 – Validación de registros fuera de horario.
    """
    section_title("⏰", "CAAT 1 – Registros fuera de horario")
    with st.expander("¿Cómo usar este módulo? / Tips y errores comunes", expanded=False):
        st.write("""
        **Objetivo:** identificar eventos fuera del horario laboral definido por el auditor.

        **Pasos rápidos:**
        1. Sube tu log de actividades (CSV/XLSX).
        2. Elige la columna **Usuario** y **Fecha/Hora**. (Opcional **Acción/Severidad**).
        3. Define el **horario laboral** e indica si solo consideras **días hábiles (L–V)**.
        4. Revisa métricas, hallazgos y descarga el **CSV** para evidencias.

        **Si te sale error**:
        - Asegúrate de seleccionar la **columna de fecha/hora** correcta.
        - Si tu Excel muestra "No se pudo leer el Excel", revisa que **no esté protegido** y que sea un **.xlsx válido**.
        - Si ves una advertencia de columnas vacías/NaT, revisa el **formato de fecha**.
        """)

    dfs, _ = read_table_uploader(
        "Log de actividades (CSV/XLSX)",
        "caat1_log",
        help_txt="Tu bitácora o log con usuario, fecha/hora y (opcional) acción/severidad.",
        accept_multiple=False,
    )
    if not dfs:
        tiny("Sube un archivo para continuar…")
        return

    work = dfs[0]
    all_cols = list(work.columns)

    c1, c2, c3 = st.columns([1, 1, 1])
    user_col = c1.selectbox("Columna Usuario", options=all_cols, key="c1_user")
    dt_col   = c2.selectbox("Columna Fecha/Hora", options=all_cols, key="c1_dt")
    act_col  = c3.selectbox("Columna Acción (opcional)", options=["(ninguna)"] + all_cols, key="c1_action")

    c4, c5, c6 = st.columns([1, 1, 1])
    start_h = c4.selectbox("Inicio jornada", [f"{h:02d}:{m:02d}" for h in range(0,24) for m in (0,15,30,45)], index=32)  # 08:00
    end_h   = c5.selectbox("Fin jornada", [f"{h:02d}:{m:02d}" for h in range(0,24) for m in (0,15,30,45)], index=72)   # 18:00
    weekdays_only = c6.checkbox("Solo días hábiles (L–V)", value=True)

    # Procesamiento
    df = work[[user_col, dt_col] + ([act_col] if act_col != "(ninguna)" else [])].copy()
    df.rename(columns={user_col: "user", dt_col: "dt"}, inplace=True)
    if act_col != "(ninguna)":
        df.rename(columns={act_col: "action"}, inplace=True)

    df["dt"] = ensure_datetime_series(df["dt"])
    df = df[~df["dt"].isna()].copy()

    def hhmm_to_time(s: str) -> dtime:
        h, m = s.split(":")
        return dtime(int(h), int(m))

    sh = hhmm_to_time(start_h)
    eh = hhmm_to_time(end_h)

    df["weekday"] = df["dt"].dt.weekday  # 0 lunes - 6 domingo
    df["dt_mins"] = (df["dt"].dt.hour * 60) + df["dt"].dt.minute

    start_mins = sh.hour * 60 + sh.minute
    end_mins   = eh.hour * 60 + eh.minute

    in_schedule = (df["dt_mins"] >= start_mins) & (df["dt_mins"] <= end_mins)
    if weekdays_only:
        in_schedule &= df["weekday"].between(0, 4)

    df["fuera_horario"] = ~in_schedule

    total = len(df)
    fh    = int(df["fuera_horario"].sum())
    pct   = (fh / total * 100) if total else 0.0

    cA, cB, cC = st.columns(3)
    with cA: metric_card("Eventos totales", f"{total:,}")
    with cB: metric_card("Fuera de horario", f"{fh:,}")
    with cC: metric_card("% fuera de horario", f"{pct:.2f}%")

    hallazgos = df[df["fuera_horario"]].copy()
    if not hallazgos.empty:
        st.write("#### Hallazgos")
        st.dataframe(hallazgos.head(200), use_container_width=True)

        rep = hallazgos.copy()
        st.download_button(
            "⬇️ Descargar reporte (CSV)",
            data=to_csv_bytes(rep),
            file_name="CAAT1_fuera_de_horario.csv",
            mime="text/csv",
        )
    else:
        st.info("No se encontraron eventos fuera del horario laboral con los filtros elegidos.")

    st.divider()
    st.write("**Conclusión de CAAT 1**")
    if fh == 0:
        st.success("No se identificaron registros fuera de horario con los parámetros definidos.")
    else:
        st.warning(
            f"Se identificaron **{fh}** registros fuera de horario. "
            f"Revisar usuarios y acciones involucradas para validar justificaciones."
        )

def module_caat2():
    """
    CAAT 2 – Auditoría de privilegios (roles críticos y SoD).
    """
    section_title("🛡️", "CAAT 2 – Auditoría de privilegios (roles críticos y SoD)")
    with st.expander("¿Cómo usar este módulo? / Tips y errores comunes", expanded=False):
        st.write("""
        **Objetivo:** identificar **roles críticos** y violaciones de **Segregación de Funciones (SoD)**.

        **Pasos rápidos:**
        1. Sube tu maestro de **Usuarios/Roles**.
        2. Elige **Usuario** y **Rol**.
        3. Marca la columna que indica si el rol es **crítico** (o define una lista).
        4. Escribe reglas SoD (una por línea) en formato `ROL_A -> ROL_B`.

        **Si te sale error**:
        - Selecciona correctamente las columnas de **usuario** y **rol**.
        - Si tu Excel no abre: revisa protección, y que sea **xlsx** válido.
        """)

    dfs, _ = read_table_uploader(
        "Usuarios/Roles (CSV/XLSX)",
        "caat2_roles",
        help_txt="Archivo con las asignaciones de usuarios a roles (y, opcionalmente, criticidad).",
        accept_multiple=False
    )
    if not dfs:
        tiny("Sube un archivo para continuar…")
        return

    base = dfs[0]
    cols = list(base.columns)

    c1, c2, c3 = st.columns([1,1,1])
    ucol = c1.selectbox("Columna Usuario", cols, key="c2_user")
    rcol = c2.selectbox("Columna Rol", cols, key="c2_role")
    ccol = c3.selectbox("Columna es_crítico (opcional)", ["(ninguna)"] + cols, key="c2_crit")

    base = base[[ucol, rcol] + ([ccol] if ccol != "(ninguna)" else [])].copy()
    base.rename(columns={ucol:"user", rcol:"role"}, inplace=True)
    if ccol != "(ninguna)":
        base.rename(columns={ccol:"is_critical"}, inplace=True)
        base["is_critical"] = base["is_critical"].astype(str).str.lower().str.strip().isin(["1","true","si","sí","critical","critico","crítico"])
    else:
        base["is_critical"] = False

    st.write("#### Reglas SoD (formato: `ROL_A -> ROL_B`, una por línea)")
    sod_text = st.text_area(
        "Reglas SoD",
        value="APROBACIONES -> PAGOS\nTESORERIA -> CONTABILIDAD",
        help="Puedes copiar/pegar tus reglas SoD. Se verifica si un mismo usuario posee ambos roles definidos.",
        key="c2_sod_rules"
    )

    # Análisis
    # 1) Roles críticos por usuario
    crit = base[base["is_critical"]].copy()

    # 2) Violaciones SoD
    sod_rules = []
    for line in sod_text.splitlines():
        line = line.strip()
        if not line or "->" not in line:
            continue
        a, b = [x.strip() for x in line.split("->", 1)]
        if a and b:
            sod_rules.append((a, b))

    sod_findings = []
    if sod_rules:
        user_roles = base.groupby("user")["role"].apply(set).to_dict()
        for u, roles in user_roles.items():
            for (a, b) in sod_rules:
                if a in roles and b in roles:
                    sod_findings.append({"user": u, "rule": f"{a} -> {b}", "roles": ", ".join(sorted(list(roles)))})

    sod_df = pd.DataFrame(sod_findings)

    # Métricas
    tot_users = base["user"].nunique()
    tot_asig  = len(base)
    tot_crit  = crit["user"].nunique() if not crit.empty else 0
    tot_sod   = len(sod_df)

    cA, cB, cC, cD = st.columns(4)
    with cA: metric_card("Usuarios", f"{tot_users:,}")
    with cB: metric_card("Asignaciones", f"{tot_asig:,}")
    with cC: metric_card("Usuarios con rol crítico", f"{tot_crit:,}")
    with cD: metric_card("Violaciones SoD", f"{tot_sod:,}")

    st.write("#### Hallazgos")
    if not crit.empty:
        st.markdown("**Roles críticos**")
        st.dataframe(crit.head(200), use_container_width=True)
        st.download_button("⬇️ Descargar roles críticos (CSV)", to_csv_bytes(crit), "CAAT2_roles_criticos.csv", mime="text/csv")

    if not sod_df.empty:
        st.markdown("**Violaciones SoD**")
        st.dataframe(sod_df.head(200), use_container_width=True)
        st.download_button("⬇️ Descargar violaciones SoD (CSV)", to_csv_bytes(sod_df), "CAAT2_violaciones_sod.csv", mime="text/csv")

    if crit.empty and sod_df.empty:
        st.info("No se detectaron roles críticos ni violaciones SoD con los datos/ reglas provistas.")

def module_caat3():
    """
    CAAT 3 – Conciliación de logs vs transacciones.
    """
    section_title("🔗", "CAAT 3 – Conciliación de logs vs transacciones")
    with st.expander("¿Cómo usar este módulo? / Tips y errores comunes", expanded=False):
        st.write("""
        **Objetivo:** conciliar logs del sistema vs transacciones, buscar **IDs faltantes** y **desfases de tiempo**.

        **Pasos rápidos:**
        1. Sube **Logs** y **Transacciones**.
        2. En ambos: elige **ID** y **Fecha/Hora**.
        3. Define la **tolerancia** (en minutos) para marcar desfase.

        **Errores frecuentes**:
        - Selecciona la **misma clave** (ID) en ambos archivos.
        - Convierte y valida el **formato de fecha**.
        """)

    st.subheader("Logs (CSV/XLSX)")
    logs, _ = read_table_uploader("Arrastra o busca el archivo de Logs", "c3_logs", accept_multiple=False)
    st.subheader("Transacciones (CSV/XLSX)")
    txs, _  = read_table_uploader("Arrastra o busca el archivo de Transacciones", "c3_txs", accept_multiple=False)

    if not logs or not txs:
        tiny("Sube ambos archivos para continuar…")
        return

    logs = logs[0]
    txs  = txs[0]

    c1, c2 = st.columns(2)
    lid  = c1.selectbox("ID en Logs", logs.columns.tolist(), key="c3_lid")
    ldt  = c1.selectbox("Fecha/Hora en Logs", logs.columns.tolist(), key="c3_ldt")
    tid  = c2.selectbox("ID en Transacciones", txs.columns.tolist(), key="c3_tid")
    tdt  = c2.selectbox("Fecha/Hora en Transacciones", txs.columns.tolist(), key="c3_tdt")

    tol = st.slider("Tolerancia de tiempo (min)", min_value=0, max_value=120, value=15, step=5)

    L = logs[[lid, ldt]].rename(columns={lid:"id", ldt:"ldt"}).copy()
    T = txs[[tid, tdt]].rename(columns={tid:"id", tdt:"tdt"}).copy()
    L["ldt"] = ensure_datetime_series(L["ldt"])
    T["tdt"] = ensure_datetime_series(T["tdt"])

    # IDs faltantes
    ids_logs = set(L["id"].astype(str))
    ids_txs  = set(T["id"].astype(str))
    missing_in_txs  = sorted(list(ids_logs - ids_txs))
    missing_in_logs = sorted(list(ids_txs - ids_logs))

    # Desfase de tiempo
    merged = pd.merge(L, T, on="id", how="inner")
    merged["diff_mins"] = (merged["ldt"] - merged["tdt"]).dt.total_seconds().div(60).abs()
    out_of_tol = merged[merged["diff_mins"] > tol].copy()

    # KPIs
    cA, cB, cC = st.columns(3)
    with cA: metric_card("IDs en Logs", f"{len(ids_logs):,}")
    with cB: metric_card("IDs en Transacciones", f"{len(ids_txs):,}")
    with cC: metric_card("Desfases > tolerancia", f"{len(out_of_tol):,}")

    st.write("#### Hallazgos")
    if missing_in_txs:
        df1 = pd.DataFrame({"id_faltantes_en_transacciones": missing_in_txs})
        st.dataframe(df1, use_container_width=True)
        st.download_button("⬇️ Descargar IDs faltantes en transacciones", to_csv_bytes(df1), "CAAT3_ids_faltantes_en_txs.csv")

    if missing_in_logs:
        df2 = pd.DataFrame({"id_faltantes_en_logs": missing_in_logs})
        st.dataframe(df2, use_container_width=True)
        st.download_button("⬇️ Descargar IDs faltantes en logs", to_csv_bytes(df2), "CAAT3_ids_faltantes_en_logs.csv")

    if not out_of_tol.empty:
        st.dataframe(out_of_tol.head(200), use_container_width=True)
        st.download_button("⬇️ Descargar desfases (CSV)", to_csv_bytes(out_of_tol), "CAAT3_desfases_fuera_tolerancia.csv")

    if not (missing_in_txs or missing_in_logs or len(out_of_tol)):
        st.info("No se detectaron diferencias con los parámetros actuales.")

def module_caat4():
    """
    CAAT 4 – Variación inusual de pagos – outliers.
    """
    section_title("💸", "CAAT 4 – Variación inusual de pagos – outliers")
    with st.expander("¿Cómo usar este módulo? / Tips y errores comunes", expanded=False):
        st.write("""
        **Objetivo:** encontrar **picos/caídas atípicas** en pagos mensuales por proveedor.

        **Pasos rápidos:**
        1. Sube **Pagos** (CSV/XLSX).
        2. Selecciona **Proveedor**, **Fecha** y **Monto**.
        3. Ajusta el **umbral de outliers (|z| robusto)**.

        **Errores frecuentes**:
        - Verifica columna **Monto** como numérica.
        - Si tu Excel no abre: revisa protección/xlsx válido.
        """)

    dfs, _ = read_table_uploader(
        "Histórico de pagos (CSV/XLSX)",
        "caat4_pagos",
        help_txt="Archivo con pagos por proveedor/fecha/monto.",
        accept_multiple=False
    )
    if not dfs:
        tiny("Sube un archivo para continuar…")
        return

    pay = dfs[0]
    cols = list(pay.columns)

    c1, c2, c3 = st.columns(3)
    prov = c1.selectbox("Columna Proveedor", cols, key="c4_prov")
    fcol = c2.selectbox("Columna Fecha", cols, key="c4_fecha")
    mcol = c3.selectbox("Columna Monto", cols, key="c4_monto")
    th   = st.slider("Umbral |z| (robusto)", min_value=2.0, max_value=6.0, value=3.5, step=0.5)

    df = pay[[prov, fcol, mcol]].rename(columns={prov:"proveedor", fcol:"fecha", mcol:"monto"}).copy()
    df["fecha"]  = ensure_datetime_series(df["fecha"])
    df["monto"]  = pd.to_numeric(df["monto"], errors="coerce")
    df = df.dropna(subset=["fecha", "monto"]).copy()

    # Z-score robusto por proveedor (MAD)
    outliers = []
    for prov_name, g in df.groupby("proveedor"):
        x = g["monto"].astype(float)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) or 1e-9
        zrob = 0.6745 * (x - med) / mad
        g2 = g.copy()
        g2["zrob"] = zrob
        g2["is_outlier"] = np.abs(zrob) > th
        outliers.append(g2)

    res = pd.concat(outliers, ignore_index=True) if outliers else pd.DataFrame()

    # Métricas
    tot_rows = len(res)
    out_cnt  = int(res["is_outlier"].sum()) if tot_rows else 0
    cA, cB = st.columns(2)
    with cA: metric_card("Registros", f"{tot_rows:,}")
    with cB: metric_card("Outliers detectados", f"{out_cnt:,}")

    # Hallazgos
    if out_cnt > 0:
        st.write("#### Hallazgos (outliers)")
        st.dataframe(res[res["is_outlier"]].head(200), use_container_width=True)
        st.download_button(
            "⬇️ Descargar outliers (CSV)",
            to_csv_bytes(res[res["is_outlier"]]),
            "CAAT4_outliers.csv",
            mime="text/csv"
        )
    else:
        st.info("No se detectaron outliers con el umbral actual.")

def module_caat5():
    """
    CAAT 5 – Duplicados y anomalías simples.
    (Flexible para validar duplicidad por columnas elegidas)
    """
    section_title("📊", "CAAT 5 – Duplicados / anomalías simples")
    with st.expander("¿Cómo usar este módulo? / Tips y errores comunes", expanded=False):
        st.write("""
        **Objetivo:** detectar **duplicados**, **nulos** y **anomalías simples** en un registro maestro.

        **Pasos rápidos:**
        1. Sube el **maestro** (CSV/XLSX).
        2. Selecciona las **columnas clave** para validar duplicidad.
        3. Revisa nulos, duplicados y descarga el **reporte**.

        **Errores frecuentes**:
        - Elige columnas que **identifiquen** unívocamente el registro (ID, código, etc.).
        """)

    dfs, _ = read_table_uploader(
        "Maestro (CSV/XLSX)",
        "caat5_maestro",
        help_txt="Archivo con registros maestros o catálogos; puedes elegir columnas clave.",
        accept_multiple=False
    )
    if not dfs:
        tiny("Sube un archivo para continuar…")
        return

    base = dfs[0]
    cols = list(base.columns)

    keys = st.multiselect("Columnas clave para detectar duplicados", cols, default=cols[:1], key="c5_keys")

    # Hallazgos
    report = {}
    if keys:
        dup = base[base.duplicated(subset=keys, keep=False)].copy()
        report["duplicados"] = dup
    else:
        dup = pd.DataFrame()

    nulls = base.isna().sum().sort_values(ascending=False)
    nulls_df = pd.DataFrame({"columna": nulls.index, "n_nulos": nulls.values})

    cA, cB = st.columns(2)
    with cA: metric_card("Filas totales", f"{len(base):,}")
    with cB: metric_card("Duplicados (por claves)", f"{len(dup):,}")

    st.write("#### Nulos por columna")
    st.dataframe(nulls_df, use_container_width=True)
    st.download_button("⬇️ Descargar nulos (CSV)", to_csv_bytes(nulls_df), "CAAT5_nulos_por_columna.csv")

    if not dup.empty:
        st.write("#### Duplicados detectados")
        st.dataframe(dup.head(200), use_container_width=True)
        st.download_button("⬇️ Descargar duplicados (CSV)", to_csv_bytes(dup), "CAAT5_duplicados.csv")
    else:
        st.info("No se detectaron duplicados con las claves elegidas.")

# =========================================================
# ===================== MODO LIBRE ========================
# =========================================================

def free_mode():
    section_title("🧪", "Modo libre – Exploración y pruebas rápidas")
    st.write("""
    **Aquí puedes subir literalmente cualquier archivo** (CSV o XLSX) y la app intentará:
    - Leerlo y **normalizar** columnas.
    - Mostrar **tipos**, **nulos**, **valores únicos** y **rangos de fechas** si aplica.
    - Si detecta columnas clave para algún CAAT, te sugerirá correrlo.
    """)

    dfs, names = read_table_uploader("Arrastra tus archivos (CSV/XLSX)", "free_mode", accept_multiple=True)
    if not dfs:
        tiny("Sube uno o más archivos para continuar…")
        return

    for i, (df, name) in enumerate(zip(dfs, names)):
        st.subheader(f"Archivo {i+1}: {name}")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("**Perfil rápido**")
        info = pd.DataFrame({
            "columna": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "n_nulos": [int(df[c].isna().sum()) for c in df.columns],
            "n_unicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
        })
        st.dataframe(info, use_container_width=True)
        st.download_button(f"⬇️ Descargar perfil ({i+1})", to_csv_bytes(info), f"free_perfil_{i+1}.csv")

        # Heurísticas mínimas para sugerir un CAAT
        columns = set(df.columns)
        sugg = []
        if {"usuario", "user"} & columns and {"timestamp", "fecha", "fechahora", "dt"} & columns:
            sugg.append("CAAT 1 – Registros fuera de horario")
        if {"usuario","user"} & columns and {"rol","role"} & columns:
            sugg.append("CAAT 2 – Privilegios / SoD")
        if {"id"} & columns and {"fecha","fechahora","timestamp","dt"} & columns:
            sugg.append("CAAT 3 – Conciliación (si subes transacciones del otro lado)")
        if {"proveedor"} & columns and {"monto","importe"} & columns:
            sugg.append("CAAT 4 – Outliers de pagos")
        if df.shape[1] >= 2:
            sugg.append("CAAT 5 – Duplicados (elige columnas clave)")

        if sugg:
            st.info("Sugerencias de análisis posibles: " + ", ".join(sugg))

# =========================================================
# ===================== CONCLUSIÓN ========================
# =========================================================

def global_guidance():
    with st.expander("📘 Ayuda general (antes de empezar)", expanded=False):
        st.write(f"""
        {badge("Qué hace esta app", "#4f46e5")}  
        Esta aplicación te permite correr **5 CAAT** comunes de auditoría de datos y un **Modo libre** para explorar cualquier archivo.

        {badge("Errores comunes", "#ef4444")}  
        - *No se pudo leer el Excel*: asegúrate de que el archivo **no esté protegido** y sea un **.xlsx válido**.  
        - *Fechas vacías/NaT*: verifica el **formato** o selecciona la **columna correcta**.  
        - *IDs no coinciden*: selecciona las **claves** adecuadas en CAAT 3.  
        - *Selectbox duplicado*: cada selección en esta app usa claves **únicas**, por lo que no deberías ver este error.

        {badge("Descargas", "#059669")}  
        Cada módulo genera su **reporte** en CSV para evidencia.
        """)

# =========================================================
# ===================== LAYOUT PRINCIPAL ==================
# =========================================================

st.markdown("## Aprendizaje Colaborativo y Práctico – 2do Parcial")
st.markdown("Suite de herramientas CAAT para auditoría asistida.")

tab1, tab2 = st.tabs(["🧩 CAAT 1–5", "🧪 Modo libre"])

with tab1:
    global_guidance()

    # Un solo “contenedor” con los 5 módulos – cada uno con su expander y reportes
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
    st.subheader("📌 Conclusión general")
    st.write("""
    - Usa los hallazgos de cada módulo como **alertas** que requieren **corroboración**.  
    - En caso de **hallazgos críticos** (fuera de horario, SoD, outliers), prioriza su **explicación y evidencia**.  
    - Exporta los reportes de cada módulo y **documenta** tus decisiones de auditoría.
    """)

with tab2:
    free_mode()

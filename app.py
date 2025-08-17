import io
import base64
from datetime import datetime, time
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------- #
# -------- Utilidades UI -------- #
# ------------------------------- #

APP_TITLE = "Aprendizaje Colaborativo y Práctico – 2do Parcial"
st.set_page_config(page_title=APP_TITLE, layout="wide")


def badge(text: str, color: str = "#4f46e5"):
    st.markdown(
        f"""<span style="background:{color};color:#fff;padding:2px 8px;border-radius:8px;font-size:12px">{text}</span>""",
        unsafe_allow_html=True,
    )


def success_note(msg: str):
    st.markdown(
        f"""<div style="background:#ecfdf5;border:1px solid #10b981;color:#065f46;padding:10px 12px;border-radius:8px">{msg}</div>""",
        unsafe_allow_html=True,
    )


def warn_note(msg: str):
    st.markdown(
        f"""<div style="background:#fff7ed;border:1px solid #f59e0b;color:#92400e;padding:10px 12px;border-radius:8px">{msg}</div>""",
        unsafe_allow_html=True,
    )


def error_note(msg: str):
    st.markdown(
        f"""<div style="background:#fef2f2;border:1px solid #ef4444;color:#7f1d1d;padding:10px 12px;border-radius:8px">{msg}</div>""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------ #
# -- Lectura robusta de archivos (CSV / XLSX) ---- #
# ------------------------------------------------ #

def load_table(file) -> pd.DataFrame:
    """
    Lee CSV/XLSX con tolerancia:
    - CSV: encoding UTF-8 y fallback latin-1
    - XLSX: engine openpyxl
    """
    if file is None:
        return pd.DataFrame()

    name = getattr(file, "name", "archivo_cargado")
    suffix = name.split(".")[-1].lower()

    try:
        if suffix in ["csv", "txt"]:
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(file, encoding="latin-1")
        elif suffix in ["xlsx", "xlsm", "xls"]:
            # openpyxl para xlsx/xlsm
            file.seek(0)
            df = pd.read_excel(file, engine="openpyxl")
        else:
            error_note("Formato no soportado. Usa CSV o XLSX.")
            return pd.DataFrame()
    except Exception as e:
        error_note(f"No se pudo leer el archivo: {e}")
        return pd.DataFrame()

    # Quitar filas completamente vacías
    df = df.dropna(how="all").copy()
    return df


def upload_box(label: str, key: str, help_text: str = "") -> pd.DataFrame:
    """
    Uploader con persistencia en session_state.
    """
    col1, col2 = st.columns([1, 4])
    with col1:
        badge("Cargar CSV/XLSX", "#059669")
    with col2:
        st.caption(help_text)

    file = st.file_uploader(
        label,
        type=["csv", "xlsx", "xlsm", "txt"],
        key=f"uploader_{key}",
        help="Arrastra tu archivo (máx 200MB).",
    )

    if file:
        st.session_state[f"file_{key}"] = file
        st.session_state[f"df_{key}"] = load_table(file)

    if f"df_{key}" in st.session_state:
        df = st.session_state[f"df_{key}"]
        success_note("Archivo listo ✅ (se conserva en memoria tras el rerun).")
        with st.expander("Vista rápida (primeras filas)", expanded=False):
            st.dataframe(df.head(50), use_container_width=True)
        return df

    warn_note("Sube un archivo para comenzar.")
    return pd.DataFrame()


def pick_column(df: pd.DataFrame, label: str, candidates: List[str], key: str, required=True) -> str:
    """
    Sugerencia automática: intenta seleccionar columna por nombres candidatos.
    """
    options = list(df.columns)
    suggested = None
    lowered = {c.lower(): c for c in options}
    for c in candidates:
        if c.lower() in lowered:
            suggested = lowered[c.lower()]
            break

    col = st.selectbox(
        label,
        options=options,
        index=options.index(suggested) if suggested in options else 0 if options else None,
        key=key,
    )
    if required and not col:
        error_note(f"Selecciona **{label}**.")
    return col or ""


def to_datetime_safe(series: pd.Series) -> pd.Series:
    """Convierte de forma segura a datetime."""
    try:
        s = pd.to_datetime(series, errors="coerce", utc=False, dayfirst=False, infer_datetime_format=True)
    except Exception:
        s = pd.to_datetime(series.astype(str), errors="coerce")
    return s


def download_csv_button(df: pd.DataFrame, filename: str, key: str):
    if df.empty:
        return
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar reporte (CSV)",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key=f"dl_{key}",
        use_container_width=True,
    )


# --------------------------------------------- #
# ----------------- CAAT 1 -------------------- #
# Registros fuera de horario                    #
# --------------------------------------------- #
def caat1():
    st.subheader("CAAT 1 – Registros fuera de horario")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. **Sube tu log** (CSV/XLSX).
2. **Usuario** y **Fecha/Hora** (se sugieren automáticamente).
3. Define **inicio/fin** de jornada y si aplican **solo días hábiles** (L–V).
4. Revisa **KPIs**, **hallazgos** y descarga el **reporte**.
        """)

    df = upload_box("Log de actividades", "caat1", "CSV/XLSX con columnas de usuario y fecha/hora.")
    if df.empty:
        return

    c1, c2, c3 = st.columns(3)
    col_user = pick_column(df, "Columna Usuario", ["usuario", "user", "empleado"], key="c1_user")
    col_dt   = pick_column(df, "Columna Fecha/Hora", ["timestamp", "fecha_registro", "fecha", "datetime", "dt"], key="c1_dt")
    col_action = st.selectbox("Columna Acción (opcional)", ["(ninguna)"] + list(df.columns), key="c1_action")

    c4, c5, c6, c7 = st.columns(4)
    start_h = c4.time_input("Inicio jornada", value=time(8, 0), key="c1_start")
    end_h   = c5.time_input("Fin jornada", value=time(18, 0), key="c1_end")
    weekdays_only = c6.checkbox("Solo días hábiles (L–V)", value=True, key="c1_weekdays")
    top_n = c7.slider("Top N reincidentes", min_value=5, max_value=50, value=10, step=1, key="c1_topn")

    # Transformaciones
    work = df.copy()
    work["__user__"] = work[col_user].astype(str)
    work["__dt__"] = to_datetime_safe(work[col_dt])

    # Filtrado válido
    work = work[work["__dt__"].notna()].copy()
    work["hour"] = work["__dt__"].dt.hour + work["__dt__"].dt.minute / 60.0
    sh = start_h.hour + start_h.minute / 60.0
    eh = end_h.hour + end_h.minute / 60.0
    in_sched = (work["hour"] >= sh) & (work["hour"] <= eh)

    if weekdays_only:
        work["weekday"] = work["__dt__"].dt.weekday  # 0 Lunes ... 6 Domingo
        in_sched &= work["weekday"].between(0, 4)

    work["fuera_horario"] = ~in_sched

    # KPIs
    total = int(len(work))
    out_count = int(work["fuera_horario"].sum())
    pct = (out_count / total * 100) if total else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Eventos totales", f"{total}")
    k2.metric("Fuera de horario", f"{out_count}")
    k3.metric("% fuera de horario", f"{pct:.2f}%")

    # Hallazgos
    findings = work[work["fuera_horario"]].copy()
    if col_action != "(ninguna)":
        findings["Acción"] = work[col_action]

    show_cols = ["__user__", "__dt__", "fuera_horario"]
    if "weekday" in findings.columns:
        show_cols += ["weekday"]
    if "Acción" in findings.columns:
        show_cols += ["Acción"]

    findings = findings.rename(columns={"__user__": "Usuario", "__dt__": "FechaHora"})
    st.dataframe(findings[show_cols], use_container_width=True, height=300)

    # Top reincidentes
    st.caption("Top reincidentes (fuera de horario)")
    top = findings.groupby("Usuario").size().sort_values(ascending=False).head(top_n).reset_index(name="FueraHorario")
    st.dataframe(top, use_container_width=True)

    # Descarga
    if findings.empty:
        success_note("✅ **Sin riesgos detectados** para las condiciones elegidas.")
    download_csv_button(findings, f"Reporte_CAAT1_{datetime.now():%Y%m%d_%H%M}.csv", key="caat1")


# --------------------------------------------- #
# ----------------- CAAT 2 -------------------- #
# Privilegios / SoD                             #
# --------------------------------------------- #
def caat2():
    st.subheader("CAAT 2 – Auditoría de privilegios (roles críticos y SoD)")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. **Sube tu maestro** de *Usuarios/Roles* (CSV/XLSX).
2. **Usuario** y **Rol** (se sugieren automáticamente).
3. (Opcional) marca/indica **roles críticos**.
4. Escribe **reglas SoD** (una por línea), formato `ROL_A -> ROL_B`.
5. Revisa **KPIs**, hallazgos y descarga **reporte**.
        """)

    df = upload_box("Usuarios/Roles", "caat2", "CSV/XLSX con columnas de usuario y rol.")
    if df.empty:
        return

    c1, c2, c3 = st.columns(3)
    col_user = pick_column(df, "Columna Usuario", ["usuario", "user", "empleado"], key="c2_user")
    col_role = pick_column(df, "Columna Rol", ["rol", "role", "módulo", "modulo"], key="c2_role")
    col_crit = st.selectbox("Columna es_crítico (opcional)", ["(ninguna)"] + list(df.columns), key="c2_crit")

    # Roles críticos manuales
    st.write("**Roles críticos (opcional)**")
    crit_manual = st.text_input(
        "Lista separada por comas (ej. ADMIN, TESORERIA, AUTORIZADOR)",
        value="",
        key="c2_crit_list"
    )

    # Reglas SoD
    st.write("**Reglas SoD (una por línea, formato `ROL_A -> ROL_B`)**")
    sod_text = st.text_area(
        "Reglas SoD",
        value="",
        key="c2_sod_text",
        height=120,
    )

    # Preparación
    base = df[[col_user, col_role]].copy()
    base.columns = ["user", "role"]
    base["user"] = base["user"].astype(str).str.strip()
    base["role"] = base["role"].astype(str).str.strip()

    # roles críticos
    crit_from_col = set()
    if col_crit != "(ninguna)":
        crit_from_col = set(df.loc[df[col_crit].astype(str).str.lower().isin(["1", "true", "sí", "si", "y"]), col_role].astype(str))

    crit_from_manual = set([x.strip() for x in crit_manual.split(",") if x.strip()]) if crit_manual else set()
    critical_roles = {r.upper() for r in (crit_from_col | crit_from_manual)}

    # reglas SoD
    rules = []
    if sod_text.strip():
        for line in sod_text.strip().splitlines():
            if "->" in line:
                a, b = line.split("->", 1)
                rules.append((a.strip().upper(), b.strip().upper()))

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Usuarios únicos", f"{base['user'].nunique()}")
    k2.metric("Roles únicos", f"{base['role'].nunique()}")
    k3.metric("Reglas SoD", f"{len(rules)}")

    # SoD hallazgos
    findings = []
    if rules:
        roles_by_user = base.groupby("user")["role"].apply(lambda x: {r.upper() for r in x}).to_dict()
        for u, rset in roles_by_user.items():
            for a, b in rules:
                if a in rset and b in rset:
                    findings.append({"Usuario": u, "Regla": f"{a} -> {b}", "Rol_A": a, "Rol_B": b})
    sod_df = pd.DataFrame(findings)

    # críticos
    crit_df = pd.DataFrame()
    if critical_roles:
        x = base.copy()
        x["EsCrítico"] = x["role"].str.upper().isin(critical_roles)
        crit_df = x[x["EsCrítico"]].rename(columns={"user": "Usuario", "role": "Rol"})[["Usuario", "Rol", "EsCrítico"]]

    # Mostrar
    st.markdown("**Hallazgos SoD (roles incompatibles)**")
    if sod_df.empty:
        success_note("✅ **Sin riesgos detectados** según las reglas SoD ingresadas.")
    else:
        st.dataframe(sod_df, use_container_width=True, height=260)
        download_csv_button(sod_df, f"Reporte_CAAT2_SoD_{datetime.now():%Y%m%d_%H%M}.csv", key="caat2_sod")

    st.markdown("**Roles críticos**")
    if crit_df.empty:
        if critical_roles:
            success_note("✅ **Sin riesgos detectados** (ningún usuario posee roles marcados como críticos).")
        else:
            warn_note("No se indicaron **roles críticos**. (Opcional)")
    else:
        st.dataframe(crit_df, use_container_width=True, height=260)
        download_csv_button(crit_df, f"Reporte_CAAT2_Criticos_{datetime.now():%Y%m%d_%H%M}.csv", key="caat2_crit")


# --------------------------------------------- #
# ----------------- CAAT 3 -------------------- #
# Conciliación logs vs transacciones            #
# --------------------------------------------- #
def caat3():
    st.subheader("CAAT 3 – Conciliación de logs vs transacciones")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. Sube **Logs** y **Transacciones** (CSV/XLSX).
2. En ambos: elige **ID** y **Fecha/Hora**.
3. Define **tolerancia** (minutos) para marcar desfases.
4. Revisa **No conciliados** y descarga el **reporte**.
        """)

    st.markdown("### Logs")
    logs = upload_box("Logs (CSV/XLSX)", "caat3_logs")
    st.markdown("### Transacciones")
    txs = upload_box("Transacciones (CSV/XLSX)", "caat3_txs")

    if logs.empty or txs.empty:
        return

    c1, c2 = st.columns(2)
    id_logs = pick_column(logs, "ID (logs)", ["id", "id_log", "transaccion", "referencia"], key="c3_id_logs")
    dt_logs = pick_column(logs, "Fecha/Hora (logs)", ["timestamp", "fecha", "datetime", "dt"], key="c3_dt_logs")

    id_txs = pick_column(txs, "ID (transacciones)", ["id", "id_tx", "transaccion", "referencia"], key="c3_id_txs")
    dt_txs = pick_column(txs, "Fecha/Hora (transacciones)", ["timestamp", "fecha", "datetime", "dt"], key="c3_dt_txs")

    tol = st.slider("Tolerancia (minutos)", min_value=1, max_value=120, value=15, step=1, key="c3_tol")

    L = logs[[id_logs, dt_logs]].copy()
    L.columns = ["id", "dt"]
    L["dt"] = to_datetime_safe(L["dt"])

    T = txs[[id_txs, dt_txs]].copy()
    T.columns = ["id", "dt"]
    T["dt"] = to_datetime_safe(T["dt"])

    L = L[L["dt"].notna()]
    T = T[T["dt"].notna()]

    # Merge por id y buscar desfase mínimo
    out = []
    for ident, grpL in L.groupby("id"):
        grpT = T[T["id"] == ident]
        if grpT.empty:
            for _, r in grpL.iterrows():
                out.append({"id": ident, "dt_log": r["dt"], "dt_tx": pd.NaT, "dt_mins": np.nan, "conciliado": False})
        else:
            for _, r in grpL.iterrows():
                diffs = (grpT["dt"] - r["dt"]).abs()
                min_idx = diffs.idxmin()
                min_diff = diffs[min_idx]
                mins = int(min_diff.total_seconds() / 60)
                conciliado = mins <= tol
                out.append({"id": ident, "dt_log": r["dt"], "dt_tx": grpT.loc[min_idx, "dt"], "dt_mins": mins, "conciliado": conciliado})

    recon = pd.DataFrame(out).sort_values(["conciliado", "id", "dt_mins"])
    k1, k2 = st.columns(2)
    k1.metric("Registros", f"{len(recon)}")
    k2.metric("No conciliados", f"{int((~recon['conciliado']).sum())}")

    st.markdown("**Detalle (ordenado por conciliado y desfase en minutos)**")
    st.dataframe(recon, use_container_width=True, height=320)

    # Reportes
    no_conc = recon[~recon["conciliado"]].copy()
    if no_conc.empty:
        success_note("✅ **Sin riesgos detectados** (todas las coincidencias dentro de la tolerancia).")
    download_csv_button(no_conc, f"Reporte_CAAT3_NoConciliados_{datetime.now():%Y%m%d_%H%M}.csv", key="caat3")


# --------------------------------------------- #
# ----------------- CAAT 4 -------------------- #
# Variación inusual de pagos – outliers          #
# --------------------------------------------- #
def caat4():
    st.subheader("CAAT 4 – Variación inusual de pagos (outliers)")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. Sube tu **histórico de pagos** (CSV/XLSX).
2. Elige **Proveedor**, **Fecha** y **Monto**.
3. Ajusta el **umbral de outliers (|z| robusto)**.
4. Revisa **outliers** y descarga el **reporte**.
        """)

    df = upload_box("Pagos (CSV/XLSX)", "caat4", "Incluye proveedor, fecha y monto.")
    if df.empty:
        return

    c1, c2, c3 = st.columns(3)
    col_prov = pick_column(df, "Proveedor", ["proveedor", "supplier", "cliente"], key="c4_prov")
    col_dt   = pick_column(df, "Fecha", ["fecha", "f_registro", "date", "dt"], key="c4_dt")
    col_amt  = pick_column(df, "Monto", ["monto", "importe", "total", "amount", "valor"], key="c4_amt")

    th = st.slider("Umbral |z| robusto (MAD)", min_value=2.0, max_value=6.0, value=3.5, step=0.5, key="c4_th")

    work = df[[col_prov, col_dt, col_amt]].copy()
    work.columns = ["proveedor", "fecha", "monto"]
    work["fecha"] = to_datetime_safe(work["fecha"])
    work = work[work["fecha"].notna()]
    work["mes"] = work["fecha"].dt.to_period("M").dt.to_timestamp()
    work = work.dropna(subset=["monto"])

    # z-score robusto por proveedor/mes
    g = work.groupby(["proveedor", "mes"])["monto"]
    med = g.transform("median")
    mad = (g.transform(lambda x: np.median(np.abs(x - np.median(x))))).replace(0, np.nan)
    work["z_rob"] = 0.6745 * (work["monto"] - med) / mad
    work["z_rob"] = work["z_rob"].fillna(0)

    outliers = work[np.abs(work["z_rob"]) >= th].copy().sort_values(["proveedor", "mes", "z_rob"], ascending=[True, True, False])

    k1, k2 = st.columns(2)
    k1.metric("Registros", f"{len(work)}")
    k2.metric("Outliers", f"{len(outliers)}")

    st.dataframe(outliers, use_container_width=True, height=320)
    if outliers.empty:
        success_note("✅ **Sin riesgos detectados** con el umbral actual.")
    download_csv_button(outliers, f"Reporte_CAAT4_Outliers_{datetime.now():%Y%m%d_%H%M}.csv", key="caat4")


# --------------------------------------------- #
# ----------------- CAAT 5 -------------------- #
# Duplicidades y patrones                        #
# --------------------------------------------- #
def caat5():
    st.subheader("CAAT 5 – Duplicidades y patrones")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. Sube tu **base** (CSV/XLSX).
2. Elige **Proveedor**, **Fecha**, **Monto** y **ID** (opcional).
3. Define **ventana (días)** para buscar **posibles pagos duplicados**.
4. Revisa **hallazgos** y descarga el **reporte**.
        """)

    df = upload_box("Base (CSV/XLSX)", "caat5")
    if df.empty:
        return

    c1, c2, c3, c4 = st.columns(4)
    col_prov = pick_column(df, "Proveedor", ["proveedor", "supplier", "cliente"], key="c5_prov")
    col_dt   = pick_column(df, "Fecha", ["fecha", "f_registro", "date", "dt"], key="c5_dt")
    col_amt  = pick_column(df, "Monto", ["monto", "importe", "total", "amount", "valor"], key="c5_amt")
    col_id   = st.selectbox("ID (opcional)", ["(ninguna)"] + list(df.columns), key="c5_id")

    days = st.slider("Ventana de días para detectar duplicados", min_value=1, max_value=60, value=7, step=1, key="c5_days")

    work = df[[col_prov, col_dt, col_amt] + ([] if col_id == "(ninguna)" else [col_id])].copy()
    cols = ["proveedor", "fecha", "monto"]
    if col_id != "(ninguna)":
        cols.append("id")
    work.columns = cols
    work["fecha"] = to_datetime_safe(work["fecha"])
    work = work[work["fecha"].notna()]

    work = work.sort_values(["proveedor", "monto", "fecha"])
    findings = []
    for (p, m), grp in work.groupby(["proveedor", "monto"]):
        dates = grp["fecha"].tolist()
        idx = grp.index.tolist()
        for i in range(len(dates) - 1):
            diff_days = (dates[i + 1] - dates[i]).days
            if 0 < diff_days <= days:
                row_a = grp.loc[idx[i]]
                row_b = grp.loc[idx[i + 1]]
                dic = {
                    "Proveedor": p,
                    "Monto": m,
                    "Fecha_A": row_a["fecha"],
                    "Fecha_B": row_b["fecha"],
                    "Dif_dias": diff_days,
                }
                if col_id != "(ninguna)":
                    dic["ID_A"] = row_a["id"]
                    dic["ID_B"] = row_b["id"]
                findings.append(dic)

    dup_df = pd.DataFrame(findings).sort_values(["Proveedor", "Monto", "Dif_dias"]) if findings else pd.DataFrame()

    k1, k2 = st.columns(2)
    k1.metric("Registros", f"{len(work)}")
    k2.metric("Posibles duplicados", f"{len(dup_df)}")

    st.dataframe(dup_df, use_container_width=True, height=320)
    if dup_df.empty:
        success_note("✅ **Sin riesgos detectados** bajo los parámetros actuales.")
    download_csv_button(dup_df, f"Reporte_CAAT5_Duplicados_{datetime.now():%Y%m%d_%H%M}.csv", key="caat5")


# --------------------------------------------- #
# --------------- Modo libre ------------------ #
# Exploración libre con filtros simples          #
# --------------------------------------------- #
def modo_libre():
    st.subheader("Modo libre – Explora cualquier archivo")
    with st.expander("¿Qué hace este modo?", expanded=False):
        st.markdown("""
Sube **cualquier CSV/XLSX** y explóralo rápidamente:
- Vista de tabla completa.
- Filtro rápido por **columna** y **valor**.
- Descarga del resultado filtrado.
        """)

    df = upload_box("Archivo libre", "free")
    if df.empty:
        return

    cols = list(df.columns)
    c1, c2 = st.columns(2)
    col = c1.selectbox("Columna para filtrar", ["(ninguna)"] + cols, key="free_col")
    val = c2.text_input("Contiene (texto)", value="", key="free_val")

    view = df.copy()
    if col != "(ninguna)" and val:
        view = view[view[col].astype(str).str.contains(val, case=False, na=False)]

    st.dataframe(view, use_container_width=True, height=360)
    if view.empty:
        warn_note("No hay filas con ese filtro.")
    download_csv_button(view, f"Reporte_ModoLibre_{datetime.now():%Y%m%d_%H%M}.csv", key="free")


# ------------------------------- #
# ----------- Layout -------------#
# ------------------------------- #
st.markdown(f"## {APP_TITLE}")
st.caption("Suite de herramientas CAAT para auditoría asistida.")

tabs = st.tabs(["CAAT 1–5", "Modo libre"])

with tabs[0]:
    with st.expander("Ayuda general (antes de empezar)", expanded=False):
        st.markdown("""
**¿Qué hace esta app?** Permite correr **5 CAAT** comunes de auditoría y un **Modo libre** para explorar cualquier archivo.

**Errores comunes**
- **No se pudo leer el Excel**: asegúrate de que el archivo no esté protegido y sea un **.xlsx** válido (o carga **CSV**).
- **Fechas vacías/NaT**: revisa el **formato** o selecciona la **columna correcta**.
- **IDs no coinciden** (CAAT 3): confirma que **ID** y **Fecha** sean consistentes en ambos archivos.
- **Selectbox duplicado**: cada selección en esta app usa claves **únicas**, por lo que no deberías ver ese error.

**Descargas**
Cada módulo genera su **reporte (CSV)** con los hallazgos para evidencia.
        """)

    # Módulos de corrido (uno debajo del otro)
    st.divider()
    caat1()
    st.divider()
    caat2()
    st.divider()
    caat3()
    st.divider()
    caat4()
    st.divider()
    caat5()

with tabs[1]:
    modo_libre()

# app.py
# ======================================================
# Aprendizaje Colaborativo y Práctico – 2do Parcial
# Suite didáctica de herramientas CAAT (1–5) en una sola página.
# Lectura preferida: CSV (sin dependencias extra).
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import time, datetime

st.set_page_config(
    page_title="Aprendizaje Colaborativo y Práctico – 2do Parcial",
    layout="wide"
)

# ================= Utilidades robustas =================

def read_any(file):
    """
    Lector seguro de archivos. Soporta CSV nativamente.
    Si suben XLSX/XLS, muestra un mensaje para convertir a CSV (evitamos openpyxl/xlrd).
    """
    if file is None:
        return None
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            # infiere separador y codificación
            return pd.read_csv(file, encoding="utf-8", sep=None, engine="python")
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            st.error(
                "Esta app trabaja con **CSV** para evitar dependencias de Excel. "
                "Abre tu archivo en Excel y usa **Guardar como → CSV (UTF-8)**, "
                "luego vuelve a subirlo."
            )
            return None
        else:
            st.error("Formato no soportado. Sube un archivo **.csv**.")
            return None
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None

def normalize_cols(df):
    """
    Limpia espacios y hace únicos los nombres de columna (evita duplicados).
    Si hay repes, renombra como 'Fecha', 'Fecha.2', 'Fecha.3', ...
    """
    df = df.copy()
    cols = [str(c).strip() for c in df.columns]
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 1
            new_cols.append(c)
    df.columns = new_cols
    return df

def smart_datetime_cast(series):
    """Conversión segura a datetime cuando 'series' SÍ es una Serie."""
    try:
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    except Exception:
        return pd.to_datetime(series, errors="coerce")

def ensure_datetime_series(col):
    """
    Acepta Serie o DataFrame:
    - Si es DataFrame con columnas year/month/day(/hour/minute/second), arma la fecha.
    - Si no, toma la primera subcolumna y castea.
    """
    if isinstance(col, pd.DataFrame):
        lc = [str(c).strip().lower() for c in col.columns]

        def take(*names):
            for nm in names:
                if nm in lc:
                    return col.iloc[:, lc.index(nm)]
            return None

        y = take("year","año","anio")
        m = take("month","mes")
        d = take("day","día","dia")
        if (y is not None) and (m is not None) and (d is not None):
            h  = take("hour","hora")
            mi = take("minute","min","minuto")
            s  = take("second","seg","segundo")
            parts = {"year": y, "month": m, "day": d}
            if h  is not None: parts["hour"]   = h
            if mi is not None: parts["minute"] = mi
            if s  is not None: parts["second"] = s
            return pd.to_datetime(parts, errors="coerce")
        # fallback: primera subcolumna
        col = col.iloc[:, 0]
    return smart_datetime_cast(col)

def risk_label(score):
    if score >= 80: return "Crítico"
    if score >= 60: return "Alto"
    if score >= 40: return "Medio"
    if score >= 20: return "Bajo"
    return "Muy Bajo"

def show_score(score, title="Riesgo agregado"):
    col1, col2 = st.columns([1,2])
    col1.metric(title, f"{score:.0f}/100", delta=None)
    pct = min(max(float(score),0.0),100.0)/100.0
    bar = "█" * int(pct*25) + "░" * (25-int(pct*25))
    col2.code(f"[{bar}] {score:.0f} ({risk_label(score)})")

def methodology_box(text):
    with st.expander("Metodología aplicada (cómo y por qué)"):
        st.markdown(text)

def recommendations_box(items):
    with st.expander("Recomendaciones"):
        st.markdown("\n".join([f"- {i}" for i in items]))

def explain_findings(df, empty_msg="Sin observaciones."):
    if df is None or df.empty:
        st.success(empty_msg)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.warning(f"Se encontraron **{len(df)}** observaciones.")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar hallazgos (CSV)", data=csv, file_name="hallazgos.csv")

# -------- Encabezado global --------
st.title("🔍 Aprendizaje Colaborativo y Práctico – 2do Parcial")
st.caption("Suite didáctica de herramientas CAAT para auditoría de bases de datos y sistemas (CAAT 1–5).")
show_tips = st.checkbox("🛈 Mostrar ayudas en pantalla", value=True)
st.markdown("> **Nota:** esta app usa archivos **CSV** para simplificar la carga y evitar dependencias de Excel.")

# ========================== CAAT 1 ==========================
st.header("🕒 Módulo 1: Registros fuera de horario (CAAT 1)")
st.caption("Objetivo: identificar eventos fuera del horario laboral definido por el auditor.")

with st.expander("¿Cómo usar este módulo?", expanded=show_tips):
    st.markdown("""
1) **Usuario**: quién ejecuta el evento (ej.: `Usuario`, `user`, `Empleado`).
2) **Fecha/Hora**: momento del evento (ej.: `Timestamp`, `Fecha_Registro`).
3) *(Opcional)* **Acción/Severidad**: si tu log la tiene (ej.: `Severidad`, `Acción`).
4) Define el **horario laboral** y marca **solo días hábiles** si aplica.
5) Revisa **métricas**, **hallazgos** y descarga el **CSV** para evidencias.
""")

f1 = st.file_uploader("Log de actividades (CSV)", type=["csv"], key="c1_file")
df1 = read_any(f1)

if df1 is not None:
    df1 = normalize_cols(df1)
    st.dataframe(df1.head(20), use_container_width=True)
    cols1 = df1.columns.tolist()

    c1a, c1b, c1c = st.columns(3)
    c1_user = c1a.selectbox("Columna Usuario", cols1)
    c1_dt   = c1b.selectbox("Columna Fecha/Hora", cols1)
    c1_act  = c1c.selectbox("Columna Acción (opcional)", ["(ninguna)"]+cols1)

    p1a, p1b, p1c, p1d = st.columns(4)
    start_h = p1a.time_input("Inicio jornada", value=time(8,0))
    end_h   = p1b.time_input("Fin jornada", value=time(18,0))
    weekdays_only = p1c.checkbox("Solo días hábiles (L–V)", True)
    rango = p1d.slider("Top N reincidentes", 5, 50, 10)

    work = df1[[c1_user, c1_dt] + ([] if c1_act=="(ninguna)" else [c1_act])].copy()
    work.rename(columns={c1_user:"user", c1_dt:"dt", c1_act:"action" if c1_act!="(ninguna)" else c1_act}, inplace=True)
    work["dt"] = ensure_datetime_series(work["dt"])

    nat_ratio = work["dt"].isna().mean()
    if nat_ratio > 0.3:
        st.warning(f"⚠️ La columna de **Fecha/Hora** parece no ser válida (~{nat_ratio:.0%} NaT). "
                   "Prueba con otra columna que contenga fecha y hora reales.")

    work["weekday"] = work["dt"].dt.weekday

    # Comparación en minutos desde medianoche
    work["dt_mins"] = (work["dt"].dt.hour.fillna(-1)*60 + work["dt"].dt.minute.fillna(0)).astype(int)
    start_m = start_h.hour*60 + start_h.minute
    end_m   = end_h.hour*60 + end_h.minute
    if end_m >= start_m:
        in_schedule = (work["dt_mins"] >= start_m) & (work["dt_mins"] <= end_m)
    else:  # turno nocturno
        in_schedule = (work["dt_mins"] >= start_m) | (work["dt_mins"] <= end_m)

    if weekdays_only:
        in_schedule &= work["weekday"].between(0,4)

    work["fuera_horario"] = ~in_schedule
    out = work[work["fuera_horario"]].copy().sort_values("dt")

    # Salida legible (fecha/hora)
    out_report = out.copy()
    out_report["fecha"] = out_report["dt"].dt.strftime("%Y-%m-%d").fillna("")
    out_report["hora"]  = out_report["dt"].dt.strftime("%H:%M:%S").fillna("")
    cols_export = ["user","fecha","hora"]
    if "action" in out_report.columns: cols_export += ["action"]
    cols_export += ["weekday","fuera_horario"]
    out_report = out_report[[c for c in cols_export if c in out_report.columns]]

    total = len(work)
    fuera = len(out)
    pct = (fuera/total*100) if total else 0.0
    sc = min(100.0, pct*1.2)

    m1, m2, m3 = st.columns(3)
    m1.metric("Eventos totales", total)
    m2.metric("Fuera de horario", fuera)
    m3.metric("% fuera de horario", f"{pct:.2f}%")

    show_score(sc, "Riesgo agregado CAAT 1")

    st.subheader("Hallazgos")
    if out_report.empty:
        st.success("No se detectaron eventos fuera de horario con los parámetros configurados.")
    else:
        st.dataframe(out_report, use_container_width=True, hide_index=True)
        csv = out_report.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar hallazgos (CSV)", data=csv, file_name="hallazgos_caat1.csv")

        st.subheader("Usuarios reincidentes (Top N)")
        top = out.groupby("user").size().reset_index(name="eventos_fuera").sort_values("eventos_fuera", ascending=False).head(rango)
        st.dataframe(top, use_container_width=True, hide_index=True)

    methodology_box("""
**Qué hace esta prueba:**  
Convierte la hora del evento a **minutos desde medianoche** y contrasta con el rango que definas (y días hábiles).
""")
    recommendations_box([
        "Validar excepciones de horario con RRHH o jefaturas.",
        "Configurar alertas automáticas por accesos fuera de horario.",
        "Aplicar MFA y cierre automático de sesiones inactivas."
    ])

st.divider()

# ========================== CAAT 2 ==========================
st.header("🛡️ Módulo 2: Privilegios y SoD – Segregación de Funciones (CAAT 2)")
st.caption("Objetivo: detectar excesos de privilegios y conflictos de Segregación de Funciones (SoD).")

with st.expander("¿Cómo usar este módulo?", expanded=show_tips):
    st.markdown("""
1) Selecciona **Usuario** y **Rol/Perfil**.
2) Marca **roles críticos** (por columna o listado).
3) Define **reglas SoD** (una por línea: `A -> B`).
4) Revisa usuarios críticos y violaciones; descarga hallazgos.
""")

f2 = st.file_uploader("Usuarios/Roles (CSV)", type=["csv"], key="c2_file")
df2 = read_any(f2)

if df2 is not None:
    df2 = normalize_cols(df2)
    st.dataframe(df2.head(20), use_container_width=True)
    cols2 = df2.columns.tolist()

    c2a, c2b, c2c = st.columns(3)
    c2_user = c2a.selectbox("Columna Usuario", cols2)
    c2_role = c2b.selectbox("Columna Rol/Perfil", cols2)
    c2_crit = c2c.selectbox("Columna indicador 'crítico' (opcional)", ["(ninguna)"]+cols2)

    roles_crit_txt = st.text_area("Roles críticos (si no tienes columna, sepáralos por coma)",
                                  value="ADMIN, APROBADOR_PAGOS, TESORERIA, SUPERUSER")
    crit_list = [r.strip().lower() for r in roles_crit_txt.split(",")] if roles_crit_txt else []

    st.caption("Reglas de Segregación de Funciones (SoD). Una por línea, formato 'A -> B'")
    sod_text = st.text_area("Reglas SoD", value="CREAR_PROVEEDOR -> APROBAR_PAGO\nREGISTRAR_FACTURA -> APROBAR_PAGO")
    sod_pairs = []
    for line in sod_text.splitlines():
        if "->" in line:
            a,b = line.split("->",1)
            sod_pairs.append((a.strip().lower(), b.strip().lower()))

    base = df2[[c2_user, c2_role] + ([] if c2_crit=="(ninguna)" else [c2_crit])].copy()
    base.rename(columns={c2_user:"user", c2_role:"role", c2_crit:"is_critical" if c2_crit!="(ninguna)" else c2_crit}, inplace=True)
    base["role_norm"] = base["role"].astype(str).str.lower().str.strip()

    if c2_crit!="(ninguna)":
        base["critical_flag"] = base["is_critical"].astype(str).str.lower().isin(["1","true","si","sí","y","yes","x"])
    else:
        base["critical_flag"] = base["role_norm"].isin(crit_list)

    crit_users = base[base["critical_flag"]].groupby("user")["role"].apply(list).reset_index(name="roles_criticos")

    user_roles = base.groupby("user")["role_norm"].apply(set).reset_index(name="roles_set")
    def check_sod(roles_set, pairs):
        viol = []
        for a,b in pairs:
            if a in roles_set and b in roles_set:
                viol.append(f"{a} + {b}")
        return viol
    sod_rows = []
    for _, r in user_roles.iterrows():
        viol = check_sod(r["roles_set"], sod_pairs)
        if viol:
            sod_rows.append({"user": r["user"], "violaciones_sod": ", ".join(viol)})
    sod_df = pd.DataFrame(sod_rows)

    total_users = base["user"].nunique()
    n_crit = crit_users["user"].nunique()
    n_sod = len(sod_df)
    score2 = min(100, (n_crit*10) + (n_sod*20))

    m1, m2, m3 = st.columns(3)
    m1.metric("Usuarios totales", total_users)
    m2.metric("Usuarios con roles críticos", n_crit)
    m3.metric("Usuarios con violaciones SoD", n_sod)
    show_score(score2, "Riesgo agregado CAAT 2")

    st.subheader("Usuarios con roles críticos")
    explain_findings(crit_users, "No se detectaron usuarios con roles críticos.")

    st.subheader("Violaciones SoD")
    explain_findings(sod_df, "No se detectaron combinaciones de roles incompatibles.")

    methodology_box("""
**Objetivo:** Identificar excesos de privilegios y conflictos SoD.  
**Procedimiento:** Marcado de roles críticos por lista o columna; agregación de roles por usuario; contraste con reglas SoD.
""")
    recommendations_box([
        "Aplicar principio de privilegio mínimo.",
        "Recertificar accesos trimestralmente.",
        "Documentar excepciones con controles compensatorios."
    ])

st.divider()

# ========================== CAAT 3 ==========================
st.header("🔗 Módulo 3: Conciliación de logs vs transacciones (CAAT 3)")
st.caption("Objetivo: conciliar trazabilidad entre eventos de sistema y registros transaccionales.")

with st.expander("¿Cómo usar este módulo?", expanded=show_tips):
    st.markdown("""
1) Sube **Logs (CSV)** y **Transacciones (CSV)**.
2) Selecciona en ambos: **ID de transacción** y **Fecha/Hora**.
3) Define tolerancia de desfase y revisa: sin correspondencia y desfaces.
""")

f3_logs = st.file_uploader("Logs del sistema (CSV)", type=["csv"], key="c3_logs")
f3_tx   = st.file_uploader("Transacciones (CSV)", type=["csv"], key="c3_tx")
df3L = read_any(f3_logs)
df3T = read_any(f3_tx)

if df3L is not None and df3T is not None:
    df3L, df3T = normalize_cols(df3L), normalize_cols(df3T)
    st.markdown("**Vista rápida – Logs**")
    st.dataframe(df3L.head(15), use_container_width=True)
    st.markdown("**Vista rápida – Transacciones**")
    st.dataframe(df3T.head(15), use_container_width=True)

    c3a, c3b = st.columns(2)
    id_log = c3a.selectbox("Logs: columna ID transacción", df3L.columns.tolist())
    dt_log = c3b.selectbox("Logs: columna fecha/hora", df3L.columns.tolist())
    c3c, c3d = st.columns(2)
    id_tx  = c3c.selectbox("Transacciones: columna ID transacción", df3T.columns.tolist())
    dt_tx  = c3d.selectbox("Transacciones: columna fecha/hora", df3T.columns.tolist())

    L = df3L[[id_log, dt_log]].copy().rename(columns={id_log:"id", dt_log:"dtL"})
    T = df3T[[id_tx, dt_tx]].copy().rename(columns={id_tx:"id", dt_tx:"dtT"})
    L["dtL"] = ensure_datetime_series(L["dtL"])
    T["dtT"] = ensure_datetime_series(T["dtT"])

    idsL = set(L["id"].dropna().astype(str))
    idsT = set(T["id"].dropna().astype(str))

    solo_logs = sorted(list(idsL - idsT))
    solo_tx   = sorted(list(idsT - idsL))
    intersecc = sorted(list(idsL & idsT))

    df_solo_logs = pd.DataFrame({"id_sin_transaccion": solo_logs})
    df_solo_tx   = pd.DataFrame({"id_sin_log": solo_tx})

    Lc = L[L["id"].astype(str).isin(intersecc)].copy()
    Tc = T[T["id"].astype(str).isin(intersecc)].copy()
    Lc_grp = Lc.groupby("id")["dtL"].min().reset_index()
    Tc_grp = Tc.groupby("id")["dtT"].min().reset_index()
    merged = pd.merge(Lc_grp, Tc_grp, on="id", how="inner")
    merged["delay_sec"] = (merged["dtT"] - merged["dtL"]).dt.total_seconds()

    tol_min = st.slider("Tolerancia de desfase (minutos)", 0, 180, 60)
    out_mask = merged["delay_sec"].abs() > (tol_min*60)
    out_count = int(out_mask.sum())

    total_ids = len(idsL | idsT)
    m_logs = len(solo_logs)
    m_tx   = len(solo_tx)
    score3 = min(100, (m_logs*10) + (m_tx*10) + (out_count*2))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("IDs totales (consolidados)", total_ids)
    m2.metric("IDs en logs sin transacción", m_logs)
    m3.metric("IDs en transacción sin log", m_tx)
    m4.metric(f"Desfaces > {tol_min} min", out_count)
    show_score(score3, "Riesgo agregado CAAT 3")

    st.subheader("IDs presentes solo en logs")
    explain_findings(df_solo_logs, "No hay IDs exclusivos en logs.")

    st.subheader("IDs presentes solo en transacciones")
    explain_findings(df_solo_tx, "No hay IDs exclusivos en transacciones.")

    st.subheader("Desfaces (primer log vs primera transacción)")
    st.caption("delay_sec > 0: transacción posterior al log; < 0: transacción antes del log (anómalo).")
    st.dataframe(merged.sort_values("delay_sec", key=lambda s: s.abs(), ascending=False).head(200), use_container_width=True)

    # Exportación opcional de conciliación
    merged_exp = merged.copy()
    merged_exp["fecha_log"] = pd.to_datetime(merged_exp["dtL"]).dt.strftime("%Y-%m-%d").fillna("")
    merged_exp["hora_log"]  = pd.to_datetime(merged_exp["dtL"]).dt.strftime("%H:%M:%S").fillna("")
    merged_exp["fecha_tx"]  = pd.to_datetime(merged_exp["dtT"]).dt.strftime("%Y-%m-%d").fillna("")
    merged_exp["hora_tx"]   = pd.to_datetime(merged_exp["dtT"]).dt.strftime("%H:%M:%S").fillna("")
    merged_exp = merged_exp[["id","fecha_log","hora_log","fecha_tx","hora_tx","delay_sec"]]
    csv = merged_exp.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar conciliación (CSV)", data=csv, file_name="conciliacion_caat3.csv")

    methodology_box("""
**Objetivo:** Conciliar trazabilidad entre logs y transacciones.  
**Procedimiento:** Match por ID; listar sin correspondencia; calcular desfase y marcar fuera de tolerancia.
""")
    recommendations_box([
        "Asegurar correlación 1:1 entre eventos y asientos.",
        "Implementar alertas por desfases fuera de tolerancia."
    ])

st.divider()

# ========================== CAAT 4 ==========================
st.header("📈 Módulo 4: Variación inusual de pagos – outliers (CAAT 4)")
st.caption("Objetivo: encontrar picos/caídas atípicas en pagos mensuales por proveedor.")

with st.expander("¿Cómo usar este módulo?", expanded=show_tips):
    st.markdown("""
1) Sube **Pagos (CSV)**.
2) Selecciona **Proveedor**, **Fecha** y **Monto**.
3) Ajusta el **umbral de outliers (|z| robusto)**.
""")

f4 = st.file_uploader("Histórico de pagos (CSV)", type=["csv"], key="c4_file")
df4 = read_any(f4)

if df4 is not None:
    df4 = normalize_cols(df4)
    st.dataframe(df4.head(20), use_container_width=True)
    cols4 = df4.columns.tolist()

    c4a, c4b, c4c = st.columns(3)
    c4_sup = c4a.selectbox("Columna Proveedor", cols4)
    c4_dt  = c4b.selectbox("Columna Fecha", cols4)
    c4_amt = c4c.selectbox("Columna Monto", cols4)

    dfp = df4[[c4_sup,c4_dt,c4_amt]].copy().rename(columns={c4_sup:"proveedor", c4_dt:"fecha", c4_amt:"monto"})
    dfp["fecha"] = ensure_datetime_series(dfp["fecha"])
    dfp["monto"] = pd.to_numeric(dfp["monto"], errors="coerce").fillna(0.0)

    fcol1, fcol2 = st.columns(2)
    min_date = pd.to_datetime(dfp["fecha"].min())
    max_date = pd.to_datetime(dfp["fecha"].max())
    if pd.isna(min_date): min_date = datetime.today()
    if pd.isna(max_date): max_date = datetime.today()
    date_range = fcol1.date_input("Rango de fechas", value=(min_date.date(), max_date.date()))
    proveedor_sel = fcol2.multiselect("Filtrar proveedores", sorted(dfp["proveedor"].dropna().unique().tolist()))

    if len(date_range)==2:
        d1 = pd.to_datetime(date_range[0]); d2 = pd.to_datetime(date_range[1])
        dfp = dfp[(dfp["fecha"]>=d1) & (dfp["fecha"]<=d2)]
    if proveedor_sel:
        dfp = dfp[dfp["proveedor"].isin(proveedor_sel)]

    dfp["year_month"] = dfp["fecha"].dt.to_period("M").astype(str)
    monthly = dfp.groupby(["proveedor","year_month"], as_index=False)["monto"].sum()

    z_thr = st.slider("Umbral |z| (robusto)", 2.0, 5.0, 3.5, 0.1,
                      help=("Valores mayores al umbral se marcan como atípicos (mediana/MAD)." if show_tips else None))

    def robust_z_values(arr):
        arr = np.asarray(arr, dtype=float)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) or 1.0
        return (arr - med) / (1.4826*mad)

    out_rows = []
    for sup, grp in monthly.groupby("proveedor"):
        z = robust_z_values(grp["monto"].values)
        grp2 = grp.copy(); grp2["zscore"] = z
        suspicious = grp2[np.abs(grp2["zscore"]) >= z_thr].copy()
        if not suspicious.empty: out_rows.append(suspicious)
    out_df = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["proveedor","year_month","monto","zscore"])

    total_rows = len(dfp)
    n_out = len(out_df)
    score4 = min(100, n_out * 2 + (out_df["monto"].sum()/ (dfp["monto"].sum()+1e-9)) * 100)

    m1, m2, m3 = st.columns(3)
    m1.metric("Pagos (registros)", total_rows)
    m2.metric("Anomalías detectadas", n_out)
    m3.metric("% monto anómalo", f"{(out_df['monto'].sum()/(dfp['monto'].sum()+1e-9))*100:0.2f}%")
    show_score(score4, "Riesgo agregado CAAT 4")

    st.subheader("Pagos mensuales anómalos por proveedor")
    explain_findings(out_df.sort_values("zscore", key=lambda s: s.abs(), ascending=False), "No se detectaron variaciones inusuales bajo el umbral configurado.")

    st.subheader("Tendencia mensual (monto total)")
    trend = dfp.groupby("year_month", as_index=False)["monto"].sum().sort_values("year_month")
    if not trend.empty:
        st.line_chart(trend.set_index("year_month"))

    methodology_box("""
**Objetivo:** Detectar picos o caídas inusuales en pagos.  
**Procedimiento:** Agregación mensual por proveedor; z-score robusto (mediana/MAD) con umbral ajustable.
""")
    recommendations_box([
        "Solicitar soportes para meses con picos.",
        "Cruzar con órdenes de compra y recepción.",
        "Revisar proveedores con anomalías repetidas."
    ])

st.divider()

# ========================== CAAT 5 ==========================
st.header("✅ Módulo 5: Criterios de selección de proveedores (CAAT 5)")
st.caption("Objetivo: validar criterios mínimos de selección y permanencia de proveedores.")

with st.expander("¿Cómo usar este módulo?", expanded=show_tips):
    st.markdown("""
1) Sube maestro de **Proveedores (CSV)**.
2) Selecciona columnas: Proveedor, RUC, y (opcional) Blacklist, Fecha vigencia, Cuenta validada, Aprobado.
3) Marca criterios a verificar y fija **fecha de corte**.
""")

f5 = st.file_uploader("Maestro de proveedores (CSV)", type=["csv"], key="c5_file")
df5 = read_any(f5)

if df5 is not None:
    df5 = normalize_cols(df5)
    st.dataframe(df5.head(20), use_container_width=True)
    cols5 = df5.columns.tolist()

    c5a, c5b, c5c = st.columns(3)
    c5_sup = c5a.selectbox("Columna Proveedor", cols5)
    c5_ruc = c5b.selectbox("Columna RUC (tax id)", cols5)
    c5_bl  = c5c.selectbox("Columna Blacklist/Sanción (1/0 o Sí/No)", ["(ninguna)"]+cols5)

    c5d, c5e, c5f = st.columns(3)
    c5_doc  = c5d.selectbox("Columna Fecha vigencia documento", ["(ninguna)"]+cols5)
    c5_bank = c5e.selectbox("Columna Cuenta bancaria validada (1/0, Sí/No)", ["(ninguna)"]+cols5)
    c5_appr = c5f.selectbox("Columna Aprobado/Precalificado (1/0, Sí/No)", ["(ninguna)"]+cols5)

    rules = st.multiselect(
        "Criterios a verificar",
        ["RUC válido (13 dígitos numéricos)", "No estar en Blacklist", "Documento vigente", "Cuenta bancaria validada", "Proveedor aprobado/precalificado"],
        default=["RUC válido (13 dígitos numéricos)", "No estar en Blacklist", "Documento vigente", "Cuenta bancaria validada", "Proveedor aprobado/precalificado"]
    )
    corte = st.date_input("Fecha de corte de vigencia documental", value=datetime.today().date())

    base = df5.copy()
    base.rename(columns={c5_sup:"proveedor", c5_ruc:"ruc"}, inplace=True)

    def as_bool(s):
        return s.astype(str).str.lower().isin(["1","true","si","sí","y","yes","x"])

    checks = pd.DataFrame()
    checks["proveedor"] = base["proveedor"]
    checks["ruc"] = base["ruc"].astype(str)

    if "RUC válido (13 dígitos numéricos)" in rules:
        checks["ruc_valido"] = checks["ruc"].str.isnumeric() & (checks["ruc"].str.len()==13)

    if c5_bl!="(ninguna)" and "No estar en Blacklist" in rules:
        checks["no_blacklist"] = ~as_bool(base[c5_bl])

    if c5_doc!="(ninguna)" and "Documento vigente" in rules:
        venc = ensure_datetime_series(base[c5_doc])
        checks["doc_vigente"] = venc >= pd.Timestamp(corte)

    if c5_bank!="(ninguna)" and "Cuenta bancaria validada" in rules:
        checks["cuenta_val"] = as_bool(base[c5_bank])

    if c5_appr!="(ninguna)" and "Proveedor aprobado/precalificado" in rules:
        checks["aprobado"] = as_bool(base[c5_appr])

    crit_cols = [c for c in checks.columns if c not in ["proveedor","ruc"]]
    if crit_cols:
        checks["cumple_todo"] = checks[crit_cols].all(axis=1)
        faltas = checks[~checks["cumple_todo"]].copy()
    else:
        faltas = pd.DataFrame(columns=checks.columns)

    total = len(checks)
    n_fallas = len(faltas)
    score5 = min(100, n_fallas*5)

    m1, m2, m3 = st.columns(3)
    m1.metric("Proveedores evaluados", total)
    m2.metric("Proveedores con incumplimientos", n_fallas)
    m3.metric("Criterios verificados", len(crit_cols))
    show_score(score5, "Riesgo agregado CAAT 5")

    st.subheader("Proveedores con incumplimientos")
    if not faltas.empty:
        det = checks.merge(base, left_on="proveedor", right_on="proveedor", how="left")
        det = det[det["proveedor"].isin(faltas["proveedor"])]
        st.dataframe(det, use_container_width=True)
        csv = det.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar incumplimientos (CSV)", data=csv, file_name="proveedores_incumplidos.csv")
    else:
        st.success("Todos los proveedores cumplen los criterios seleccionados.")

    methodology_box("""
**Objetivo:** Validar criterios mínimos de selección y permanencia de proveedores.  
**Procedimiento:** Verificación de formato RUC, blacklist, vigencia documental a una fecha de corte, validación bancaria y aprobación.
""")
    recommendations_box([
        "Bloquear compras a proveedores no aprobados o con documentos vencidos.",
        "Automatizar verificación de RUC/cuenta con fuentes oficiales.",
        "Programar recordatorios de renovación documental."
    ])

st.markdown("---")
st.caption("© 2025 – Proyecto académico. Esta app trabaja con CSV para evitar dependencias de Excel.")

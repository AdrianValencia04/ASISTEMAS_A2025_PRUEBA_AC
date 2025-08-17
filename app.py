# -*- coding: utf-8 -*-
# App: Aprendizaje Colaborativo y Práctico – 2do Parcial
# Suite de CAATs (1–5) con ayudas, validaciones, reportes por módulo
# y reporte consolidado con resumen ejecutivo.

import io
import zipfile
from datetime import datetime, time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# ==============
# CONFIG INICIAL
# ==============
st.set_page_config(
    page_title="Aprendizaje Colaborativo y Práctico – 2do Parcial",
    page_icon="🔎",
    layout="wide",
)

TITLE = "Aprendizaje Colaborativo y Práctico – 2do Parcial"
SUBTITLE = "Suite de pruebas CAAT (1–5) para auditoría de datos con guías, validaciones y reportes."

if "reports" not in st.session_state:
    # Lista de tuplas (titulo_visible, dataframe, nombre_archivo_sugerido)
    st.session_state.reports: List[Tuple[str, pd.DataFrame, str]] = []
if "metrics" not in st.session_state:
    # Métricas por módulo para el resumen final
    st.session_state.metrics = {}

st.title(f"🔍 {TITLE}")
st.caption(SUBTITLE)

show_help = st.checkbox("Mostrar ayudas en pantalla", value=True)

st.divider()


# =================
# FUNCIONES COMUNES
# =================
def info_help(msg: str):
    if show_help:
        st.info(msg)


def warn_help(msg: str):
    st.warning(msg)


def error_help(msg: str):
    st.error(msg)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def ensure_datetime(s: pd.Series) -> pd.Series:
    """Convierte a datetime con coerción; retorna serie (puede contener NaT)."""
    if not isinstance(s, pd.Series):
        return pd.Series([], dtype="datetime64[ns]")
    return pd.to_datetime(s, errors="coerce")


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """z-score robusto usando mediana y MAD (sin scipy)."""
    x = np.asarray(x, dtype="float64")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def add_report(title: str, df: pd.DataFrame, filename: str):
    if df is None or df.empty:
        return
    st.session_state.reports.append((title, df.copy(), filename))


def make_zip_reports() -> bytes:
    """Crea un ZIP en memoria con los CSVs de st.session_state.reports y un resumen."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        # Añadir todos los CSVs
        for title, df, fname in st.session_state.reports:
            z.writestr(fname, df.to_csv(index=False))
        # Resumen ejecutivo
        resumen = build_resumen_ejecutivo_text()
        z.writestr("00_resumen_ejecutivo.txt", resumen)
    buffer.seek(0)
    return buffer.read()


def build_resumen_ejecutivo_text() -> str:
    """Genera un texto de resumen en base a st.session_state.metrics."""
    mx = st.session_state.metrics

    # Colecta puntuaciones (0-100) por módulo si están disponibles
    puntajes = []
    for k in ["CAAT1", "CAAT2", "CAAT3", "CAAT4", "CAAT5"]:
        if k in mx and "score" in mx[k]:
            puntajes.append(mx[k]["score"])
    global_score = round(np.mean(puntajes), 2) if puntajes else 0.0

    nivel = (
        "Crítico" if global_score >= 80 else
        "Alto" if global_score >= 60 else
        "Medio" if global_score >= 40 else
        "Bajo" if global_score >= 20 else
        "Muy Bajo"
    )

    lines = []
    lines.append("RESUMEN EJECUTIVO – Suite CAAT (1–5)")
    lines.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Índice global de riesgo: {global_score}/100 ({nivel})")
    lines.append("")
    lines.append("Detalle por módulo:")
    for k, v in mx.items():
        desc = v.get("desc", "")
        score = v.get("score", "-")
        extra = v.get("extra", "")
        lines.append(f"  - {k}: {score}/100 {desc}")
        if extra:
            lines.append(f"      {extra}")
    lines.append("")
    lines.append("Conclusión automática:")
    if global_score >= 80:
        lines.append("- Se identifican riesgos críticos. Priorizar acciones inmediatas en los módulos con mayor puntaje.")
    elif global_score >= 60:
        lines.append("- Riesgo alto. Se recomienda implementar controles adicionales y seguimiento continuo.")
    elif global_score >= 40:
        lines.append("- Riesgo medio. Ajustar controles y monitorear áreas con hallazgos frecuentes.")
    else:
        lines.append("- Riesgo bajo/muy bajo. Mantener controles actuales y monitoreo periódico.")
    lines.append("")
    lines.append("Recomendaciones generales:")
    lines.append("- Configurar alertas sobre actividades fuera de horario o picos de pagos.")
    lines.append("- Revisar roles críticos y reglas SoD, reducir accesos innecesarios.")
    lines.append("- Conciliar sistemáticamente logs vs. transacciones con tolerancias definidas.")
    lines.append("- Verificar criterios de selección de proveedores y vigencias.")
    return "\n".join(lines)


def read_table_uploader(label: str, key: str) -> Optional[pd.DataFrame]:
    """Uploader genérico: lee CSV o XLSX usando openpyxl si aplica."""
    file = st.file_uploader(label, type=["csv", "xlsx"], key=key)
    if not file:
        return None

    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file)
            return df
        elif name.endswith(".xlsx"):
            # Intento de elegir hoja si hay varias
            try:
                file.seek(0)
                xls = pd.ExcelFile(file, engine="openpyxl")
                sheet = st.selectbox(
                    "Hoja de Excel", xls.sheet_names,
                    help="Selecciona la hoja que contiene tus datos"
                )
                df = xls.parse(sheet)
                return df
            except Exception as e:
                error_help("No se pudo leer el Excel. Asegúrate de que el archivo no esté protegido y que sea .xlsx válido.")
                st.exception(e)
                return None
        else:
            warn_help("Formato no reconocido. Sube un CSV o XLSX.")
            return None
    except Exception:
        error_help(
            "No se pudo procesar el archivo. Si es Excel, verifica que el entorno tenga 'openpyxl' instalado. "
            "Si es CSV, revisa el separador y el encoding."
        )
        return None


def column_picker(df: pd.DataFrame, label: str, options_hint: List[str], optional=False) -> Optional[str]:
    """Selector de columna con ayuda y búsqueda por nombre aproximado."""
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    default = None
    # Heurística simple: intenta preseleccionar por nombres sugeridos
    for h in options_hint:
        for c in cols:
            if h.lower() == str(c).lower():
                default = c
                break
        if default is not None:
            break
    idx = cols.index(default) if default in cols else 0
    selected = st.selectbox(label, cols, index=idx if cols else 0)
    if not selected and not optional:
        warn_help(f"Selecciona una columna para '{label}'.")
    return selected


def hours_slider(label: str, default_h: str) -> time:
    """Selector de hora 'HH:MM'."""
    hh, mm = map(int, default_h.split(":"))
    return time(hh, mm)


def validate_required_columns(df: pd.DataFrame, required: List[str]) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        error_help(f"Faltan columnas requeridas: {missing}")
        return False
    return True


def score_from_pct(pct: float) -> int:
    """Convierte un % a un score 0–100 simple (más % => más riesgo)."""
    pct = 0 if np.isnan(pct) else pct
    return int(np.clip(round(pct), 0, 100))


def score_from_count(n: int, scale: int = 100) -> int:
    """Score aproximado por cantidad (recortado a 0–100)."""
    return int(np.clip(round(100 * (n / max(scale, 1))), 0, 100))


# ======================
# MÓDULO 1 – CAAT: HORARIO
# ======================
st.header("⏰ Módulo 1: Registros fuera de horario (CAAT 1)")

with st.expander("¿Cómo usar este módulo?", expanded=show_help):
    st.markdown(
        """
**Objetivo:** detectar registros realizados **fuera del horario laboral** que definas.
1. Sube tu archivo (CSV/XLSX).
2. Elige **Usuario** y **Fecha/Hora** (opcional **Acción/Severidad**).
3. Define el **horario** y marca **Solo días hábiles (L–V)** si aplica.
4. Revisa hallazgos, métricas y descarga el **CSV**.
        """
    )

df1 = read_table_uploader("Log de actividades (CSV/XLSX)", key="caat1_upl")
if df1 is not None and not df1.empty:
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        col_user = column_picker(df1, "Columna Usuario", ["usuario", "user", "empleado"])
    with c2:
        col_dt = column_picker(df1, "Columna Fecha/Hora", ["timestamp", "fecha_registro", "fecha", "datetime"])
    with c3:
        col_action = column_picker(df1, "Columna Acción (opcional)", ["accion", "acción", "severidad", "nivel"], optional=True)

    c4, c5, c6 = st.columns([1,1,1])
    with c4:
        start_h = st.selectbox("Inicio jornada", [f"{h:02d}:{m:02d}" for h in range(0,24) for m in (0,15,30,45)], index=32)
    with c5:
        end_h   = st.selectbox("Fin jornada", [f"{h:02d}:{m:02d}" for h in range(0,24) for m in (0,15,30,45)], index=72)
    with c6:
        only_weekdays = st.checkbox("Solo días hábiles (L–V)", value=True)

    # Validaciones
    valid = True
    if col_user not in df1.columns or col_dt not in df1.columns:
        error_help("Selecciona correctamente **Usuario** y **Fecha/Hora**.")
        valid = False

    work = df1.copy()
    if valid:
        work["user"] = work[col_user].astype(str)
        work["dt"] = ensure_datetime(work[col_dt])
        if work["dt"].isna().all():
            error_help("La columna seleccionada para **Fecha/Hora** no tiene fechas válidas. Elige otra columna.")
            valid = False

    if valid:
        work["hour"] = work["dt"].dt.hour + work["dt"].dt.minute / 60.0
        work["weekday"] = work["dt"].dt.weekday  # 0=Lunes ... 6=Domingo
        start_hh, start_mm = map(int, start_h.split(":"))
        end_hh, end_mm     = map(int, end_h.split(":"))
        start_dec = start_hh + start_mm/60.0
        end_dec   = end_hh + end_mm/60.0

        if start_dec <= end_dec:
            in_schedule = (work["hour"] >= start_dec) & (work["hour"] <= end_dec)
        else:
            # Jornada nocturna cruzando medianoche
            in_schedule = (work["hour"] >= start_dec) | (work["hour"] <= end_dec)

        if only_weekdays:
            in_schedule = in_schedule & (work["weekday"] <= 4)

        work["fuera_horario"] = ~in_schedule
        if col_action in work.columns:
            work["accion"] = work[col_action].astype(str)
        else:
            work["accion"] = ""

        hall = work.loc[work["fuera_horario"], ["user","dt","weekday","accion","fuera_horario"]].copy()
        hall["fecha"] = hall["dt"].dt.date.astype(str)
        hall["hora"]  = hall["dt"].dt.time.astype(str)
        hall = hall.drop(columns=["dt"])

        total = len(work)
        fuera = len(hall)
        pct = 100.0 * fuera / total if total else 0.0

        cA, cB, cC = st.columns(3)
        cA.metric("Eventos totales", f"{total:,}")
        cB.metric("Fuera de horario", f"{fuera:,}")
        cC.metric("% fuera de horario", f"{pct:.2f}%")

        # Score y metadatos para resumen final
        st.session_state.metrics["CAAT1"] = {
            "score": score_from_pct(pct),
            "desc": "(más alto => más riesgo)",
            "extra": f"Fuera de horario: {fuera}/{total} ({pct:.2f}%)."
        }

        st.subheader("Hallazgos")
        st.dataframe(hall, use_container_width=True, height=300)

        # Descarga
        fname = f"reporte_caat1_{datetime.now():%Y%m%d_%H%M}.csv"
        st.download_button("⬇️ Descargar hallazgos (CSV)", data=df_to_csv_bytes(hall), file_name=fname, mime="text/csv")
        add_report("CAAT1_Fuera_Horario", hall, fname)

st.divider()


# ==========================
# MÓDULO 2 – CAAT: PRIVILEGIOS
# ==========================
st.header("🛡️ Módulo 2: Auditoría de privilegios (roles críticos y SoD) (CAAT 2)")

with st.expander("¿Cómo usar este módulo?", expanded=show_help):
    st.markdown(
        """
**Objetivo:** identificar **roles críticos** y violaciones de **Segregación de Funciones (SoD)**.
1. Sube tu maestro de **Usuarios/Roles**.
2. Elige **Usuario** y **Rol**.
3. Define **roles críticos** (lista) o marca si tienes una columna de crítico.
4. Escribe **reglas SoD** (una por línea) en formato `ROL_A -> ROL_B`.
        """
    )

df2 = read_table_uploader("Usuarios/Roles (CSV/XLSX)", key="caat2_upl")
if df2 is not None and not df2.empty:
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        col_user2 = column_picker(df2, "Columna Usuario", ["usuario","user","empleado"])
    with c2:
        col_role2 = column_picker(df2, "Columna Rol", ["rol","role","perfil"])
    with c3:
        col_crit2 = column_picker(df2, "Columna es_crítico (opcional)", ["critico","crítico","es_critico","is_critical"], optional=True)

    roles_crit_txt = st.text_input("Lista de roles críticos (separados por coma)", value="ADMIN, SUPERUSER")
    sod_rules_txt  = st.text_area("Reglas SoD (una por línea, formato ROL_A -> ROL_B)", height=100,
                                  value="CREAR_PROVEEDOR -> APROBAR_PAGO")
    # Normaliza
    if col_user2 not in df2.columns or col_role2 not in df2.columns:
        error_help("Selecciona correctamente **Usuario** y **Rol**.")
    else:
        base = df2.copy()
        base["user"] = base[col_user2].astype(str)
        base["role"] = base[col_role2].astype(str)

        # Críticos
        crit_set = set([r.strip().lower() for r in roles_crit_txt.split(",") if r.strip()]) if roles_crit_txt else set()
        if col_crit2 in base.columns:
            base["is_crit"] = base[col_crit2].astype(str).str.lower().isin(["1","true","si","sí","y","yes"])
        else:
            base["is_crit"] = base["role"].str.lower().isin(crit_set)

        crit_df = base.loc[base["is_crit"], ["user","role"]].groupby("user")["role"].agg(lambda x: "; ".join(sorted(set(x)))).reset_index()
        crit_df.rename(columns={"role":"roles_criticos"}, inplace=True)

        # SoD
        rules = []
        for line in sod_rules_txt.splitlines():
            if "->" in line:
                a,b = line.split("->",1)
                a,b = a.strip(), b.strip()
                if a and b:
                    rules.append((a,b))
        if not rules and show_help:
            warn_help("No se detectaron reglas SoD válidas. Usa el formato `ROL_A -> ROL_B`.")

        # Roles por usuario
        roles_user = base.groupby("user")["role"].agg(set).reset_index()

        def user_has_rule(rset: set, a: str, b: str) -> bool:
            return a in rset and b in rset

        sod_rows = []
        for _, row in roles_user.iterrows():
            u, rset = row["user"], row["role"]
            for a,b in rules:
                if user_has_rule(rset, a, b) or user_has_rule(rset, b, a):
                    sod_rows.append(
                        {
                            "usuario": u,
                            "violacion": f"{a} + {b}",
                            "roles_usuario": "; ".join(sorted(rset)),
                            "detalle": "Combinación incompatible según regla SoD definida"
                        }
                    )
        sod_df = pd.DataFrame(sod_rows)

        # Métricas
        n_crit = len(crit_df)
        n_sod  = len(sod_df)
        cA, cB = st.columns(2)
        cA.metric("Usuarios con roles críticos", f"{n_crit:,}")
        cB.metric("Violaciones SoD", f"{n_sod:,}")

        st.session_state.metrics["CAAT2"] = {
            "score": score_from_count(n_crit + n_sod, scale=max(len(roles_user),1)*2),
            "desc": "(más alto => más riesgo)",
            "extra": f"Críticos={n_crit}, SoD={n_sod}"
        }

        st.subheader("Usuarios con roles críticos")
        st.dataframe(crit_df, use_container_width=True, height=230)
        f1 = f"reporte_caat2_criticos_{datetime.now():%Y%m%d_%H%M}.csv"
        st.download_button("⬇️ Descargar críticos (CSV)", df_to_csv_bytes(crit_df), f1, "text/csv")
        add_report("CAAT2_Criticos", crit_df, f1)

        st.subheader("Violaciones SoD")
        st.dataframe(sod_df, use_container_width=True, height=230)
        f2 = f"reporte_caat2_sod_{datetime.now():%Y%m%d_%H%M}.csv"
        st.download_button("⬇️ Descargar SoD (CSV)", df_to_csv_bytes(sod_df), f2, "text/csv")
        add_report("CAAT2_SoD", sod_df, f2)

st.divider()


# ===================================
# MÓDULO 3 – CAAT: CONCILIACIÓN LOGS/TX
# ===================================
st.header("🔗 Módulo 3: Conciliación de logs vs transacciones (CAAT 3)")

with st.expander("¿Cómo usar este módulo?", expanded=show_help):
    st.markdown(
        """
**Objetivo:** conciliar **logs** del sistema vs **transacciones**, buscar **IDs faltantes** y **desfases de tiempo**.
1. Sube **Logs** y **Transacciones**.
2. En ambos: elige **ID** y **Fecha/Hora**.
3. Define la **tolerancia** (minutos) para marcar desface.
        """
    )

c1, c2 = st.columns(2)
with c1:
    df3_logs = read_table_uploader("📄 Logs (CSV/XLSX)", key="caat3_logs")
with c2:
    df3_tx   = read_table_uploader("💰 Transacciones (CSV/XLSX)", key="caat3_tx")

if df3_logs is not None and not df3_logs.empty and df3_tx is not None and not df3_tx.empty:
    c1, c2 = st.columns(2)
    with c1:
        id_logs = column_picker(df3_logs, "ID (logs)", ["id","id_registro","cod"])
        dt_logs = column_picker(df3_logs, "Fecha/Hora (logs)", ["timestamp","fecha","datetime","fecha_registro"])
    with c2:
        id_tx = column_picker(df3_tx, "ID (tx)", ["id","id_registro","cod"])
        dt_tx = column_picker(df3_tx, "Fecha/Hora (tx)", ["timestamp","fecha","datetime","fecha_registro"])

    tol_min = st.slider("Tolerancia de desfase (minutos)", 0, 240, 60)

    # Validaciones mínimas
    ok = True
    for df, idc, dtc, tag in [(df3_logs, id_logs, dt_logs, "logs"), (df3_tx, id_tx, dt_tx, "tx")]:
        if idc not in df.columns or dtc not in df.columns:
            error_help(f"Selecciona correctamente ID y Fecha/Hora para {tag}.")
            ok = False

    if ok:
        L = df3_logs.copy()
        T = df3_tx.copy()
        L["id"] = L[id_logs].astype(str)
        L["dt"] = ensure_datetime(L[dt_logs])
        T["id"] = T[id_tx].astype(str)
        T["dt"] = ensure_datetime(T[dt_tx])

        L = L[~L["dt"].isna()]
        T = T[~T["dt"].isna()]

        ids_logs = set(L["id"].unique())
        ids_tx   = set(T["id"].unique())

        only_logs = sorted(list(ids_logs - ids_tx))
        only_tx   = sorted(list(ids_tx - ids_logs))

        df_only_logs = pd.DataFrame({"id": only_logs, "detalle": "ID en logs sin transacción asociada"})
        df_only_tx   = pd.DataFrame({"id": only_tx,   "detalle": "ID en transacción sin log asociado"})

        # Para desfases: emparejamos por ID (primer match por sencillez)
        M = pd.merge(L[["id","dt"]], T[["id","dt"]], on="id", suffixes=("_log","_tx"))
        M["delay_sec"] = (M["dt_tx"] - M["dt_log"]).dt.total_seconds().astype(float)
        M["fuera_tolerancia"] = np.abs(M["delay_sec"]) > tol_min*60
        df_desf = M.loc[M["fuera_tolerancia"], :].copy()

        cA, cB, cC = st.columns(3)
        cA.metric("IDs solo en logs", f"{len(df_only_logs):,}")
        cB.metric("IDs solo en tx", f"{len(df_only_tx):,}")
        cC.metric("Desfaces > tolerancia", f"{len(df_desf):,}")

        st.session_state.metrics["CAAT3"] = {
            "score": score_from_count(len(df_only_logs) + len(df_only_tx) + len(df_desf), scale=max(len(M),1)),
            "desc": "(más alto => más riesgo)",
            "extra": f"SoloLogs={len(df_only_logs)}, SoloTx={len(df_only_tx)}, Desfaces={len(df_desf)}"
        }

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Solo en logs")
            st.dataframe(df_only_logs, use_container_width=True, height=220)
            f1 = f"reporte_caat3_solo_logs_{datetime.now():%Y%m%d_%H%M}.csv"
            st.download_button("⬇️ Descargar (CSV)", df_to_csv_bytes(df_only_logs), f1, "text/csv")
            add_report("CAAT3_SoloLogs", df_only_logs, f1)

        with colB:
            st.subheader("Solo en transacciones")
            st.dataframe(df_only_tx, use_container_width=True, height=220)
            f2 = f"reporte_caat3_solo_tx_{datetime.now():%Y%m%d_%H%M}.csv"
            st.download_button("⬇️ Descargar (CSV)", df_to_csv_bytes(df_only_tx), f2, "text/csv")
            add_report("CAAT3_SoloTx", df_only_tx, f2)

        st.subheader("Desfases fuera de tolerancia")
        # Agrega columnas legibles
        if not df_desf.empty:
            df_desf["fecha_log"] = df_desf["dt_log"].dt.date.astype(str)
            df_desf["hora_log"]  = df_desf["dt_log"].dt.time.astype(str)
            df_desf["fecha_tx"]  = df_desf["dt_tx"].dt.date.astype(str)
            df_desf["hora_tx"]   = df_desf["dt_tx"].dt.time.astype(str)
        st.dataframe(df_desf[["id","fecha_log","hora_log","fecha_tx","hora_tx","delay_sec","fuera_tolerancia"]],
                     use_container_width=True, height=240)
        f3 = f"reporte_caat3_desfaces_{datetime.now():%Y%m%d_%H%M}.csv"
        st.download_button("⬇️ Descargar desfases (CSV)", df_to_csv_bytes(df_desf), f3, "text/csv")
        add_report("CAAT3_Desfases", df_desf, f3)

st.divider()


# ======================================
# MÓDULO 4 – CAAT: VARIACIÓN INUSUAL PAGO
# ======================================
st.header("📈 Módulo 4: Variación inusual de pagos – outliers (CAAT 4)")

with st.expander("¿Cómo usar este módulo?", expanded=show_help):
    st.markdown(
        """
**Objetivo:** detectar **meses atípicos** por proveedor usando z-score robusto (mediana y MAD).
1. Sube tu histórico de **Pagos**.
2. Elige **Proveedor**, **Fecha** y **Monto**.
3. Ajusta el **umbral |z|** (típico 3.0–3.5).
        """
    )

df4 = read_table_uploader("Histórico de pagos (CSV/XLSX)", key="caat4_upl")
if df4 is not None and not df4.empty:
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        col_prov = column_picker(df4, "Proveedor", ["proveedor","vendor","nombre"])
    with c2:
        col_date = column_picker(df4, "Fecha", ["fecha","fecha_pago","date","timestamp"])
    with c3:
        col_amt  = column_picker(df4, "Monto", ["monto","importe","amount","valor"])

    thr = st.slider("Ajusta el umbral de outliers (|z| robusto)", 2.0, 6.0, 3.5, 0.1)

    ok = True
    for c in [col_prov, col_date, col_amt]:
        if c not in df4.columns:
            ok = False
    if not ok:
        error_help("Selecciona correctamente Proveedor, Fecha y Monto.")
    else:
        P = df4.copy()
        P["proveedor"] = P[col_prov].astype(str)
        P["dt"] = ensure_datetime(P[col_date])
        P[col_amt] = pd.to_numeric(P[col_amt], errors="coerce")

        P = P[~P["dt"].isna()]
        P = P[~P[col_amt].isna()]
        if P.empty:
            error_help("No hay registros válidos con fecha y monto numérico.")
        else:
            P["year_month"] = P["dt"].dt.to_period("M").astype(str)
            g = P.groupby(["proveedor","year_month"], as_index=False)[col_amt].sum()
            g.rename(columns={col_amt: "monto"}, inplace=True)

            out_rows = []
            for prov, sub in g.groupby("proveedor"):
                z = robust_zscore(sub["monto"].values)
                sub = sub.copy()
                sub["zscore"] = z
                sub["is_outlier"] = np.abs(z) >= thr
                out_rows.append(sub)
            OUT = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()

            anomalies = OUT.loc[OUT["is_outlier"]].copy()
            n_anom = len(anomalies)
            pct_anom_monto = 100.0 * anomalies["monto"].sum() / OUT["monto"].sum() if len(OUT) else 0.0

            cA, cB = st.columns(2)
            cA.metric("Anomalías detectadas", f"{n_anom:,}")
            cB.metric("% monto anómalo", f"{pct_anom_monto:.2f}%")

            st.session_state.metrics["CAAT4"] = {
                "score": score_from_count(n_anom, scale=max(len(OUT),1)),
                "desc": "(más alto => más riesgo)",
                "extra": f"Umbral |z|={thr}."
            }

            st.subheader("Meses atípicos por proveedor")
            st.dataframe(anomalies, use_container_width=True, height=260)
            f1 = f"reporte_caat4_outliers_{datetime.now():%Y%m%d_%H%M}.csv"
            st.download_button("⬇️ Descargar outliers (CSV)", df_to_csv_bytes(anomalies), f1, "text/csv")
            add_report("CAAT4_Outliers", anomalies, f1)

st.divider()


# =========================================
# MÓDULO 5 – CAAT: CRITERIOS DE PROVEEDORES
# =========================================
st.header("✅ Módulo 5: Criterios de selección de proveedores (CAAT 5)")

with st.expander("¿Cómo usar este módulo?", expanded=show_help):
    st.markdown(
        """
**Objetivo:** verificar criterios mínimos (RUC válido, no en blacklist, documento vigente, cuenta validada, aprobado).
1. Sube tu maestro de **Proveedores**.
2. Elige columnas: **Proveedor**, **RUC** y las opcionales (Blacklist, Vigencia, Cuenta, Aprobado).
3. Marca los **criterios** a verificar y revisa incumplimientos.
        """
    )

df5 = read_table_uploader("Maestro de Proveedores (CSV/XLSX)", key="caat5_upl")
if df5 is not None and not df5.empty:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        col_pv = column_picker(df5, "Proveedor", ["proveedor","nombre","vendor"])
    with c2:
        col_ruc = column_picker(df5, "RUC", ["ruc","tax_id","id_fiscal"])
    with c3:
        col_bl  = column_picker(df5, "Blacklist (opcional)", ["blacklist","en_lista_negra","bloqueado"], optional=True)
    with c4:
        col_vig = column_picker(df5, "Fecha vigencia doc (opcional)", ["vigencia","fecha_vigencia","expira"], optional=True)
    with c5:
        col_cta = column_picker(df5, "Cuenta validada (opcional)", ["cuenta_val","cuenta_validada","cuenta_ok"], optional=True)
    with c6:
        col_apr = column_picker(df5, "Aprobado (opcional)", ["aprobado","aprob","approved"], optional=True)

    crit_ruc = st.checkbox("Verificar RUC válido (11-13 dígitos num.)", value=True)
    crit_bl  = st.checkbox("No estar en Blacklist", value=True)
    crit_vig = st.checkbox("Documento vigente (>= hoy)", value=True)
    crit_cta = st.checkbox("Cuenta validada", value=False)
    crit_apr = st.checkbox("Aprobado", value=False)

    ok = True
    if col_pv not in df5.columns or col_ruc not in df5.columns:
        ok = False
        error_help("Selecciona al menos Proveedor y RUC.")
    if ok:
        B = df5.copy()
        B["proveedor"] = B[col_pv].astype(str)
        B["ruc"] = B[col_ruc].astype(str)

        def ruc_valido(r: str) -> bool:
            r = "".join([c for c in r if c.isdigit()])
            return 11 <= len(r) <= 13

        incum_rows = []
        today = pd.Timestamp.today().normalize()

        for _, row in B.iterrows():
            fails = []
            # RUC
            if crit_ruc and not ruc_valido(row["ruc"]):
                fails.append("RUC inválido")
            # Blacklist
            if crit_bl and col_bl in B.columns:
                val = str(row[col_bl]).strip().lower()
                if val in ["1","true","si","sí","y","yes","blacklist","bloqueado"]:
                    fails.append("Proveedor en Blacklist")
            # Vigencia
            if crit_vig and col_vig in B.columns:
                dt = pd.to_datetime(row[col_vig], errors="coerce")
                if pd.isna(dt) or dt < today:
                    fails.append("Documento no vigente")
            # Cuenta
            if crit_cta and col_cta in B.columns:
                val = str(row[col_cta]).strip().lower()
                if val not in ["1","true","si","sí","y","yes","ok","validada"]:
                    fails.append("Cuenta no validada")
            # Aprobado
            if crit_apr and col_apr in B.columns:
                val = str(row[col_apr]).strip().lower()
                if val not in ["1","true","si","sí","y","yes","aprobado"]:
                    fails.append("No aprobado")

            if fails:
                incum_rows.append({
                    "proveedor": row["proveedor"],
                    "ruc": row["ruc"],
                    "criterios_incumplidos": "; ".join(fails),
                    "detalle": "Incumplimientos detectados"
                })

        INC = pd.DataFrame(incum_rows)
        n_inc = len(INC)
        st.metric("Proveedores con incumplimientos", f"{n_inc:,}")

        st.session_state.metrics["CAAT5"] = {
            "score": score_from_count(n_inc, scale=max(len(B),1)),
            "desc": "(más alto => más riesgo)",
            "extra": f"Incumplimientos={n_inc}"
        }

        st.subheader("Incumplimientos detectados")
        st.dataframe(INC, use_container_width=True, height=260)
        f1 = f"reporte_caat5_incumplimientos_{datetime.now():%Y%m%d_%H%M}.csv"
        st.download_button("⬇️ Descargar (CSV)", df_to_csv_bytes(INC), f1, "text/csv")
        add_report("CAAT5_Incumplimientos", INC, f1)

st.divider()


# ===========================
# REPORTE FINAL CONSOLIDADO
# ===========================
st.header("🧾 Reporte final consolidado")
info_help(
    "Este reporte reúne las descargas de todos los módulos ejecutados en esta sesión y agrega un **resumen ejecutivo**."
)

if st.session_state.reports:
    zip_bytes = make_zip_reports()
    file_zip = f"reporte_consolidado_caats_{datetime.now():%Y%m%d_%H%M}.zip"
    st.download_button("⬇️ Descargar ZIP consolidado (CSV + Resumen)", data=zip_bytes, file_name=file_zip, mime="application/zip")
else:
    st.caption("Aún no hay reportes generados. Ejecuta algún módulo y descarga sus hallazgos para incluirlos aquí.")


# ===========================
# NOTAS FINALES / AYUDAS
# ===========================
with st.expander("Guía rápida de archivos aceptados", expanded=False):
    st.markdown(
        """
- **CSV**: delimitado por coma (encoding UTF-8 recomendado).
- **Excel .xlsx**: requiere el paquete `openpyxl` (incluido en requirements).
- Si tu Excel tiene varias hojas, **elige la hoja** en el selector que aparece al subirlo.
- Si el módulo no muestra resultados, revisa:
  - Columnas requeridas seleccionadas (Usuario, Fecha/Hora, etc.).
  - Que la columna Fecha/Hora tenga **fechas válidas**.
  - Que las columnas numéricas (ej. Monto) sean realmente numéricas.
        """
    )

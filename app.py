# ------------------------------------------------------------
# Aprendizaje Colaborativo y Práctico – 2do Parcial
# Suite CAAT (1–5) + Modo libre — Versión optimizada
# ------------------------------------------------------------
# Cambios clave:
# ✔ Cacheo de parsing de archivos (rápido en reruns)
# ✔ Límite de tamaño de subida y mensajes claros
# ✔ Selects más robustos (evita fallos sin columnas)
# ✔ CAAT1: jornadas que cruzan medianoche + feriados opcionales
# ✔ CAAT3: tolerancias por tipo de transacción (opcional)
# ✔ CAAT5: near-duplicados con parejas evidenciadas
# ✔ Informe en pantalla (resumen ejecutivo) en CAAT1/3/4/5
# ✔ Config opcional vía config.yaml (si existe)
# ✔ Requisitos mínimos sin matplotlib
# ------------------------------------------------------------

import io
import math
import hashlib
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
import streamlit as st

# Carga opcional de YAML (no obligatorio)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# ------------------------------
# Configuración
# ------------------------------

@st.cache_resource(show_spinner=False)
def load_config(path: str = "config.yaml") -> dict:
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

CFG = load_config()

APP_TITLE = CFG.get("app", {}).get("title", "Aprendizaje Colaborativo y Práctico – 2do Parcial")
MAX_UPLOAD_MB = float(CFG.get("app", {}).get("max_upload_mb", 25))

# ------------------------------
# Estilo y utilitarios generales
# ------------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")

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


def metric_badge(label: str, value, level: str = "ok", help_text: str = ""):
    colors = {"ok": "#dcfce7", "warn": "#fef9c3", "bad": "#fee2e2"}
    bg = colors.get(level, "#f9fafb")
    st.markdown(
        f'<div class="kpi" style="background:{bg}"><b>{label}</b><br>'
        f'<span style="font-size:26px">{value}</span></div>',
        unsafe_allow_html=True,
    )
    if help_text:
        st.caption(help_text)

# ------------------------------
# Sesión y lectura de archivos (optimizada)
# ------------------------------

def _save_in_session(key: str, file) -> None:
    if file is None:
        return
    content = file.getvalue() if hasattr(file, "getvalue") else file.read()
    st.session_state[key] = {"name": file.name, "bytes": content}


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@st.cache_data(show_spinner=False)
def _parse_file_cached(name: str, content_hash: str, ext: str, sep_guess: bool = True):
    raw = st.session_state.get("_file_bytes_cache", {}).get(content_hash)
    if raw is None:
        return None
    bio = io.BytesIO(raw)
    if ext == ".csv":
        try:
            return pd.read_csv(bio, encoding_errors="ignore")
        except Exception:
            if sep_guess:
                bio.seek(0)
                return pd.read_csv(bio, sep=";", encoding_errors="ignore")
            raise
    elif ext == ".xlsx":
        return pd.read_excel(bio, engine="openpyxl")
    return None


def _read_from_session(key: str):
    if key not in st.session_state:
        return None
    info = st.session_state[key]
    name = info["name"]
    raw = info["bytes"]

    # Límite de tamaño
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"El archivo pesa {size_mb:.1f} MB (límite {MAX_UPLOAD_MB:.0f} MB). Sube un archivo más liviano.")
        return None

    name_l = name.lower()
    if name_l.endswith(".csv"):
        ext = ".csv"
    elif name_l.endswith(".xlsx"):
        ext = ".xlsx"
    else:
        st.warning("Formato no soportado. Usa CSV o XLSX.")
        return None

    h = _hash_bytes(raw)
    st.session_state.setdefault("_file_bytes_cache", {})[h] = raw

    if ext == ".xlsx":
        # Soporte multishoja con selección por defecto a la primera con datos
        try:
            with pd.ExcelFile(io.BytesIO(raw), engine="openpyxl") as xls:
                sheets = xls.sheet_names
                # preview para detectar primera no vacía
                previews = {}
                for s in sheets:
                    try:
                        previews[s] = pd.read_excel(xls, sheet_name=s, nrows=5, engine="openpyxl")
                    except Exception:
                        previews[s] = pd.DataFrame()
                non_empty = [s for s, d in previews.items() if d.shape[1] > 0]
                default_sheet = non_empty[0] if non_empty else sheets[0]
                sheet = st.selectbox(
                    "Hoja de Excel",
                    sheets,
                    index=sheets.index(default_sheet),
                    key=f"sheet_{key}",
                    help="Selecciona la hoja que contiene tus datos.",
                )
                df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
                return df
        except Exception:
            # fallback cache parser
            return _parse_file_cached(name, h, ext)

    # CSV o fallback
    return _parse_file_cached(name, h, ext)


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
        else:
            st.success("Archivo listo ✅ (se conserva en memoria para no perderlo tras el rerun).")
            with st.expander("Vista rápida (primeras filas)", expanded=False):
                st.dataframe(df.head(50), use_container_width=True)
            return df

# ------------------------------
# Utilidades de columnas y fechas
# ------------------------------

def guess_column(cols, candidates):
    clower = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in clower:
            return cols[clower.index(cand.lower())]
    return None


def choose_column(df, label, candidates, key_suffix, help_text=""):
    cols = list(df.columns)
    if not cols:
        st.warning("El archivo no tiene columnas legibles. Revisa el formato.")
        return None
    guess = guess_column(cols, candidates)
    idx = cols.index(guess) if (guess in cols) else 0
    return st.selectbox(
        label,
        cols,
        index=min(idx, len(cols) - 1),
        key=f"sel_{key_suffix}",
        help=help_text,
    )


def ensure_datetime(series: pd.Series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def robust_zscore(series: pd.Series):
    x = pd.to_numeric(series, errors="coerce").astype(float).values
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - median) / mad


def in_schedule_vectorized(hour_f: pd.Series, start_t: time, end_t: time) -> pd.Series:
    s = start_t.hour + start_t.minute / 60
    e = end_t.hour + end_t.minute / 60
    if s <= e:
        return (hour_f >= s) & (hour_f <= e)
    else:
        # Jornada que cruza medianoche: ej. 22:00–06:00
        return (hour_f >= s) | (hour_f <= e)

# ------------------------------
# CAAT 1 – Registros fuera de horario
# ------------------------------

def module_caat1():
    st.subheader("CAAT 1 – Registros fuera de horario")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown(
            """
1. **Sube tu log** (CSV/XLSX).
2. **Usuario** y **Fecha/Hora** (sugerimos automáticamente).
3. Define **inicio/fin de jornada** y si deseas **solo días hábiles (L–V)**.
4. (Opcional) Sube **feriados** para excluirlos.
5. Revisa KPIs, informe y descarga hallazgos.
"""
        )

    df = file_uploader_block("Log de actividades", key="caat1")
    if df is None:
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        user_col = choose_column(
            df,
            "Columna Usuario",
            ["usuario", "user", "empleado", "usuario_id", "id_usuario"],
            "caat1_user",
        )
    with c2:
        dt_col = choose_column(
            df,
            "Columna Fecha/Hora",
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

    if not user_col or not dt_col:
        st.warning("Selecciona las columnas requeridas para continuar.")
        return

    # Parámetros de horario
    c4, c5, c6, c7 = st.columns([1, 1, 1, 2])
    default_start = CFG.get("caat1", {}).get("start_time", "08:00")
    default_end = CFG.get("caat1", {}).get("end_time", "18:00")
    ds_h, ds_m = map(int, default_start.split(":")) if ":" in default_start else (8, 0)
    de_h, de_m = map(int, default_end.split(":")) if ":" in default_end else (18, 0)

    with c4:
        start_h = st.time_input("Inicio de jornada", time(ds_h, ds_m), key="caat1_start")
    with c5:
        end_h = st.time_input("Fin de jornada", time(de_h, de_m), key="caat1_end")
    with c6:
        only_weekdays = st.checkbox(
            "Solo días hábiles (L–V)",
            value=bool(CFG.get("caat1", {}).get("weekdays_only", True)),
            key="caat1_week",
        )
    with c7:
        topn = st.slider("Top N reincidentes", 5, 100, 20, key="caat1_topn")

    st.markdown("**Feriados (opcional)**")
    hol = file_uploader_block(
        "Calendario de feriados",
        key="caat1_holidays",
        help_text="Debe incluir una columna llamada 'fecha'",
    )

    work = df[[user_col, dt_col] + ([action_col] if action_col else [])].copy()
    work.rename(columns={user_col: "user", dt_col: "dt"}, inplace=True)
    work["dt"] = ensure_datetime(work["dt"])
    work = work.dropna(subset=["dt"])
    work["weekday"] = work["dt"].dt.weekday  # 0=Lunes
    work["hour"] = work["dt"].dt.hour + work["dt"].dt.minute / 60

    in_schedule = in_schedule_vectorized(work["hour"], start_h, end_h)

    # Días hábiles y feriados
    if only_weekdays:
        in_schedule &= work["weekday"].between(0, 4)
    if hol is not None and "fecha" in hol.columns:
        holidays = set(pd.to_datetime(hol["fecha"], errors="coerce").dt.date.dropna().tolist())
        work["is_holiday"] = work["dt"].dt.date.isin(holidays)
        in_schedule &= ~work["is_holiday"]

    work["fuera_horario"] = ~in_schedule

    total_events = len(work)
    out_of_hours = int(work["fuera_horario"].sum())
    pct = (out_of_hours / total_events * 100) if total_events else 0

    # KPI con semáforo
    kcol1, kcol2, kcol3 = st.columns(3)
    kcol1.metric("Eventos totales", f"{total_events:,}")
    level = "bad" if pct > 5 else ("warn" if pct > 1 else "ok")
    metric_badge("Fuera de horario", f"{out_of_hours:,}", level=level)
    kcol3.metric("% fuera de horario", f"{pct:.2f}%")

    findings = work[work["fuera_horario"]].copy()
    if not findings.empty:
        # reincidentes
        top = (
            findings.groupby("user", as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(topn)
        )
        with st.expander("Top reincidentes", expanded=True):
            st.dataframe(top, use_container_width=True)

        with st.expander("Hallazgos (detallado)", expanded=False):
            st.dataframe(findings, use_container_width=True)

        csv = findings.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar hallazgos (CSV)", csv, file_name="CAAT1_fuera_horario.csv", mime="text/csv"
        )
    else:
        st.info("No se detectaron registros fuera de horario con los parámetros actuales.")

    # Informe rápido
    st.markdown("### Informe rápido")
    reinc = ", ".join(top["user"].astype(str).head(5)) if not findings.empty else "N/A"
    st.markdown(
        f"- Eventos totales analizados: **{total_events:,}**\n"
        f"- Registros fuera de horario: **{out_of_hours:,}** ({pct:.2f}%)\n"
        f"- Reincidentes TOP {topn}: {reinc if reinc else 'N/A'}\n"
        f"- Parámetros: jornada **{start_h.strftime('%H:%M')}–{end_h.strftime('%H:%M')}**, "
        f"{'solo L–V' if only_weekdays else 'todos los días'}"
    )

# ------------------------------
# CAAT 2 – Auditoría de privilegios (roles críticos y SoD)
# ------------------------------

def module_caat2():
    st.subheader("CAAT 2 – Auditoría de privilegios (roles críticos y SoD)")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown(
            """
1. Sube el **maestro de Usuarios/Roles**.
2. Elige **Usuario** y **Rol** (autodetección).
3. Define roles **críticos** (columna booleana *o* lista manual).
4. Escribe reglas **SoD** (una por línea) en el formato `ROL_A -> ROL_B`.
"""
        )

    df = file_uploader_block("Usuarios/Roles", key="caat2")
    if df is None:
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        user_col = choose_column(df, "Columna Usuario", ["usuario", "user", "empleado"], "caat2_user")
    with c2:
        role_col = choose_column(df, "Columna Rol", ["rol", "role", "módulo", "modulo"], "caat2_role")
    with c3:
        crit_col = st.selectbox("Columna es_crítico (opcional)", ["(ninguna)"] + list(df.columns), key="caat2_crit")
        if crit_col == "(ninguna)":
            crit_col = None

    if not user_col or not role_col:
        st.warning("Selecciona las columnas requeridas para continuar.")
        return

    base = df[[user_col, role_col] + ([crit_col] if crit_col else [])].copy()
    base.rename(columns={user_col: "user", role_col: "role"}, inplace=True)
    if crit_col:
        base["is_critical"] = base[crit_col].astype(str).str.lower().isin(
            ["true", "1", "si", "sí", "x", "critical", "critico", "crítico"]
        )
    else:
        base["is_critical"] = False

    # Roles críticos manuales desde config o input
    cfg_crit = set(map(str.upper, CFG.get("caat2", {}).get("critical_roles", [])))
    st.markdown("**Roles críticos (opcional)**")
    manual_crit = st.text_input(
        "Lista separada por comas (ej. ADMIN, TESORERIA, AUTORIZADOR)",
        value=", ".join(sorted(cfg_crit)) if cfg_crit else "",
        key="caat2_manualcrit",
    )
    extra_crit = {x.strip().upper() for x in manual_crit.split(",") if x.strip()}
    if extra_crit:
        base["is_critical"] = base["is_critical"] | base["role"].astype(str).str.upper().isin(extra_crit)

    st.markdown("**Reglas SoD (una por línea, formato `ROL_A -> ROL_B`)**")
    default_rules = "\n".join(CFG.get("caat2", {}).get("sod_rules", ["AUTORIZAR_PAGO -> REGISTRAR_PAGO"]))
    sod_text = st.text_area("Reglas SoD", value=default_rules, key="caat2_sod")
    rules = []
    for line in sod_text.splitlines():
        if "->" in line:
            a, b = [x.strip().upper() for x in line.split("->", 1)]
            if a and b:
                rules.append((a, b))

    # Conflictos SoD por usuario
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
        csv = conflicts_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar conflictos SoD", csv, "CAAT2_conflictos_SoD.csv", mime="text/csv")
    else:
        st.info("No se detectaron conflictos SoD con las reglas actuales.")

    with st.expander("Roles críticos (detalle)", expanded=False):
        st.dataframe(base[base["is_critical"]].sort_values(["user", "role"]), use_container_width=True)
        if not base[base["is_critical"]].empty:
            csv2 = base[base["is_critical"]].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar roles críticos", csv2, "CAAT2_roles_criticos.csv", mime="text/csv")

# ------------------------------
# CAAT 3 – Conciliación de logs vs transacciones (+ banca opcional)
# ------------------------------

def module_caat3():
    st.subheader("CAAT 3 – Conciliación de logs vs transacciones")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown(
            """
1. Sube **Logs** y **Transacciones**.
2. En ambos: selecciona **ID** y **Fecha/Hora**.
3. Define **tolerancia** (minutos) para marcar desfase. (Opcional) por tipo.
4. Descarga los **no conciliados** y los **fuera de tolerancia**.

**Opcional:** Conciliación bancaria simple (extracto bancario vs libro).
"""
        )

    c = st.columns(2)
    with c[0]:
        logs = file_uploader_block("Logs (CSV/XLSX)", key="caat3_logs")
    with c[1]:
        txs = file_uploader_block("Transacciones (CSV/XLSX)", key="caat3_txs")

    if logs is None or txs is None:
        return

    # Selecciones
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

    if not id_logs or not dt_logs or not id_txs or not dt_txs:
        st.warning("Selecciona las columnas requeridas para continuar.")
        return

    tol_min_default = int(CFG.get("caat3", {}).get("default_tol_min", 30))
    tol_min = st.slider("Tolerancia (minutos) por defecto", 0, 180, tol_min_default, key="caat3_tol")

    # Tolerancias por tipo (opcional, desde txs)
    type_col = st.selectbox("Columna tipo (opcional, en Transacciones)", ["(ninguna)"] + list(txs.columns), key="caat3_type")
    custom_tols_txt = st.text_area(
        "Tolerancias por tipo (ej: TRANSFER=15; CASHOUT=5)", key="caat3_tols_txt"
    )
    tol_map = {}
    for part in custom_tols_txt.split(";"):
        if "=" in part:
            k, v = part.split("=")
            try:
                tol_map[k.strip().upper()] = float(v.strip())
            except Exception:
                pass

    logs2 = logs[[id_logs, dt_logs]].copy()
    logs2.rename(columns={id_logs: "id", dt_logs: "dt"}, inplace=True)
    logs2["dt"] = ensure_datetime(logs2["dt"])

    txs_cols = [id_txs, dt_txs]
    if type_col and type_col != "(ninguna)":
        txs_cols.append(type_col)
    txs2 = txs[txs_cols].copy()
    txs2.rename(columns={id_txs: "id", dt_txs: "dt"}, inplace=True)
    txs2["dt"] = ensure_datetime(txs2["dt"])
    if type_col and type_col != "(ninguna)":
        txs2["_type"] = txs2[type_col].astype(str).str.upper()
    else:
        txs2["_type"] = "__DEFAULT__"

    # Conciliación por ID
    merged = pd.merge(logs2, txs2, on="id", how="outer", suffixes=("_log", "_tx"))
    missing_log = merged[merged["dt_log"].isna()]
    missing_tx = merged[merged["dt_tx"].isna()]

    # desfase por tolerancia (por tipo si aplica)
    both = merged.dropna(subset=["dt_log", "dt_tx"]).copy()
    both["delta_min"] = (both["dt_tx"] - both["dt_log"]).dt.total_seconds().div(60).abs()

    def row_tol(row):
        t = str(row.get("_type", "__DEFAULT__")).upper()
        return tol_map.get(t, tol_min)

    both["tol_row"] = both.apply(row_tol, axis=1)
    out_tol = both[both["delta_min"] > both["tol_row"]]

    k1, k2, k3 = st.columns(3)
    k1.metric("IDs solo en Transacciones", f"{len(missing_log):,}")
    k2.metric("IDs solo en Logs", f"{len(missing_tx):,}")
    k3.metric("Fuera de tolerancia", f"{len(out_tol):,}")

    with st.expander("IDs solo en Transacciones", expanded=False):
        st.dataframe(missing_log, use_container_width=True)
        if not missing_log.empty:
            st.download_button("⬇️ Descargar CSV", missing_log.to_csv(index=False).encode("utf-8"), "CAAT3_solo_en_tx.csv")

    with st.expander("IDs solo en Logs", expanded=False):
        st.dataframe(missing_tx, use_container_width=True)
        if not missing_tx.empty:
            st.download_button("⬇️ Descargar CSV", missing_tx.to_csv(index=False).encode("utf-8"), "CAAT3_solo_en_logs.csv")

    with st.expander("Fuera de tolerancia", expanded=True):
        st.dataframe(out_tol, use_container_width=True)
        if not out_tol.empty:
            st.download_button("⬇️ Descargar CSV", out_tol.to_csv(index=False).encode("utf-8"), "CAAT3_fuera_tolerancia.csv")

    # Informe rápido
    st.markdown("### Informe rápido")
    st.markdown(
        f"- IDs solo en Transacciones: **{len(missing_log):,}**\n"
        f"- IDs solo en Logs: **{len(missing_tx):,}**\n"
        f"- Registros fuera de tolerancia: **{len(out_tol):,}**\n"
        f"- Tolerancia por defecto: **{tol_min} min**"
        + (f" | Reglas por tipo: **{len(tol_map)}**" if tol_map else "")
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

        if not amt_b or not date_b or not amt_c or not date_c:
            st.warning("Selecciona las columnas requeridas para continuar.")
            return

        tol_days = st.slider("Tolerancia de fecha (días)", 0, 15, 3, key="caat3_bank_tol")

        b2 = bank[[amt_b, date_b]].copy().rename(columns={amt_b: "amount", date_b: "date"})
        c2_ = book[[amt_c, date_c]].copy().rename(columns={amt_c: "amount", date_c: "date"})
        b2["date"] = pd.to_datetime(b2["date"], errors="coerce").dt.date
        c2_["date"] = pd.to_datetime(c2_["date"], errors="coerce").dt.date
        b2["amount_r"] = pd.to_numeric(b2["amount"], errors="coerce").round(2)
        c2_["amount_r"] = pd.to_numeric(c2_["amount"], errors="coerce").round(2)

        matches = []
        used_idx = set()
        for i, br in b2.dropna(subset=["amount_r", "date"]).iterrows():
            cands = c2_[(c2_["amount_r"] == br["amount_r"]) & ~c2_.index.isin(used_idx)]
            if cands.empty:
                continue
            deltas = cands["date"].apply(lambda d: abs((d - br["date"]).days))
            j = deltas.idxmin()
            if deltas.loc[j] <= tol_days:
                matches.append({"bank_idx": i, "book_idx": j})
                used_idx.add(j)

        matched_b = {m["bank_idx"] for m in matches}
        matched_c = {m["book_idx"] for m in matches}

        bank_unmatched = b2[~b2.index.isin(matched_b)]
        book_unmatched = c2_[~c2_.index.isin(matched_c)]

        u1, u2 = st.columns(2)
        with u1:
            st.markdown("**No conciliados (Banco)**")
            st.dataframe(bank_unmatched, use_container_width=True)
            if not bank_unmatched.empty:
                st.download_button(
                    "⬇️ Descargar banco no conciliado",
                    bank_unmatched.to_csv(index=False).encode("utf-8"),
                    "CAAT3_banco_no_conciliado.csv",
                )
        with u2:
            st.markdown("**No conciliados (Libro)**")
            st.dataframe(book_unmatched, use_container_width=True)
            if not book_unmatched.empty:
                st.download_button(
                    "⬇️ Descargar libro no conciliado",
                    book_unmatched.to_csv(index=False).encode("utf-8"),
                    "CAAT3_libro_no_conciliado.csv",
                )

# ------------------------------
# CAAT 4 – Variación inusual de pagos (outliers)
# ------------------------------

def module_caat4():
    st.subheader("CAAT 4 – Variación inusual de pagos (outliers)")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown(
            """
1. Sube tu **histórico de pagos**.
2. Selecciona **Proveedor**, **Fecha** y **Monto**.
3. Ajusta el **umbral de outliers (|z| robusto)** y descarga hallazgos.
"""
        )

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

    if not vendor_col or not date_col or not amount_col:
        st.warning("Selecciona las columnas requeridas para continuar.")
        return

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
    workz = pd.concat(out_list).reset_index(drop=True) if out_list else pd.DataFrame(columns=["vendor", "date", "amount", "z_robusto"])

    default_thr = float(CFG.get("caat4", {}).get("z_threshold", 3.5))
    thr = st.slider("Umbral |z| robusto", 2.0, 6.0, float(default_thr), 0.5, key="caat4_thr")
    outliers = workz[workz["z_robusto"].abs() >= thr]

    k1, k2 = st.columns(2)
    k1.metric("Registros", f"{len(work):,}")
    k2.metric("Outliers", f"{len(outliers):,}")

    with st.expander("Outliers detectados", expanded=True):
        st.dataframe(outliers.sort_values(["vendor", "date"]), use_container_width=True)
        if not outliers.empty:
            st.download_button("⬇️ Descargar outliers", outliers.to_csv(index=False).encode("utf-8"), "CAAT4_outliers.csv")

    # Informe rápido
    st.markdown("### Informe rápido")
    vendors_top = (
        outliers.groupby("vendor").size().sort_values(ascending=False).head(5).index.tolist()
        if not outliers.empty
        else []
    )
    st.markdown(
        f"- Registros analizados: **{len(work):,}**\n"
        f"- Outliers identificados: **{len(outliers):,}** (umbral |z| ≥ {thr})\n"
        f"- Proveedores con más outliers: {', '.join(map(str, vendors_top)) if vendors_top else 'N/A'}"
    )

# ------------------------------
# CAAT 5 – Duplicados / near-duplicados
# ------------------------------

def module_caat5():
    st.subheader("CAAT 5 – Duplicados / near-duplicados")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown(
            """
1. Sube tu **tabla** (pagos, registros, etc.).
2. Selecciona **columnas clave** y opcionalmente **Fecha** y **Monto**.
3. Descarga **duplicados** y, si quieres, **near-duplicados** (mismo monto y fecha cercana) con **parejas** evidenciadas.
"""
        )

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
        dups = df[df.duplicated(subset=key_cols, keep=False)].sort_values(key_cols).copy()

    k1, k2 = st.columns(2)
    k1.metric("Registros", f"{len(df):,}")
    k2.metric("Duplicados exactos", f"{len(dups):,}")

    with st.expander("Duplicados", expanded=not dups.empty):
        if dups.empty:
            st.info("No se detectaron duplicados exactos con las columnas seleccionadas.")
        else:
            st.dataframe(dups, use_container_width=True)
            st.download_button("⬇️ Descargar duplicados", dups.to_csv(index=False).encode("utf-8"), "CAAT5_duplicados.csv")

    st.markdown("**Near-duplicados (opcional)**")
    if date_col and amt_col:
        tol_default = int(CFG.get("caat5", {}).get("neardup_tol_days", 2))
        tol_days = st.slider("Tolerancia de fecha (días)", 0, 10, tol_default, key="caat5_tol")
        w = df[[date_col, amt_col] + key_cols].copy()
        w["date"] = pd.to_datetime(w[date_col], errors="coerce").dt.date
        w["amount"] = pd.to_numeric(w[amt_col], errors="coerce").round(2)
        w = w.dropna(subset=["date", "amount"])

        pairs = []
        for amount, grp in w.groupby("amount"):
            g = grp.sort_values("date").reset_index(drop=True)
            for i in range(len(g) - 1):
                d1, d2 = g.loc[i, "date"], g.loc[i + 1, "date"]
                if abs((d2 - d1).days) <= tol_days:
                    pairs.append(g.loc[[i, i + 1]])
        near_pairs = pd.concat(pairs) if pairs else pd.DataFrame()

        with st.expander("Near-duplicados (parejas)", expanded=not near_pairs.empty):
            if near_pairs.empty:
                st.info("No se detectaron near-duplicados con los parámetros actuales.")
            else:
                st.dataframe(near_pairs, use_container_width=True)
                st.download_button(
                    "⬇️ Descargar near-duplicados",
                    near_pairs.to_csv(index=False).encode("utf-8"),
                    "CAAT5_near_duplicados.csv",
                )

    # Informe rápido
    st.markdown("### Informe rápido")
    st.markdown(
        f"- Duplicados exactos: **{len(dups):,}**"
        + (f" | Near-duplicados encontrados: **{int(len(near_pairs)/2):,}** parejas" if date_col and amt_col else "")
    )

# ------------------------------
# Modo libre – EDA guiado
# ------------------------------

def module_free_mode():
    st.subheader("Modo libre – Sube cualquier archivo y exploramos")
    st.markdown(
        """
Este modo acepta **cualquier CSV/XLSX**. Intentamos detectar tipos de columna y brindamos:
- Conteo de filas/columnas, valores faltantes, tipos
- Filtros dinámicos por columna
- KPIs básicos (suma, promedio si aplica) y **descarga** del resultado filtrado
"""
    )

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
            "nulos": df.isna().sum().values,
        })
        st.dataframe(info, use_container_width=True)

    st.markdown("### Filtros rápidos")
    filtered = df.copy()
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                rng = st.slider(
                    f"{c} (rango)", float(df[c].min()), float(df[c].max()),
                    (float(df[c].min()), float(df[c].max())), key=f"free_rng_{c}"
                )
                filtered = filtered[filtered[c].between(rng[0], rng[1])]
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                d1 = pd.to_datetime(df[c].min())
                d2 = pd.to_datetime(df[c].max())
                start, end = st.date_input(f"{c} (fecha)", value=(d1.date(), d2.date()), key=f"free_date_{c}")
                filtered = filtered[(pd.to_datetime(filtered[c]).dt.date >= start) & (pd.to_datetime(filtered[c]).dt.date <= end)]
            else:
                unique_vals = df[c].dropna().astype(str).unique().tolist()
                if 0 < len(unique_vals) <= 100:
                    vals = st.multiselect(f"{c} (valores)", unique_vals, default=unique_vals, key=f"free_ms_{c}")
                    filtered = filtered[filtered[c].astype(str).isin(vals)]
        except Exception:
            pass

    st.markdown("### Resultado filtrado")
    st.dataframe(filtered.head(1000), use_container_width=True)
    st.download_button("⬇️ Descargar filtrado (CSV)", filtered.to_csv(index=False).encode("utf-8"), "modo_libre_filtrado.csv")

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
    st.title(APP_TITLE)
    st.caption("Suite de herramientas CAAT para auditoría asistida.")

    tabs = st.tabs(["CAAT 1–5", "Modo libre"])

    with tabs[0]:
        with st.expander("Ayuda general (antes de empezar)", expanded=True):
            st.markdown(
                """
<span class=\"badge\">¿Qué hace esta app?</span> Corre **5 CAAT** comunes de auditoría y un **Modo libre** para explorar cualquier archivo.

<span class=\"badge-red\">Errores comunes</span>  
- **No se pudo leer el Excel**: verifica que el archivo **no** esté protegido y sea un **.xlsx válido**.  
- **Fechas vacías/NaT**: verifica el **formato** o elige la **columna correcta**.  
- **Selectbox duplicado**: aquí cada select tiene **claves únicas**, no deberías ver este error.  

<span class=\"badge-green\">Descargas</span>  
Todos los módulos generan **CSV** de hallazgos.
                """,
                unsafe_allow_html=True,
            )

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
        st.markdown(
            """
- Prioriza por KPIs: **concentración** de outliers por proveedor (CAAT 4),  
  **reincidencia** fuera de horario (CAAT 1) y **conflictos** SoD (CAAT 2).
- En conciliación (CAAT 3), enfócate en **gaps mayores** de tolerancia y **no conciliados**.
- Descarga los CSV y **documenta tu evidencia** en los papeles de trabajo.
            """
        )

    with tabs[1]:
        module_free_mode()


if __name__ == "__main__":
    main()

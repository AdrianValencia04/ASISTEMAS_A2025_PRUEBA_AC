# -*- coding: utf-8 -*-
import io
import sys
import uuid
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
import streamlit as st

# Fuzzy: rapidfuzz (preferido). Si no existe, degradamos a similitud básica.
try:
    from rapidfuzz import fuzz
    def fuzzy_ratio(a, b):
        try:
            return fuzz.token_set_ratio(str(a), str(b)) / 100.0
        except Exception:
            return 0.0
except Exception:
    def fuzzy_ratio(a, b):
        # Degradado simple si no está rapidfuzz
        a, b = str(a).lower(), str(b).lower()
        if not a or not b:
            return 0.0
        inter = len(set(a.split()) & set(b.split()))
        union = len(set(a.split()) | set(b.split()))
        return inter / union if union else 0.0

# ------------------------------
# Utilidades UI y datos
# ------------------------------

def _k(key_base: str) -> str:
    """Genera keys únicos para widgets (evita StreamlitDuplicateElementId)."""
    return f"{key_base}_{uuid.uuid4().hex[:8]}"

def info_box(msg: str, icon="ℹ️"):
    st.markdown(f"> {icon} {msg}")

def warn_box(msg: str):
    st.warning(msg)

def success_box(msg: str):
    st.success(msg)

def error_box(msg: str):
    st.error(msg)

def read_table(file) -> pd.DataFrame:
    """
    Lee CSV/XLSX de manera robusta.
    """
    name = getattr(file, "name", "archivo")
    try:
        if name.lower().endswith(".csv"):
            return pd.read_csv(file, encoding="utf-8", engine="python")
        elif name.lower().endswith((".xlsx", ".xlsm", ".xls")):
            try:
                return pd.read_excel(file, engine="openpyxl")
            except Exception as e:
                raise RuntimeError(
                    "No se pudo leer el Excel (.xlsx). Verifica que no esté protegido y que "
                    "tengas el paquete 'openpyxl' instalado."
                ) from e
        else:
            raise RuntimeError("Formato no soportado. Sube CSV o XLSX.")
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1", engine="python")
    except Exception as e:
        raise

def read_table_with_sheet(xls_file, key_prefix="xls"):
    """
    Si un Excel tiene múltiples hojas, pregunta; si no, devuelve la única.
    Para CSV, lo devuelve directo.
    """
    name = getattr(xls_file, "name", "archivo")
    if not name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        # csv u otro
        return read_table(xls_file)

    try:
        xls = pd.ExcelFile(xls_file, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(
            "No se pudo leer el Excel (.xlsx). ¿Está protegido? ¿Se instaló 'openpyxl'?"
        ) from e

    sheets = xls.sheet_names
    if len(sheets) == 1:
        sheet = sheets[0]
    else:
        sheet = st.selectbox("Hoja de Excel", sheets, key=_k(f"{key_prefix}_sheet"), help="Selecciona la hoja con los datos")

    df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
    return df

def suggest_column(df: pd.DataFrame, hope=("usuario","user","empleado","id","name")):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for h in hope:
        if h in lower_map:
            return lower_map[h]
    # fallback: primera
    return cols[0] if cols else None

def parse_datetime_col(series: pd.Series):
    # intenta parseo robusto
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    # si es numérico o string
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def kpi_bar(label: str, value, help_txt=None):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(label, value)
    with col2:
        try:
            v = float(str(value).replace("%",""))
            v = v/100 if "%" in str(value) else v
            v = max(0.0, min(1.0, v))
            st.progress(v)
        except Exception:
            st.write("")

# ------------------------------
# CAAT 1 – Fuera de horario
# ------------------------------

def caat1():
    st.subheader("CAAT 1 – Registros fuera de horario")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. **Sube tu log** (CSV/XLSX).
2. Selecciona **Usuario** y **Fecha/Hora** (sugerimos automáticamente).
3. Define inicio/fin de jornada y si deseas **solo días hábiles (L–V)**.
4. Revisa KPIs y descarga los **hallazgos**.
        """)
    file = st.file_uploader("Log de actividades", type=["csv","xlsx"], key=_k("caat1_file"))
    if not file:
        info_box("Sube un archivo para comenzar.")
        return

    try:
        raw = read_table_with_sheet(file, key_prefix="caat1")
    except Exception as e:
        error_box(str(e))
        return

    if raw.empty:
        warn_box("El archivo está vacío.")
        return

    # Sugerencias
    user_col = suggest_column(raw, ("usuario","user","empleado","login"))
    dt_col   = suggest_column(raw, ("timestamp","fecha_registro","fecha","datetime","hora","fechahora"))

    c1, c2, c3 = st.columns(3)
    with c1:
        user_col = st.selectbox("Columna Usuario", raw.columns, index=list(raw.columns).index(user_col) if user_col in raw.columns else 0, key=_k("caat1_user"))
    with c2:
        dt_col = st.selectbox("Columna Fecha/Hora", raw.columns, index=list(raw.columns).index(dt_col) if dt_col in raw.columns else 0, key=_k("caat1_dt"))
    with c3:
        act_col = st.selectbox("Columna Acción (opcional)", ["(ninguna)"] + list(raw.columns), key=_k("caat1_act"))

    # Parámetros
    c4, c5, c6 = st.columns(3)
    with c4:
        start_day = st.time_input("Inicio de jornada", value=time(8,0), key=_k("caat1_start"))
    with c5:
        end_day   = st.time_input("Fin de jornada", value=time(18,0), key=_k("caat1_end"))
    with c6:
        only_weekdays = st.checkbox("Solo días hábiles (L–V)", value=True, key=_k("caat1_weekdays"))

    df = raw.copy()
    # Parseo fecha/hora
    df["__dt__"] = parse_datetime_col(df[dt_col])
    # Filtrar NaT
    nat_count = df["__dt__"].isna().sum()
    if nat_count:
        warn_box(f"Se encontraron {nat_count} registros con fecha/hora inválida. Serán omitidos.")
    df = df.dropna(subset=["__dt__"]).reset_index(drop=True)

    # Reglas de horario
    df["__hour__"] = df["__dt__"].dt.time
    df["__weekday__"] = df["__dt__"].dt.weekday  # 0=Lunes
    in_range = df["__hour__"].apply(lambda h: (h >= start_day) and (h <= end_day))
    if only_weekdays:
        in_range = in_range & df["__weekday__"].between(0,4)
    df["fuera_horario"] = ~in_range

    total_ev = len(df)
    out_count = int(df["fuera_horario"].sum())
    pct = (out_count/total_ev)*100 if total_ev else 0

    st.markdown("### KPIs")
    cols = st.columns(3)
    with cols[0]: kpi_bar("Eventos totales", total_ev)
    with cols[1]: kpi_bar("Fuera de horario", out_count)
    with cols[2]: kpi_bar("% fuera de horario", f"{pct:.2f}%")

    st.markdown("### Hallazgos")
    show_cols = [user_col, dt_col]
    if act_col != "(ninguna)":
        show_cols.append(act_col)
    show_cols += ["fuera_horario"]
    st.dataframe(df[show_cols].sort_values(by=dt_col).reset_index(drop=True), use_container_width=True)

    # Descarga
    out = df[show_cols]
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar hallazgos (CSV)", data=csv, file_name="caat1_fuera_horario.csv", mime="text/csv")

# ------------------------------
# CAAT 3 – Conciliación + Bancaria (beta)
# ------------------------------

def _normalize_money(x):
    try:
        return float(str(x).replace(",","").replace(" ",""))
    except Exception:
        return np.nan

def _date_within(d1, d2, days):
    try:
        return abs((d1.date() - d2.date()).days) <= days
    except Exception:
        return False

def caat3():
    st.subheader("CAAT 3 – Conciliación de logs vs transacciones")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. Sube **Logs** y **Transacciones** (CSV/XLSX).
2. Selecciona **ID** y **Fecha/Hora** en ambos.
3. Define tolerancia de tiempo para marcar desfases.

**Modo conciliación bancaria (beta)**: activa el switch para usar reglas típicas banco vs contabilidad (monto/fecha/referencia + fuzzy).
        """)
    banking_mode = st.toggle("Usar **Modo conciliación bancaria (beta)**", value=False, key=_k("bankmode"))

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Archivo A** (Logs o Extracto Bancario)")
        file_a = st.file_uploader("Sube A", type=["csv","xlsx"], key=_k("caat3_a"))
    with col_right:
        st.markdown("**Archivo B** (Transacciones o Libro Contable)")
        file_b = st.file_uploader("Sube B", type=["csv","xlsx"], key=_k("caat3_b"))

    if not (file_a and file_b):
        info_box("Sube ambos archivos para continuar.")
        return

    try:
        df_a = read_table_with_sheet(file_a, key_prefix="caat3_a")
        df_b = read_table_with_sheet(file_b, key_prefix="caat3_b")
    except Exception as e:
        error_box(str(e))
        return

    if df_a.empty or df_b.empty:
        warn_box("Uno de los archivos está vacío.")
        return

    if banking_mode:
        st.markdown("### Parámetros (bancario)")
        # Sugerencias
        a_ref_sug = suggest_column(df_a, ("referencia","detalle","descripcion","leyenda","memo"))
        a_amt_sug = suggest_column(df_a, ("monto","importe","amount","debito","credito","valor"))
        a_date_sug= suggest_column(df_a, ("fecha","date","fechahora","timestamp"))
        b_ref_sug = suggest_column(df_b, ("referencia","detalle","descripcion","leyenda","memo","cheque"))
        b_amt_sug = suggest_column(df_b, ("monto","importe","amount","debito","credito","valor"))
        b_date_sug= suggest_column(df_b, ("fecha","date","fechahora","timestamp"))

        c1,c2,c3 = st.columns(3)
        with c1:
            a_amt = st.selectbox("A: columna monto", df_a.columns, index=list(df_a.columns).index(a_amt_sug) if a_amt_sug in df_a.columns else 0, key=_k("a_amt"))
        with c2:
            a_date= st.selectbox("A: columna fecha", df_a.columns, index=list(df_a.columns).index(a_date_sug) if a_date_sug in df_a.columns else 0, key=_k("a_date"))
        with c3:
            a_ref = st.selectbox("A: referencia", df_a.columns, index=list(df_a.columns).index(a_ref_sug) if a_ref_sug in df_a.columns else 0, key=_k("a_ref"))

        d1,d2,d3 = st.columns(3)
        with d1:
            b_amt = st.selectbox("B: columna monto", df_b.columns, index=list(df_b.columns).index(b_amt_sug) if b_amt_sug in df_b.columns else 0, key=_k("b_amt"))
        with d2:
            b_date= st.selectbox("B: columna fecha", df_b.columns, index=list(df_b.columns).index(b_date_sug) if b_date_sug in df_b.columns else 0, key=_k("b_date"))
        with d3:
            b_ref = st.selectbox("B: referencia", df_b.columns, index=list(df_b.columns).index(b_ref_sug) if b_ref_sug in df_b.columns else 0, key=_k("b_ref"))

        x1,x2,x3 = st.columns(3)
        with x1:
            tol_days = st.number_input("Ventana de fechas (± días)", min_value=0, max_value=30, value=2, step=1, key=_k("tol_days"))
        with x2:
            tol_amt  = st.number_input("Tolerancia de monto (absoluta)", min_value=0.0, value=0.0, step=0.1, key=_k("tol_amt"))
        with x3:
            fuzzy_min = st.slider("Similitud mínima (fuzzy)", 0.0, 1.0, 0.80, 0.01, key=_k("fuzzy_min"))

        A = df_a[[a_amt,a_date,a_ref]].copy()
        B = df_b[[b_amt,b_date,b_ref]].copy()
        A.columns = ["amount","date","ref"]
        B.columns = ["amount","date","ref"]

        A["amount"] = A["amount"].apply(_normalize_money)
        B["amount"] = B["amount"].apply(_normalize_money)
        A["date"]   = parse_datetime_col(A["date"])
        B["date"]   = parse_datetime_col(B["date"])

        A = A.dropna(subset=["amount","date"]).reset_index(drop=True)
        B = B.dropna(subset=["amount","date"]).reset_index(drop=True)

        # 1) Exacto (monto + ventana de fecha + ref exacta si existe)
        A["_used"]=False
        B["_used"]=False
        matches = []

        # index por monto redondeado (acelera)
        idx_B = {}
        for i,row in B.iterrows():
            key = round(row["amount"], 2)
            idx_B.setdefault(key, []).append(i)

        def mark_match(iA, iB, label):
            matches.append({
                "from":"A","iA":iA,"iB":iB,"label":label,
                "amount_A":A.at[iA,"amount"],"date_A":A.at[iA,"date"],"ref_A":A.at[iA,"ref"],
                "amount_B":B.at[iB,"amount"],"date_B":B.at[iB,"date"],"ref_B":B.at[iB,"ref"],
            })
            A.at[iA,"_used"]=True
            B.at[iB,"_used"]=True

        # Exacto por monto + ventana (y ref exacta cuando ambas existen)
        for i,row in A.iterrows():
            if row["_used"]: 
                continue
            key = round(row["amount"], 2)
            cand_idx = idx_B.get(key, [])
            for j in cand_idx:
                if B.at[j,"_used"]:
                    continue
                if _date_within(row["date"], B.at[j,"date"], tol_days):
                    # si hay referencia en ambos, exigir igualdad exacta
                    refA, refB = str(row["ref"]).strip(), str(B.at[j,"ref"]).strip()
                    if (refA and refB and refA == refB) or (not refA or not refB):
                        mark_match(i, j, "exacto")
                        break

        # 2) Monto + ventana (tolerancia en monto)
        for i,row in A.iterrows():
            if row["_used"]:
                continue
            for j in range(len(B)):
                if B.at[j,"_used"]:
                    continue
                if abs(row["amount"] - B.at[j,"amount"]) <= tol_amt and _date_within(row["date"], B.at[j,"date"], tol_days):
                    mark_match(i, j, "monto+fecha")
                    break

        # 3) Fuzzy por referencia (con misma condición de monto/fecha, tolerancia)
        for i,row in A.iterrows():
            if row["_used"]:
                continue
            best_j, best_s = None, 0.0
            for j in range(len(B)):
                if B.at[j,"_used"]:
                    continue
                if abs(row["amount"] - B.at[j,"amount"]) <= tol_amt and _date_within(row["date"], B.at[j,"date"], tol_days):
                    s = fuzzy_ratio(row["ref"], B.at[j,"ref"])
                    if s > best_s:
                        best_s, best_j = s, j
            if best_j is not None and best_s >= fuzzy_min:
                mark_match(i, best_j, f"fuzzy ({best_s:.2f})")

        matched = pd.DataFrame(matches)
        unmatched_A = A[~A["_used"]][["amount","date","ref"]].rename(columns={"amount":"amount_A","date":"date_A","ref":"ref_A"}).reset_index(drop=True)
        unmatched_B = B[~B["_used"]][["amount","date","ref"]].rename(columns={"amount":"amount_B","date":"date_B","ref":"ref_B"}).reset_index(drop=True)

        st.markdown("### KPIs")
        total_A, total_B = len(A), len(B)
        conc = len(matched)
        cols = st.columns(4)
        with cols[0]: kpi_bar("Partidas A", total_A)
        with cols[1]: kpi_bar("Partidas B", total_B)
        with cols[2]: kpi_bar("Conciliadas", conc)
        with cols[3]:
            base = min(total_A,total_B) if min(total_A,total_B)>0 else 1
            kpi_bar("% conciliado", f"{(conc/base)*100:.2f}%")

        st.markdown("#### Coincidencias")
        st.dataframe(matched, use_container_width=True, height=280)
        st.download_button("⬇️ Descargar coincidencias (CSV)", matched.to_csv(index=False).encode("utf-8"),
                           file_name="caat3_bancario_matches.csv", mime="text/csv")

        cA, cB = st.columns(2)
        with cA:
            st.markdown("#### A sin conciliar")
            st.dataframe(unmatched_A, use_container_width=True, height=260)
            st.download_button("⬇️ Descargar A sin conciliar", unmatched_A.to_csv(index=False).encode("utf-8"),
                               file_name="caat3_bancario_A_unmatched.csv", mime="text/csv")
        with cB:
            st.markdown("#### B sin conciliar")
            st.dataframe(unmatched_B, use_container_width=True, height=260)
            st.download_button("⬇️ Descargar B sin conciliar", unmatched_B.to_csv(index=False).encode("utf-8"),
                               file_name="caat3_bancario_B_unmatched.csv", mime="text/csv")

        info_box("Las coincidencias *fuzzy* son indicativas, no evidencia concluyente. Revísalas antes de cerrar contablemente.", "⚠️")
        return

    # --------- modo clásico (no bancario): ejemplo mínimo ----------
    st.markdown("### Modo clásico (diferencias por fecha entre A y B)")
    # sugerencias
    a_id  = suggest_column(df_a, ("id","usuario","user","transaccion","registro"))
    a_dt  = suggest_column(df_a, ("fecha","date","timestamp","fechahora"))
    b_id  = suggest_column(df_b, ("id","usuario","user","transaccion","registro"))
    b_dt  = suggest_column(df_b, ("fecha","date","timestamp","fechahora"))

    c1,c2 = st.columns(2)
    with c1:
        a_id  = st.selectbox("A: columna ID", df_a.columns, index=list(df_a.columns).index(a_id) if a_id in df_a.columns else 0, key=_k("a_id"))
        a_dt  = st.selectbox("A: fecha", df_a.columns, index=list(df_a.columns).index(a_dt) if a_dt in df_a.columns else 0, key=_k("a_dt2"))
    with c2:
        b_id  = st.selectbox("B: columna ID", df_b.columns, index=list(df_b.columns).index(b_id) if b_id in df_b.columns else 0, key=_k("b_id"))
        b_dt  = st.selectbox("B: fecha", df_b.columns, index=list(df_b.columns).index(b_dt) if b_dt in df_b.columns else 0, key=_k("b_dt2"))

    tol_mins = st.number_input("Tolerancia (minutos) para marca de desfase", min_value=0, max_value=1440, value=10, step=5, key=_k("tol_mins"))

    A = df_a[[a_id,a_dt]].copy(); A.columns=["id","dt"]
    B = df_b[[b_id,b_dt]].copy(); B.columns=["id","dt"]
    A["dt"] = parse_datetime_col(A["dt"])
    B["dt"] = parse_datetime_col(B["dt"])
    A = A.dropna(subset=["dt"])
    B = B.dropna(subset=["dt"])

    merged = pd.merge(A,B,on="id", how="outer", suffixes=("_A","_B"))
    merged["diff_mins"] = (merged["dt_A"] - merged["dt_B"]).dt.total_seconds()/60
    merged["status"] = np.where(merged["dt_A"].isna(),"Solo en B",
                        np.where(merged["dt_B"].isna(),"Solo en A",
                        np.where(merged["diff_mins"].abs()<=tol_mins,"Ok","Desfase")))
    st.markdown("### KPIs")
    total = len(merged)
    desf  = int((merged["status"]=="Desfase").sum())
    soloA = int((merged["status"]=="Solo en A").sum())
    soloB = int((merged["status"]=="Solo en B").sum())
    cols = st.columns(4)
    with cols[0]: kpi_bar("Registros", total)
    with cols[1]: kpi_bar("Desfase", desf)
    with cols[2]: kpi_bar("Solo en A", soloA)
    with cols[3]: kpi_bar("Solo en B", soloB)

    st.dataframe(merged.sort_values("id").reset_index(drop=True), use_container_width=True, height=360)
    st.download_button("⬇️ Descargar conciliación (CSV)", merged.to_csv(index=False).encode("utf-8"),
                       file_name="caat3_conciliacion.csv", mime="text/csv")

# ------------------------------
# CAAT 2 / 4 / 5 – Placeholders útiles (ligeros)
# ------------------------------

def caat2():
    st.subheader("CAAT 2 – Auditoría de privilegios (roles críticos y SoD)")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. Sube tu **maestro Usuarios/Roles**.
2. Selecciona columnas **Usuario**, **Rol** y (opcional) **es_crítico**.
3. Si deseas reglas SoD, ingrésalas (una por línea) en formato `ROL_A -> ROL_B`.
4. Descarga hallazgos (críticos, SoD violadas, duplicados).
        """)
    file = st.file_uploader("Usuarios/Roles (CSV/XLSX)", type=["csv","xlsx"], key=_k("caat2_file"))
    if not file:
        info_box("Sube un archivo para comenzar.")
        return
    try:
        df = read_table_with_sheet(file, key_prefix="caat2")
    except Exception as e:
        error_box(str(e)); return
    if df.empty:
        warn_box("El archivo está vacío."); return

    user_sug = suggest_column(df, ("usuario","user","empleado","id"))
    role_sug = suggest_column(df, ("rol","role","módulo","modulo","perfil"))
    c1,c2,c3 = st.columns(3)
    with c1:
        user_col = st.selectbox("Columna Usuario", df.columns, index=list(df.columns).index(user_sug) if user_sug in df.columns else 0, key=_k("c2_user"))
    with c2:
        role_col = st.selectbox("Columna Rol", df.columns, index=list(df.columns).index(role_sug) if role_sug in df.columns else 0, key=_k("c2_role"))
    with c3:
        crit_col = st.selectbox("Columna es_crítico (opcional)", ["(ninguna)"]+list(df.columns), key=_k("c2_crit"))

    sod_txt = st.text_area("Reglas SoD (una por línea, formato `ROL_A -> ROL_B`)", key=_k("c2_sod"))
    df["__user__"] = df[user_col].astype(str).str.strip()
    df["__role__"] = df[role_col].astype(str).str.strip()
    if crit_col != "(ninguna)":
        df["__crit__"] = df[crit_col].astype(str).str.lower().isin(("1","true","sí","si","y","yes"))
    else:
        df["__crit__"] = False

    # críticos
    crit = df[df["__crit__"]][["__user__","__role__"]].rename(columns={"__user__":"usuario","__role__":"rol"})
    # SoD
    violations = []
    rules = []
    for line in sod_txt.splitlines():
        line=line.strip()
        if "->" in line:
            a,b = [x.strip() for x in line.split("->",1)]
            rules.append((a,b))
    if rules:
        roles_user = df.groupby("__user__")["__role__"].apply(set).to_dict()
        for u, roles in roles_user.items():
            for (a,b) in rules:
                if a in roles and b in roles:
                    violations.append({"usuario":u,"regla":f"{a} -> {b}"})
    sod_df = pd.DataFrame(violations)

    st.markdown("### KPIs")
    cols = st.columns(3)
    with cols[0]: kpi_bar("Usuarios", df["__user__"].nunique())
    with cols[1]: kpi_bar("Críticos", len(crit))
    with cols[2]: kpi_bar("SoD violadas", len(sod_df))

    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### Críticos")
        st.dataframe(crit, use_container_width=True, height=280)
        st.download_button("⬇️ Descargar críticos", crit.to_csv(index=False).encode("utf-8"), file_name="caat2_criticos.csv", mime="text/csv")
    with cB:
        st.markdown("#### SoD violadas")
        st.dataframe(sod_df, use_container_width=True, height=280)
        st.download_button("⬇️ Descargar SoD violadas", sod_df.to_csv(index=False).encode("utf-8"), file_name="caat2_sod_violadas.csv", mime="text/csv")

def caat4():
    st.subheader("CAAT 4 – Variación inusual de pagos (outliers)")
    with st.expander("¿Cómo usar este módulo?", expanded=False):
        st.markdown("""
1. Sube **Pagos** (CSV/XLSX).
2. Selecciona **Proveedor**, **Fecha**, **Monto**.
3. Ajusta el umbral robusto (|z|) para detectar picos/caídas atípicas.
        """)
    file = st.file_uploader("Pagos (CSV/XLSX)", type=["csv","xlsx"], key=_k("caat4_file"))
    if not file:
        info_box("Sube un archivo para comenzar.")
        return
    try:
        df = read_table_with_sheet(file, key_prefix="caat4")
    except Exception as e:
        error_box(str(e)); return
    if df.empty:
        warn_box("El archivo está vacío."); return

    prov_sug = suggest_column(df, ("proveedor","vendor","cliente","beneficiario"))
    date_sug = suggest_column(df, ("fecha","date","fechahora","timestamp"))
    amt_sug  = suggest_column(df, ("monto","importe","amount","valor","pago"))

    c1,c2,c3 = st.columns(3)
    with c1:
        prov = st.selectbox("Proveedor", df.columns, index=list(df.columns).index(prov_sug) if prov_sug in df.columns else 0, key=_k("c4_prov"))
    with c2:
        date = st.selectbox("Fecha", df.columns, index=list(df.columns).index(date_sug) if date_sug in df.columns else 0, key=_k("c4_date"))
    with c3:
        amt  = st.selectbox("Monto", df.columns, index=list(df.columns).index(amt_sug) if amt_sug in df.columns else 0, key=_k("c4_amt"))

    z_thr = st.slider("Umbral robusto |z|", 1.0, 6.0, 3.5, 0.5, key=_k("c4_z"))
    X = df[[prov,date,amt]].copy()
    X.columns=["prov","date","amt"]
    X["date"] = parse_datetime_col(X["date"])
    X["amt"]  = pd.to_numeric(X["amt"], errors="coerce")
    X = X.dropna(subset=["date","amt"]).reset_index(drop=True)

    # z-score robusto (MAD)
    grp = X.groupby("prov", as_index=False)
    outliers = []
    for p,g in grp:
        med = g["amt"].median()
        mad = (np.abs(g["amt"] - med)).median()
        if mad == 0:
            z = np.zeros(len(g))
        else:
            z = 0.6745 * (g["amt"] - med) / mad
        o = g[np.abs(z) >= z_thr].copy()
        o["z_robusto"]=z[np.abs(z)>=z_thr]
        outliers.append(o)
    outs = pd.concat(outliers, ignore_index=True) if outliers else pd.DataFrame(columns=["prov","date","amt","z_robusto"])

    st.markdown("### KPIs")
    cols = st.columns(3)
    with cols[0]: kpi_bar("Pagos", len(X))
    with cols[1]: kpi_bar("Proveedores", X["prov"].nunique())
    with cols[2]: kpi_bar("Outliers", len(outs))

    st.dataframe(outs.sort_values(["prov","date"]).reset_index(drop=True), use_container_width=True, height=360)
    st.download_button("⬇️ Descargar outliers (CSV)", outs.to_csv(index=False).encode("utf-8"), file_name="caat4_outliers.csv", mime="text/csv")

def caat5():
    st.subheader("CAAT 5 – (placeholder)")
    st.write("Demostración ligera: agrega aquí otra prueba CAAT o tu propio análisis.")
    st.info("Para el entregable del curso, ya tienes CAAT 1, 2, 3 (con modo bancario), 4 y Modo Libre con KPIs y descargas.")

# ------------------------------
# Modo Libre – Perfilador simple
# ------------------------------

def modo_libre():
    st.subheader("Modo Libre – Analiza cualquier archivo")
    with st.expander("¿Qué hace este modo?", expanded=False):
        st.markdown("""
- Sube **cualquier CSV/XLSX**.  
- Detectamos tipos, **KPIs** rápidos y te damos panel de **filtros**, **duplicados**, resumen por columnas y **descarga**.  
- Útil para evaluar archivos arbitrarios (lo que el profesor quiera subir).
        """)
    file = st.file_uploader("Sube tu archivo", type=["csv","xlsx"], key=_k("free_file"))
    if not file:
        info_box("Sube un archivo para comenzar.")
        return

    try:
        df = read_table_with_sheet(file, key_prefix="free")
    except Exception as e:
        error_box(str(e)); return
    if df.empty:
        warn_box("El archivo está vacío."); return

    st.markdown("### KPIs")
    cols = st.columns(4)
    with cols[0]: kpi_bar("Filas", len(df))
    with cols[1]: kpi_bar("Columnas", len(df.columns))
    with cols[2]: kpi_bar("Nulos (%)", f"{(df.isna().mean().mean()*100):.2f}%")
    with cols[3]:
        num_cols = df.select_dtypes(include=[np.number]).columns
        kpi_bar("Cols Numéricas", len(num_cols))

    st.markdown("### Vista previa")
    st.dataframe(df.head(200), use_container_width=True, height=320)

    st.markdown("### Filtros rápidos")
    with st.container():
        fcols = st.multiselect("Columnas para filtrar", df.columns, key=_k("free_fcols"))
        df_f = df.copy()
        for c in fcols:
            if df[c].dtype.kind in "biufc":
                minv, maxv = float(np.nanmin(df[c])), float(np.nanmax(df[c]))
                r = st.slider(f"{c} (rango)", min_value=minv, max_value=maxv, value=(minv, maxv), key=_k(f"rng_{c}"))
                df_f = df_f[(df_f[c] >= r[0]) & (df_f[c] <= r[1])]
            else:
                vals = sorted(list(df[c].dropna().astype(str).unique()))[:200]
                pick = st.multiselect(f"{c} (valores)", vals, key=_k(f"pick_{c}"))
                if pick:
                    df_f = df_f[df_f[c].astype(str).isin(pick)]
    st.markdown("### Resultado filtrado")
    st.dataframe(df_f.head(300), use_container_width=True, height=320)

    # duplicados
    st.markdown("### Duplicados")
    dup_cols = st.multiselect("Columnas para detectar duplicados", df.columns, key=_k("free_dups"))
    if dup_cols:
        dups = df[df.duplicated(subset=dup_cols, keep=False)].sort_values(by=dup_cols)
        st.dataframe(dups, use_container_width=True, height=260)
        st.download_button("⬇️ Descargar duplicados", dups.to_csv(index=False).encode("utf-8"), file_name="modo_libre_duplicados.csv", mime="text/csv")

    # Descargas
    st.markdown("### Descargas")
    st.download_button("⬇️ Descargar (CSV) datos filtrados", df_f.to_csv(index=False).encode("utf-8"), file_name="modo_libre_filtrado.csv", mime="text/csv")

# ------------------------------
# App
# ------------------------------

def main():
    st.set_page_config(page_title="Aprendizaje Colaborativo y Práctico – 2do Parcial", layout="wide", page_icon="🧭")
    st.title("Aprendizaje Colaborativo y Práctico – 2do Parcial")
    st.caption("Suite de herramientas CAAT para auditoría asistida.")

    tabs = st.tabs(["CAAT 1–5", "Modo libre"])
    with tabs[0]:
        with st.expander("Ayuda general (antes de empezar)", expanded=True):
            st.markdown("""
**¿Qué hace esta app?** Te permite correr **5 CAAT** comunes de auditoría y un **Modo libre** para explorar cualquier archivo.

**Errores comunes y cómo resolver:**
- *No se pudo leer el Excel*: verifica que **no esté protegido** y que sea **.xlsx válido**. Asegúrate de tener **openpyxl**.
- *Fechas vacías/NaT*: revisa el **formato** o elige bien la columna.
- *Selectbox duplicado*: ya no aplica; usamos **keys únicos** para los widgets.

**Descargas**: cada módulo genera un **CSV** con hallazgos.
            """)
        st.markdown("---")

        # Subpestañas internas (ligeras)
        sub = st.tabs(["CAAT 1", "CAAT 2", "CAAT 3", "CAAT 4", "CAAT 5"])
        with sub[0]: caat1()
        with sub[1]: caat2()
        with sub[2]: caat3()
        with sub[3]: caat4()
        with sub[4]: caat5()

    with tabs[1]:
        modo_libre()

if __name__ == "__main__":
    main()

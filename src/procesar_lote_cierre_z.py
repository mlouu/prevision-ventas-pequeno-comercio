"""
Procesa en lote los informes Z semanales de `datos_bruto/Informes Z semanal/` y
construye la serie semanal del TFM, que es la que entra al modelo.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from parser_cierre_z import parsear, validar

# El informe cuadra al céntimo consigo mismo, así que el margen es de redondeo.
TOLERANCIA_VENTA = 0.05

# El descuadre por departamento: la suma por departamento excedía la venta
# total en 28 semanas.
TOLERANCIA_DEPARTAMENTOS = 0.05

RE_NOMBRE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$")

# La deflactación: la serie se deflacta con el IPC de alimentos y bebidas no
# alcohólicas de Almería, no con el general.
IPC_AMBITO = "almeria"
IPC_CONCEPTO = "alimentos_bebidas_no_alc"


def _aplicar_arqueos_caja(serie, departamentos, carpeta_externos):
    """Suma a la serie las ventas que el back-office perdio."""
    ruta_tot = carpeta_externos / "arqueos_caja_totales.csv"
    ruta_dep = carpeta_externos / "arqueos_caja_departamentos.csv"
    serie["corregida"] = False
    if not ruta_tot.exists():
        return serie, departamentos

    arq = pd.read_csv(ruta_tot, parse_dates=["fecha"])
    dep_arq = pd.read_csv(ruta_dep, parse_dates=["fecha"])

    for _, sem in serie.iterrows():
        dentro = arq[(arq["fecha"] >= sem["periodo_inicio"]) & (arq["fecha"] <= sem["periodo_fin"])]
        if dentro.empty:
            continue
        i = sem.name

        # La venta fidelizada hay que reconstruirla en absoluto antes de volver
        # a hacer el porcentaje: los % no se suman.
        vf = (serie.at[i, "pct_venta_fidelizada"] / 100 * serie.at[i, "venta_total"]
              + (dentro["pct_venta_fidelizada"] / 100 * dentro["venta_total"]).sum())

        for col in ("venta_total", "venta_neta", "operaciones", "tickets",
                    "rectificativas", "cupones_emitidos", "cupones_redimidos"):
            serie.at[i, col] = serie.at[i, col] + dentro[col].sum()

        serie.at[i, "pct_venta_fidelizada"] = round(100 * vf / serie.at[i, "venta_total"], 2)
        serie.at[i, "pct_tickets_fidelizados"] = round(
            100 * serie.at[i, "cupones_emitidos"] / serie.at[i, "tickets"], 2)
        serie.at[i, "corregida"] = True

        d = dep_arq[(dep_arq["fecha"] >= sem["periodo_inicio"])
                    & (dep_arq["fecha"] <= sem["periodo_fin"])]
        extra = (d.groupby("departamento")[["articulos", "importe"]].sum().reset_index())
        extra.insert(0, "periodo_inicio", sem["periodo_inicio"])
        departamentos = pd.concat([departamentos, extra], ignore_index=True)
        departamentos = (departamentos.groupby(["periodo_inicio", "departamento"], as_index=False)
                         .agg({"articulos": "sum", "importe": "sum"}))

        print(f"  [la reconstrucción con arqueos] {sem['periodo_inicio'].date()}: sumados "
              f"{len(dentro)} arqueos de caja, {dentro['venta_total'].sum():,.2f} EUR "
              f"y {int(dentro['operaciones'].sum())} operaciones que el back-office perdio.")

    return serie, departamentos

#  / los cambios de IVA. Los tipos de IVA cambiaron cinco veces dentro de la ventana:
# la rebaja temporal a los alimentos basicos y su retirada por tramos. El mismo
# producto tributo al 0 %, al 2 % y al 4 %, asi que el tipo no identifica al
# producto y no se pueden comparar dos semanas tipo a tipo. Lo que si es
# estable es el grupo de producto, y son dos:
#
#   g1  basicos             0 % -> 2 % -> 4 %
#   g2  alimentacion gral.  5 % -> 7,5 % -> 10 %  (salvo el aceite de oliva: ver ACEITE_OLIVA)
#
# El 21 % (bebidas alcoholicas, drogueria, bazar) nunca cambio y ademas queda
# fuera del indice con el que se deflacta (alimentos y bebidas no alcoholicas),
# asi que no entra en el calculo.
GRUPO_IVA = {0.0: "g1", 2.0: "g1", 4.0: "g1", 5.0: "g2", 7.5: "g2", 10.0: "g2"}

# Regimen vigente desde el 2025-02-10, al que se normaliza todo el periodo.
TIPO_IVA_REFERENCIA = {"g1": 0.04, "g2": 0.10}

# El aceite de oliva no sigue al grupo g2.
ACEITE_OLIVA = r"^(?:\(P\))?ACEITE (?:DE )?(?:OLIVA|ORUJO)"
FIN_ACEITE_AL_5 = pd.Timestamp("2024-07-01")


def _aceite_oliva_al_5(impuestos: pd.DataFrame,
                       ruta_consolidado: str | Path) -> pd.Series | None:
    """Base del aceite de oliva dentro del 5 %, por semana, hasta junio de 2024."""
    ruta = Path(ruta_consolidado)
    if not ruta.exists():
        return None
    art = pd.read_parquet(ruta, columns=["descripcion", "total_venta", "periodo_inicio"])
    art = art[art["descripcion"].str.contains(ACEITE_OLIVA, na=False)
              & (art["periodo_inicio"] < FIN_ACEITE_AL_5)]
    aceite_mes = art.groupby(art["periodo_inicio"].dt.to_period("M"))["total_venta"].sum()

    al5 = impuestos[(impuestos["tipo"] == 5.0)
                    & (impuestos["periodo_inicio"] < FIN_ACEITE_AL_5)]
    al5 = al5.groupby("periodo_inicio")["base"].sum()
    mes = al5.index.to_period("M")
    cuota_aceite = (mes.map(aceite_mes).fillna(0.0).to_numpy()
                    / al5.groupby(mes).transform("sum").to_numpy())
    return (al5 * cuota_aceite.clip(0, 1)).rename("aceite_oliva_al_5")


def _tipo_iva_efectivo(impuestos: pd.DataFrame,
                       aceite_oliva: pd.Series | None = None) -> pd.DataFrame:
    """Tipo de IVA que soporto de verdad la cesta de alimentacion, por semana."""
    imp = impuestos.copy()
    imp["grupo"] = imp["tipo"].map(GRUPO_IVA)
    imp = imp[imp["grupo"].notna()]

    base = imp.pivot_table(index="periodo_inicio", columns="grupo",
                           values="base", aggfunc="sum").fillna(0.0)
    for g in TIPO_IVA_REFERENCIA:
        if g not in base:
            base[g] = 0.0

    total = base.sum(axis=1)
    cuota = imp.groupby("periodo_inicio")["cuota"].sum()
    if aceite_oliva is not None:
        # El aceite de oliva al 5 % pasa de g2 a g1: su tipo de hoy es el 4 %.
        movido = aceite_oliva.reindex(base.index, fill_value=0.0)
        base["g1"] += movido
        base["g2"] -= movido
    referencia = sum(base[g] * t for g, t in TIPO_IVA_REFERENCIA.items())

    return pd.DataFrame({
        "periodo_inicio": total.index,
        "tipo_iva_efectivo": (cuota / total).to_numpy(),
        "tipo_iva_referencia": (referencia / total).to_numpy(),
    })


def _periodo_desde_nombre(ruta: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Extrae (inicio, fin) del nombre del fichero, o None si no sigue la convención."""
    m = RE_NOMBRE.match(ruta.stem)
    if not m:
        return None
    return pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2))


def _indice_ipc_diario(ruta_ipc: str | Path) -> pd.Series | None:
    """Serie diaria del índice de precios, interpolada desde el dato mensual."""
    ruta_ipc = Path(ruta_ipc)
    if not ruta_ipc.exists():
        return None
    ipc = pd.read_csv(ruta_ipc)
    ipc = ipc[(ipc["ambito"] == IPC_AMBITO) & (ipc["concepto"] == IPC_CONCEPTO)]
    if ipc.empty:
        return None
    ipc = ipc.assign(fecha=pd.to_datetime(ipc["periodo"] + "-15")).sort_values("fecha")
    serie = ipc.set_index("fecha")["indice"].astype(float)
    diario = serie.resample("D").interpolate("linear")
    # Los anclajes van a mitad de mes, así que `resample` no genera ningún día
    # posterior al día 15 del último mes publicado: la última quincena se queda
    # literalmente sin filas, y las semanas que caen ahí salían sin deflactar
    # o, peor, con una media sesgada calculada solo sobre los días que sí
    # tenían índice.
    calendario = pd.date_range(
        diario.index.min() - pd.Timedelta(days=45),
        diario.index.max() + pd.Timedelta(days=45),
        freq="D",
    )
    return diario.reindex(calendario).ffill().bfill()


def procesar_todos(
    carpeta_bruto: str | Path = "datos_bruto/Informes Z semanal",
    carpeta_limpio: str | Path = "datos_limpio",
    ruta_ipc: str | Path = "datos_externos/ipc_ine.csv",
    carpeta_externos: str | Path = "datos_externos",
) -> pd.DataFrame:
    """Parsea todos los PDF de `carpeta_bruto` y escribe la serie semanal."""
    carpeta_bruto = Path(carpeta_bruto)
    carpeta_limpio = Path(carpeta_limpio)
    carpeta_limpio.mkdir(exist_ok=True)

    ficheros = sorted(f for f in carpeta_bruto.iterdir() if f.suffix.lower() == ".pdf")
    if not ficheros:
        raise FileNotFoundError(f"No hay PDF en {carpeta_bruto}")

    filas: list[dict] = []
    departamentos: list[pd.DataFrame] = []
    impuestos: list[pd.DataFrame] = []
    informe: list[dict] = []

    for ruta in ficheros:
        periodo = _periodo_desde_nombre(ruta)
        if periodo is None:
            informe.append({"fichero": ruta.name, "error": "nombre fuera de convención", "ok": False})
            print(f"  [Error] {ruta.name}: el nombre no sigue AAAA-MM-DD_AAAA-MM-DD.pdf")
            continue
        inicio, fin = periodo

        try:
            datos = parsear(ruta)
            val = validar(datos)
        except Exception as exc:
            informe.append({"fichero": ruta.name, "error": str(exc), "ok": False})
            print(f"  [Error] {ruta.name}: {exc}")
            continue

        # El nombre manda para el periodo, pero el PDF tiene que corroborar el
        # inicio.
        fecha_pdf = pd.to_datetime(datos["fecha"], format="%d-%m-%Y", errors="coerce")
        inicio_cuadra = fecha_pdf == inicio

        dias = (fin - inicio).days + 1
        dif_docs = (val.get("venta_total_documentos") or {}).get("diferencia")
        dif_dptos = (val.get("venta_total_departamentos") or {}).get("diferencia")
        # Las ventas perdidas: documentos que el informe abarca en su rango
        # pero no cuenta.
        continuidad = val.get("continuidad_documentos") or {}
        faltan_docs = continuidad.get("documentos_que_faltan", 0)

        ok = (
            inicio_cuadra
            and dif_docs is not None
            and abs(dif_docs) <= TOLERANCIA_VENTA
            and faltan_docs == 0
        )
        aviso_dptos = dif_dptos is not None and abs(dif_dptos) > TOLERANCIA_DEPARTAMENTOS

        fid = datos.get("fidelizacion") or {}
        filas.append({
            "periodo_inicio": inicio,
            "periodo_fin": fin,
            "dias": dias,
            # Las semanas completas: la serie se construye solo con semanas
            # naturales completas.
            "semana_completa": dias == 7,
            "almacen": datos.get("almacen"),
            "venta_neta": datos.get("venta_neta"),
            "venta_total": datos.get("venta_total"),
            "operaciones": datos.get("operaciones"),
            "tickets": datos.get("tickets"),
            "albaranes": datos.get("albaranes_operaciones"),
            "rectificativas": datos.get("rectificativas_operaciones"),
            "pct_tickets_fidelizados": fid.get("pct_tickets_fidelizados"),
            "pct_venta_fidelizada": fid.get("pct_venta_fidelizada"),
            "cupones_emitidos": fid.get("cupones_emitidos"),
            "cupones_redimidos": fid.get("cupones_redimidos_cantidad"),
            "fichero": ruta.name,
        })

        dep = datos["departamentos"].copy()
        dep.insert(0, "periodo_inicio", inicio)
        departamentos.append(dep)

        imp = datos["impuestos"].copy()
        imp.insert(0, "periodo_inicio", inicio)
        impuestos.append(imp)

        dif_imp = (val.get("venta_neta_impuestos") or {}).get("diferencia")

        informe.append({
            "fichero": ruta.name,
            "periodo_inicio": inicio.date(),
            "periodo_fin": fin.date(),
            "dias": dias,
            "almacen": datos.get("almacen"),
            "inicio_cuadra_con_pdf": inicio_cuadra,
            "diff_venta_documentos": dif_docs,
            "diff_venta_departamentos": dif_dptos,
            "aviso_departamentos": aviso_dptos,
            "series_numeradas": continuidad.get("series_comprobadas"),
            "documentos_abarcados": continuidad.get("documentos_abarcados"),
            "documentos_contados": continuidad.get("documentos_contados"),
            "documentos_faltan": faltan_docs,
            "diff_venta_neta_impuestos": dif_imp,
            "ok": ok,
        })
        if not inicio_cuadra:
            print(f"  [Error] {ruta.name}: el PDF dice fecha={datos['fecha']}, "
                  f"el nombre dice {inicio.date()}. Fichero mal nombrado.")
        elif faltan_docs:
            print(f"  [Aviso] {ruta.name}: faltan {faltan_docs} documentos en la "
                  f"numeracion — al informe le faltan ventas")
        elif not ok:
            print(f"  [Aviso] {ruta.name}: diff_venta_documentos={dif_docs}")

    serie = pd.DataFrame(filas).sort_values("periodo_inicio").reset_index(drop=True)
    dep_df = pd.concat(departamentos, ignore_index=True) if departamentos else pd.DataFrame()
    serie, dep_df = _aplicar_arqueos_caja(serie, dep_df, Path(carpeta_externos))

    almacenes = serie["almacen"].dropna().unique()
    if len(almacenes) > 1:
        raise ValueError(
            f"El lote mezcla almacenes distintos: {sorted(almacenes)}. La tienda "
            "tiene más de un almacén y consolidarlos falsearía la serie."
        )

    imp_df = pd.concat(impuestos, ignore_index=True) if impuestos else pd.DataFrame()

    # --- la deflactación: columna deflactada, a euros constantes del inicio de la serie ---
    diario = _indice_ipc_diario(ruta_ipc)
    if diario is None:
        print("  [Aviso] no se encontró el IPC: la serie va solo en euros nominales.")
        serie["ipc_alimentos"] = pd.NA
        serie["ipc_alimentos_ic"] = pd.NA
        serie["venta_neta_real"] = pd.NA
    else:
        # Índice medio de los días que cubre cada semana, no el del día de inicio:
        # una semana a caballo entre dos meses toma su parte de cada uno.
        serie["ipc_alimentos"] = [
            diario.loc[i:f].mean() if not diario.loc[i:f].empty else pd.NA
            for i, f in zip(serie["periodo_inicio"], serie["periodo_fin"])
        ]

        # El IPC del INE mide precios con impuestos y `venta_neta` es base
        # imponible: sin corregirlo, la subida del IVA de los alimentos
        # (+1,31 % en la misma cesta) contaría como inflación.
        if not imp_df.empty:
            aceite = _aceite_oliva_al_5(imp_df, carpeta_limpio / "consolidado_mensual.parquet")
            if aceite is None:
                print("  [Aviso] sin consolidado mensual: el aceite de oliva de 2024 "
                      "se trata como el resto del 5 % (ver ACEITE_OLIVA).")
            serie = serie.merge(_tipo_iva_efectivo(imp_df, aceite), on="periodo_inicio", how="left")
            serie["ipc_alimentos_ic"] = (
                serie["ipc_alimentos"]
                * (1 + serie["tipo_iva_referencia"]) / (1 + serie["tipo_iva_efectivo"])
            )
        else:
            serie["ipc_alimentos_ic"] = serie["ipc_alimentos"]

        base = serie["ipc_alimentos_ic"].dropna()
        if base.empty:
            serie["venta_neta_real"] = pd.NA
        else:
            # Base = primera semana de la serie, así la deflactada arranca igual
            # que la nominal y la separación entre ambas es la inflación acumulada.
            ipc_base = float(base.iloc[0])
            serie["venta_neta_real"] = serie["venta_neta"] / (serie["ipc_alimentos_ic"] / ipc_base)

    tabla_dep = dep_df   # ya viene consolidada y con los arqueos sumados

    serie.to_parquet(carpeta_limpio / "serie_semanal.parquet", index=False)
    if not tabla_dep.empty:
        tabla_dep.to_parquet(carpeta_limpio / "serie_semanal_departamentos.parquet", index=False)
    if not imp_df.empty:
        imp_df.to_parquet(carpeta_limpio / "serie_semanal_impuestos.parquet", index=False)
    informe_df = pd.DataFrame(informe)
    informe_df.to_csv(
        carpeta_limpio / "informe_procesado_cierre_z.csv", index=False, encoding="utf-8-sig"
    )

    n_ok = int(informe_df["ok"].sum()) if "ok" in informe_df else 0
    n_avisos = int(informe_df.get("aviso_departamentos", pd.Series(dtype=bool)).sum())
    completas = int(serie["semana_completa"].sum())
    print(
        f"\nProcesados {len(ficheros)} informes Z: {n_ok} OK, "
        f"{len(ficheros) - n_ok} con avisos/errores."
    )
    print(f"  {completas} semanas completas y {len(serie) - completas} parciales.")
    print(f"  {n_avisos} con descuadre por departamento.")
    col = informe_df.get("documentos_faltan")
    if col is not None and (col > 0).any():
        print(f"  {int((col > 0).sum())} con huecos en la numeracion de documentos "
              f": les faltan ventas que si estan en la caja.")
        for _, r in informe_df[col > 0].iterrows():
            print(f"    {r['fichero']}: faltan {int(r['documentos_faltan'])} documentos")
    if len(serie):
        # Flecha ascii a propósito: la consola de Windows usa cp1252 y no sabe
        # codificar "→", así que el script reventaría justo al terminar.
        print(
            f"  Cobertura: {serie['periodo_inicio'].min().date()} -> "
            f"{serie['periodo_fin'].max().date()}"
        )
    print(f"Detalle en {carpeta_limpio / 'informe_procesado_cierre_z.csv'}")

    return serie


if __name__ == "__main__":
    procesar_todos()

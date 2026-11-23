"""
Parser del «Diario de ventas por artículo» del TPV. Cada línea del fichero es
una fila CSV entera metida en un único campo entrecomillado, así que primero se
desenvuelve y después se lee con `csv`.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd

# El TPV exporta en codificación Windows, no UTF-8.
ENCODING = "cp1252"

# 15 campos por artículo, en el orden en que los emite el informe.
COLUMNAS = [
    "cod_articulo",
    "descripcion",
    "unidades",
    "precio",
    "total_venta",
    "costo_medio",
    "beneficio_cm",
    "pct_rent_cm",
    "costo_ultimo",
    "beneficio_cu",
    "pct_rent_cu",
    "tarifa_costo",
    "beneficio_tc",
    "pct_rent_tc",
    "oferta",
]

NUMERICAS = [c for c in COLUMNAS if c not in ("cod_articulo", "descripcion", "oferta")]

# Líneas de una sola celda que no son nombre de departamento.
RUIDO_UNICELDA = re.compile(
    r"^(Extracto del|Diario de ventas|Almac[eé]n:|Fecha:|P[áa]g\.|Precios sin|TOTAL$)",
    re.IGNORECASE,
)

RE_PERIODO = re.compile(r"Extracto del:\s*(\d{2}/\d{2}/\d{4})\s*al\s*(\d{2}/\d{2}/\d{4})")
RE_ALMACEN = re.compile(r"Almac[eé]n:\s*(.+)")
RE_EMITIDO = re.compile(r"Fecha:\s*(\d{2}/\d{2}/\d{4})")

# Códigos que no son artículos dados de alta con nombre propio. Los dos son
# venta real y suman a los totales del informe; se etiquetan para poder
# filtrarlos según el análisis.
#   D1xx   producto que existe físicamente y tiene código de barras, pero no
#          está dado de alta con nombre propio: se cobra contra un código de
#          referencia del departamento. Sale con coste 0 porque el informe no
#          tiene ficha de la que sacarlo, no porque el margen sea del 100 %.
#   RExxx  recargas de móvil. Margen pequeño, pero es facturación.
RE_GENERICO_DPTO = re.compile(r"^D\d+$", re.IGNORECASE)
RE_RECARGA = re.compile(r"^RE\d+$", re.IGNORECASE)


# Genéricas de departamento con código numérico.
_SOLO_LETRAS = re.compile(r"[^A-ZÁÉÍÓÚÑ]")


def _normaliza(texto: str) -> str:
    """Deja solo letras en mayúscula, para comparar nombres sin ruido."""
    return _SOLO_LETRAS.sub("", str(texto).upper())


def _es_generico_por_nombre(descripcion: str, departamento: str) -> bool:
    """¿La descripción es el nombre del departamento, y no un artículo real?"""
    desc, dpto = _normaliza(descripcion), _normaliza(departamento)
    if len(desc) < 4 or len(dpto) < 4:
        return False
    return desc == dpto or desc.startswith(dpto) or dpto.startswith(desc)


def _tipo_fila(cod: str) -> str:
    """Clasifica la fila por la forma del código de artículo."""
    if cod.isdigit():
        return "articulo"
    if RE_RECARGA.match(cod):
        return "recarga"
    if RE_GENERICO_DPTO.match(cod):
        return "generico_dpto"
    return "otro"


def _celdas(linea: str) -> list[str]:
    """Desenvuelve una línea del informe y devuelve sus celdas no vacías."""
    linea = linea.rstrip("\r\n")
    fila = next(csv.reader(io.StringIO(linea)), [])
    if len(fila) <= 1 and len(linea) >= 2 and linea.startswith('"') and linea.endswith('"'):
        linea = linea[1:-1].replace('""', '"')
        fila = next(csv.reader(io.StringIO(linea)), [])
    return [c.strip() for c in fila if c.strip()]


def _num(valor: str) -> float | None:
    """Convierte '1,155.00' -> 1155.0."""
    if valor is None:
        return None
    valor = valor.replace(",", "").strip()
    if not valor:
        return None
    try:
        return float(valor)
    except ValueError:
        return None


def parsear(ruta: str | Path) -> tuple[pd.DataFrame, dict]:
    """Parsea un informe del TPV."""
    ruta = Path(ruta)
    meta: dict = {"fichero": ruta.name, "totales_informe": {}, "anomalias": []}
    filas: list[dict] = []
    departamento: str | None = None
    n_lineas = 0

    with ruta.open("r", encoding=ENCODING, errors="replace", newline="") as fh:
        for n_lineas, linea in enumerate(fh, start=1):
            celdas = _celdas(linea)
            if not celdas:
                continue

            texto = " ".join(celdas)

            # --- Metadatos de cabecera -------------------------------------
            if m := RE_PERIODO.search(texto):
                meta.setdefault("periodo_inicio", m.group(1))
                meta.setdefault("periodo_fin", m.group(2))
                continue
            if m := RE_ALMACEN.search(texto):
                meta.setdefault("almacen", m.group(1).strip())
                continue
            if m := RE_EMITIDO.search(texto):
                meta.setdefault("emitido", m.group(1))
                # La primera línea trae además el nombre de la tienda.
                if len(celdas) > 1:
                    meta.setdefault("tienda", celdas[0])
                continue

            # --- Cabecera de columnas repetida en cada página --------------
            if celdas[0].startswith("Cód. Art.") or celdas[0].startswith("C\ufffdd"):
                continue

            # --- Fila de artículo: la única con los 15 campos ---------------
            # Cuidado: no vale exigir que el código sea numérico.
            if len(celdas) == len(COLUMNAS):
                registro = dict(zip(COLUMNAS, celdas))
                registro["departamento"] = departamento
                filas.append(registro)
                continue

            # --- Subtotal de departamento ----------------------------------
            if texto.startswith("Total Dpto."):
                continue

            # --- Total general del informe ---------------------------------
            # La fila total sigue el mismo layout que una de artículo, pero los
            # campos de coste son promedios unitarios (coste medio por unidad),
            # mientras que los de beneficio sí son importes acumulados.
            if celdas[0] == "TOTAL":
                v = [_num(c) for c in celdas[1:]]
                def _v(i):
                    return v[i] if len(v) > i else None
                meta["totales_informe"] = {
                    "unidades": _v(0),
                    "precio_medio": _v(1),
                    "total_venta": _v(2),
                    # sobre coste medio
                    "costo_medio_unitario": _v(3),
                    "beneficio_cm": _v(4),
                    "pct_rent_cm": _v(5),
                    # sobre último coste
                    "costo_ultimo_unitario": _v(6),
                    "beneficio_cu": _v(7),
                    "pct_rent_cu": _v(8),
                    # sobre tarifa de costo - es el margen que reproduce el
                    # informe "ranking de ventas por dpto."
                    "tarifa_costo_unitaria": _v(9),
                    "beneficio_tc": _v(10),
                    "pct_rent_tc": _v(11),
                }
                continue

            # --- Línea suelta: o es ruido de pie/cabecera, o es el nombre
            #     del departamento que abre el bloque siguiente -------------
            if len(celdas) == 1:
                if not RUIDO_UNICELDA.match(celdas[0]):
                    departamento = celdas[0]
                continue

            # --- Resto: continuación del subtotal de departamento ----------
            # El informe parte el subtotal en dos líneas; la segunda son
            # cifras sueltas sin etiqueta. No aporta nada, se descarta.
            if all(_num(c) is not None for c in celdas):
                continue

            meta["anomalias"].append(f"L{n_lineas}: sin clasificar -> {celdas[:4]}")

    meta["lineas_leidas"] = n_lineas

    df = pd.DataFrame(filas, columns=COLUMNAS + ["departamento"])
    for col in NUMERICAS:
        df[col] = df[col].map(_num)
    # El código no es entero: conviven códigos numéricos con otros tipo D101.
    df["cod_articulo"] = df["cod_articulo"].astype("string")
    df["descripcion"] = df["descripcion"].str.strip()
    df["oferta"] = df["oferta"].eq("S")
    # Distingue artículo con alta propia / producto sin alta (D1xx) / recarga.
    df["tipo_fila"] = df["cod_articulo"].map(_tipo_fila).astype("string")
    # Segunda pasada, solo sobre lo que la primera dejó como "articulo": las
    # genéricas de código numérico, que se delatan por el nombre y no por el
    # código.
    coincide_nombre = pd.Series(
        [
            _es_generico_por_nombre(d, p)
            for d, p in zip(df["descripcion"], df["departamento"])
        ],
        index=df.index,
        dtype=bool,
    )
    es_generica_num = df["tipo_fila"].eq("articulo") & coincide_nombre
    df.loc[es_generica_num, "tipo_fila"] = "generico_dpto_num"
    # Se mantiene por compatibilidad con el notebook 01: todo lo que no es
    # artículo con alta propia. Preferir `tipo_fila`, que además dice qué es.
    df["venta_generica"] = df["tipo_fila"].ne("articulo")

    # Se propaga para poder detectar si un lote mezcla almacenes distintos
    # antes de consolidar.
    df["almacen"] = meta.get("almacen")

    # El periodo como fecha, para poder apilar varios ficheros después.
    if "periodo_inicio" in meta:
        df["periodo_inicio"] = pd.to_datetime(meta["periodo_inicio"], format="%d/%m/%Y")
        df["periodo_fin"] = pd.to_datetime(meta["periodo_fin"], format="%d/%m/%Y")

    return df, meta


# Campos del total declarado que son importes acumulados y, por tanto, se
# pueden contrastar sumando nuestra columna homónima.
VALIDABLES = [
    "unidades",
    "total_venta",
    "beneficio_cm",
    "beneficio_cu",
    "beneficio_tc",
]


def validar(df: pd.DataFrame, meta: dict) -> dict:
    """Compara nuestros agregados con los totales que declara el informe."""
    declarado = meta.get("totales_informe", {})
    resultado = {}
    for campo in VALIDABLES:
        if campo not in df.columns:
            continue
        esperado = declarado.get(campo)
        obtenido = float(df[campo].sum())
        resultado[campo] = {
            "declarado": esperado,
            "calculado": round(obtenido, 2),
            "diferencia": None if esperado is None else round(obtenido - esperado, 2),
        }
    return resultado


if __name__ == "__main__":
    import json
    import sys

    ruta = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "datos_bruto/Diario de ventas por artículo acumulado mensual/05-2026.CSV"
    )
    df, meta = parsear(ruta)

    print("=" * 70)
    print("Metadatos")
    print("=" * 70)
    for k, v in meta.items():
        if k != "anomalias":
            print(f"  {k}: {v}")

    print()
    print("=" * 70)
    print("Validación contra los totales del informe")
    print("=" * 70)
    print(json.dumps(validar(df, meta), indent=2, ensure_ascii=False))

    print()
    print("=" * 70)
    print(f"Artículos: {len(df)}  |  Departamentos: {df['departamento'].nunique()}")
    print("=" * 70)
    resumen = (
        df.groupby("departamento")
        .agg(articulos=("cod_articulo", "size"),
             unidades=("unidades", "sum"),
             venta=("total_venta", "sum"))
        .sort_values("venta", ascending=False)
        .round(2)
    )
    print(resumen.to_string())
    print("\nFilas que no son artículo con alta propia:")
    no_art = df[df["tipo_fila"].ne("articulo")]
    if no_art.empty:
        print("   ninguna")
    else:
        for tipo, g in no_art.groupby("tipo_fila"):
            print(f"   {tipo:<15} {len(g):>3} filas  {g['total_venta'].sum():>10.2f} €")

    print()
    print("Nulos por columna:")
    print(df.isna().sum().to_string())

    anomalias = meta["anomalias"]
    print(f"\nAnomalías: {len(anomalias)}")
    for a in anomalias[:15]:
        print("  ", a)

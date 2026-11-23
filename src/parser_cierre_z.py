"""
Parser del cierre Z (informe de arqueo) del TPV. El informe solo existe en PDF:
se extrae el texto con `pypdf` y se lee línea a línea. El PDF solo trae la
fecha de inicio, así que el periodo se toma del nombre del fichero
(`AAAA-MM-DD_AAAA-MM-DD.pdf`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pypdf

# Secciones reconocidas por su línea de cabecera (centrada, sin dos puntos).
SECCIONES = {
    "CAJA",
    "ALBARANES",
    "FACTURAS SIMPLIFICADAS",
    "FACTURAS SIM. RECTIFICATIVAS",
    "RESUMEN VENTAS F. PAGO",
    "RESUMEN CREDITOS",
    "CONCEDIDOS",
    "COBRADOS",
    "DETALLE OPERACIONES",
    "DETALLE DE PAGOS",
    "VENDEDORES",
    "VENTAS POR DEPARTAMENTO",
    "IMPUESTOS",
    "FIDELIZACION",
}

# Campos "clave : valor" que interesan del bloque de totales (antes de la
# primera sección) y de los bloques de documentos (albaranes, facturas...).
CAMPOS_TOTALES = {
    "ALMACEN",
    "FECHA",
    "OPERACIONES",
    "VENTA NETA",
    "VENTA TOTAL",
    "IMPUESTO BOLSAS",
    "TOTAL + BOLSAS",
}
CAMPOS_DOCUMENTO = {"OPERACIONES", "VENTA TOTAL"}

RE_CAMPO = re.compile(r"^([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\.\+ ]*?)\s*:\s*(.+)$")

# Fila de tabla con nombre + 3 columnas numéricas, la última en %.
RE_FILA_DEPARTAMENTO = re.compile(
    r"^(.+?)\s+(\d+)\s+([\-\d,]+\.\d+)\s+([\-\d,]+\.\d+)%$"
)

# Bloque "numeracion de documentos", dos líneas por serie:
#     al:2 T:6  40144/2026TI6 a 41699/2026TI6
#     Doc.: 1052 Imp.: 13,538.56 99.81%
# El rango y el recuento tienen que cuadrar. Si el rango abarca más documentos
# de los que declara haber contado, al informe le faltan ventas: es lo que
# destapó las ventas perdidas, donde el semanal decía "40144 a 41699" (1.556 documentos) pero
# contaba 1.052. Los 504 que faltaban eran dos días enteros de caja que nunca
# llegaron al back-office. Sin esta comprobación, el hueco no da ningún error.
RE_RANGO_DOCUMENTOS = re.compile(
    r"^AL:\d+\s+T:\d+\s+([\d,]+)/(\d{4})([A-Z]+\d*)\s+a\s+([\d,]+)/(\d{4})([A-Z]+\d*)$"
)
RE_RECUENTO_DOCUMENTOS = re.compile(r"^Doc\.:\s*([\d,]+)\s+Imp\.")

# Campos de fidelizacion, de vocabulario fijo y conocido (ver docstring del
# módulo): unos son "cantidad" a secas, otros "cantidad + importe", y los
# últimos cuatro son "clave: valor" normales que ya coge RE_CAMPO.
# Fila del bloque impuestos: tipo + base + cuota + total, alineadas por
# espacios. El TPV imprime el tipo unas veces con espacio ("5 %") y otras sin
# él ("5%"), y con decimales cuando los tiene ("7.5 %"), así que el separador
# es opcional y el número admite punto o coma decimal.
#
#     Tipo            base     cuota     total
#     0 %           2716.52     0.00   2716.52
#     7.5 %          183.17    13.74    196.91
RE_FILA_IMPUESTO = re.compile(
    r"^([\d.,]+)\s*%\s+([\-\d,]+\.\d+)\s+([\-\d,]+\.\d+)\s+([\-\d,]+\.\d+)$"
)

RE_FIDELIZACION_CANTIDAD = re.compile(r"^(CUPONES EMITIDOS)\s+(\d+)$")
RE_FIDELIZACION_CANTIDAD_IMPORTE = re.compile(
    r"^(CUPONES REDIMIDOS|CHEQUES EMITIDOS|CHEQUES REDIMIDOS|DTOS\. EMP\. COVIRAN)"
    r"\s+(\d+)\s+([\-\d,]+\.\d+)$"
)


def _num(valor: str | None) -> float | None:
    """Convierte '1,272.57' o '25.64 %' -> float."""
    if valor is None:
        return None
    valor = valor.replace(",", "").replace("%", "").strip()
    if not valor:
        return None
    try:
        return float(valor)
    except ValueError:
        return None


def _lineas_pdf(ruta: Path) -> list[str]:
    """Extrae todo el texto del PDF como una única lista continua de líneas."""
    lector = pypdf.PdfReader(ruta)
    lineas: list[str] = []
    for pagina in lector.pages:
        texto = pagina.extract_text()
        pagina_lineas = [l.strip() for l in texto.splitlines() if l.strip()]
        if pagina_lineas and re.fullmatch(r"\d+", pagina_lineas[-1]):
            pagina_lineas = pagina_lineas[:-1]
        lineas.extend(pagina_lineas)
    return lineas


def parsear(ruta: str | Path) -> dict:
    """Parsea un cierre Z y devuelve un diccionario con lo relevante para el TFM."""
    ruta = Path(ruta)
    bloques: dict[str, dict[str, str]] = {"_totales": {}}
    filas_departamento: list[dict] = []
    filas_impuesto: list[dict] = []
    fidelizacion: dict[str, float] = {}
    rangos_documentos: list[dict] = []
    rango_pendiente: dict | None = None

    seccion = "_totales"
    for linea in _lineas_pdf(ruta):
        if set(linea) <= {"-"}:
            continue

        # El rango de documentos y su recuento vienen en dos líneas seguidas,
        # antes de la primera sección. Se emparejan con un estado de una línea.
        if m := RE_RANGO_DOCUMENTOS.match(linea):
            rango_pendiente = {
                "serie": m.group(3),
                "anio_inicio": int(m.group(2)),
                "numero_inicio": int(m.group(1).replace(",", "")),
                "anio_fin": int(m.group(5)),
                "numero_fin": int(m.group(4).replace(",", "")),
            }
            continue
        if rango_pendiente is not None:
            if m := RE_RECUENTO_DOCUMENTOS.match(linea):
                rango_pendiente["documentos"] = int(m.group(1).replace(",", ""))
                rangos_documentos.append(rango_pendiente)
            rango_pendiente = None

        if linea in SECCIONES:
            seccion = linea
            continue

        if seccion == "VENTAS POR DEPARTAMENTO":
            if m := RE_FILA_DEPARTAMENTO.match(linea):
                filas_departamento.append({
                    "departamento": m.group(1).strip(),
                    "articulos": _num(m.group(2)),
                    "importe": _num(m.group(3)),
                    "pct_total": _num(m.group(4)),
                })
            continue

        if seccion == "IMPUESTOS":
            if m := RE_FILA_IMPUESTO.match(linea):
                filas_impuesto.append({
                    "tipo": float(m.group(1).replace(",", ".")),
                    "base": _num(m.group(2)),
                    "cuota": _num(m.group(3)),
                    "total": _num(m.group(4)),
                })
            continue

        if seccion == "FIDELIZACION":
            if m := RE_FIDELIZACION_CANTIDAD.match(linea):
                fidelizacion["cupones_emitidos"] = _num(m.group(2))
                continue
            if m := RE_FIDELIZACION_CANTIDAD_IMPORTE.match(linea):
                clave = {
                    "CUPONES REDIMIDOS": "cupones_redimidos",
                    "CHEQUES EMITIDOS": "cheques_emitidos",
                    "CHEQUES REDIMIDOS": "cheques_redimidos",
                    "DTOS. EMP. COVIRAN": "dtos_emp_coviran",
                }[m.group(1)]
                fidelizacion[f"{clave}_cantidad"] = _num(m.group(2))
                fidelizacion[f"{clave}_importe"] = _num(m.group(3))
                continue
            if m := RE_CAMPO.match(linea):
                clave = {
                    "PORCENTAJE TICK FIDELIZADOS": "pct_tickets_fidelizados",
                    "PORCENTAJE VENTA FIDELIZADA": "pct_venta_fidelizada",
                    "TICKET MEDIO FIDELIZADO": "ticket_medio_fidelizado",
                    "TICKET MEDIO NO FIDELIZADO": "ticket_medio_no_fidelizado",
                }.get(m.group(1).strip())
                if clave:
                    fidelizacion[clave] = _num(m.group(2))
            continue

        if seccion in ("_totales", "ALBARANES", "FACTURAS SIMPLIFICADAS",
                       "FACTURAS SIM. RECTIFICATIVAS"):
            if m := RE_CAMPO.match(linea):
                etiqueta, valor = m.group(1).strip(), m.group(2).strip()
                campos_validos = (
                    CAMPOS_TOTALES if seccion == "_totales" else CAMPOS_DOCUMENTO
                )
                if etiqueta in campos_validos:
                    bloques.setdefault(seccion, {})[etiqueta] = valor
            continue

        # Resto de secciones (caja, resumen ventas F. pago, resumen creditos,
        # detalle operaciones, detalle de pagos, vendedores): reconocidas para
        # no perder la cuenta de en qué sección se está, pero su contenido no
        # se extrae de momento (ver docstring).

    totales = bloques["_totales"]
    albaranes = bloques.get("ALBARANES", {})
    facturas = bloques.get("FACTURAS SIMPLIFICADAS", {})
    rectificativas = bloques.get("FACTURAS SIM. RECTIFICATIVAS", {})

    departamentos = pd.DataFrame(
        filas_departamento,
        columns=["departamento", "articulos", "importe", "pct_total"],
    )
    impuestos = pd.DataFrame(
        filas_impuesto, columns=["tipo", "base", "cuota", "total"]
    )

    return {
        "fichero": ruta.name,
        "almacen": totales.get("ALMACEN"),
        "fecha": totales.get("FECHA"),
        "operaciones": _num(totales.get("OPERACIONES")),
        "venta_neta": _num(totales.get("VENTA NETA")),
        "venta_total": _num(totales.get("VENTA TOTAL")),
        "albaranes_operaciones": _num(albaranes.get("OPERACIONES")),
        "albaranes_venta_total": _num(albaranes.get("VENTA TOTAL")),
        "tickets": _num(facturas.get("OPERACIONES")),
        "tickets_venta_total": _num(facturas.get("VENTA TOTAL")),
        "rectificativas_operaciones": _num(rectificativas.get("OPERACIONES")),
        "rectificativas_venta_total": _num(rectificativas.get("VENTA TOTAL")),
        "departamentos": departamentos,
        "impuestos": impuestos,
        "fidelizacion": fidelizacion,
        "rangos_documentos": rangos_documentos,
    }


def validar(datos: dict) -> dict:
    """Cruza los totales declarados contra la suma de sus propios desgloses."""
    resultado = {}

    ops_declarado = datos["operaciones"]
    ops_calculado = sum(
        v for v in (
            datos["albaranes_operaciones"],
            datos["tickets"],
            datos["rectificativas_operaciones"],
        ) if v is not None
    )
    resultado["operaciones"] = {
        "declarado": ops_declarado,
        "calculado": ops_calculado,
        "diferencia": None if ops_declarado is None else round(ops_calculado - ops_declarado, 2),
    }

    venta_declarada = datos["venta_total"]
    venta_calculada = sum(
        v for v in (
            datos["albaranes_venta_total"],
            datos["tickets_venta_total"],
            datos["rectificativas_venta_total"],
        ) if v is not None
    )
    resultado["venta_total_documentos"] = {
        "declarado": venta_declarada,
        "calculado": round(venta_calculada, 2),
        "diferencia": None if venta_declarada is None else round(venta_calculada - venta_declarada, 2),
    }

    # Continuidad de la numeracion: el rango declarado tiene que abarcar
    # exactamente los documentos contados.
    huecos = []
    abarcados_total = contados_total = series = 0
    for r in datos.get("rangos_documentos", []):
        if r["anio_inicio"] != r["anio_fin"] or "documentos" not in r:
            continue
        abarcados = r["numero_fin"] - r["numero_inicio"] + 1
        faltan = abarcados - r["documentos"]
        series += 1
        abarcados_total += abarcados
        contados_total += r["documentos"]
        if faltan:
            huecos.append({**r, "abarcados": abarcados, "faltan": faltan})
    resultado["continuidad_documentos"] = {
        "series_comprobadas": series,
        # Los dos totales se guardan aunque no haya hueco: son lo que permite
        # decir "el informe abarca N numeros y cuenta M" en las 135 semanas, y
        # no solo en las que fallan.
        "documentos_abarcados": abarcados_total,
        "documentos_contados": contados_total,
        "documentos_que_faltan": sum(h["faltan"] for h in huecos),
        "huecos": huecos,
    }

    venta_departamentos = float(datos["departamentos"]["importe"].sum())
    resultado["venta_total_departamentos"] = {
        "declarado": venta_declarada,
        "calculado": round(venta_departamentos, 2),
        "diferencia": None if venta_declarada is None else round(venta_departamentos - venta_declarada, 2),
    }

    # La suma de bases imponibles tiene que reproducir la venta neta.
    impuestos = datos.get("impuestos")
    neta_declarada = datos["venta_neta"]
    base_impuestos = (
        float(impuestos["base"].sum()) if impuestos is not None and not impuestos.empty else None
    )
    resultado["venta_neta_impuestos"] = {
        "declarado": neta_declarada,
        "calculado": None if base_impuestos is None else round(base_impuestos, 2),
        "diferencia": (
            None if neta_declarada is None or base_impuestos is None
            else round(base_impuestos - neta_declarada, 2)
        ),
        "tipos": [] if impuestos is None else sorted(impuestos["tipo"].tolist()),
    }

    return resultado


if __name__ == "__main__":
    import sys

    ruta = (
        sys.argv[1]
        if len(sys.argv) > 1
        else r"datos_bruto\Informes Z semanal\2026-07-06_2026-07-12.pdf"
    )
    datos = parsear(ruta)

    print("=" * 70)
    print("Datos")
    print("=" * 70)
    for k, v in datos.items():
        if k in ("departamentos", "impuestos"):
            continue
        print(f"  {k}: {v}")

    print("\nDepartamentos:")
    print(datos["departamentos"].to_string(index=False))

    print("\nImpuestos:")
    print(datos["impuestos"].to_string(index=False))

    print("\n" + "=" * 70)
    print("Validación")
    print("=" * 70)
    for campo, r in validar(datos).items():
        print(f"  {campo}: {r}")

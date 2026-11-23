"""
Descarga el IPC del INE y genera datos_externos/ipc_ine.csv.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "datos_externos" / "ipc_ine.csv"

API = "https://servicios.ine.es/wstempus/js/ES/DATOS_SERIE/{cod}?nult={nult}"

# Meses de histórico. 66 cubre de 2021 a hoy con margen.
NULT = 66

# Año base: la media de este año vale 100. El INE rebasó a 2025 en las tablas
# "ver.2". Si vuelve a rebasar, actualizar aquí; la comprobación final avisará.
BASE = 2025

# (código INE, ámbito, nivel, slug)
# El nombre largo lo pone el INE y se guarda tal cual en `serie_nombre`.
SERIES: list[tuple[str, str, str, str]] = [
    # --- 1. Almería, provincial. El deflactor principal -------------------
    ("IPC305912", "almeria", "general", "general"),
    # Nivel grupo, el mismo concepto que la serie nacional: permite comparar
    # Almería con España en la cifra que suele citar la prensa.
    ("IPC305916", "almeria", "grupo", "alimentos_bebidas_no_alc"),
    ("IPC309268", "almeria", "subgrupo", "alimentos"),
    ("IPC309272", "almeria", "subgrupo", "bebidas_no_alcoholicas"),
    ("IPC309276", "almeria", "subgrupo", "bebidas_alcoholicas"),
    ("IPC309308", "almeria", "subgrupo", "mantenimiento_hogar"),
    ("IPC309328", "almeria", "subgrupo", "cuidado_personal"),
    ("IPC309300", "almeria", "subgrupo", "menaje_hogar"),
    # --- 2. Nacional. Para comparar ---------------------------------------
    ("IPC290751", "nacional", "general", "general"),
    ("IPC290755", "nacional", "grupo", "alimentos_bebidas_no_alc"),
    # --- 3. Andalucía. Los agregados sirven para validar que usar rúbricas
    #        de Andalucía como proxy de Almería es razonable: si el agregado
    #        andaluz se mueve como el almeriense, el supuesto se sostiene.
    ("IPC293660", "andalucia", "general", "general"),
    ("IPC293664", "andalucia", "grupo", "alimentos_bebidas_no_alc"),
    # --- ...y las rúbricas, el detalle para mapear departamentos ----------
    ("IPC298632", "andalucia", "rubrica", "cereales_derivados"),
    ("IPC298636", "andalucia", "rubrica", "pan"),
    ("IPC298640", "andalucia", "rubrica", "carne_vacuno"),
    ("IPC298644", "andalucia", "rubrica", "carne_ovino"),
    ("IPC298648", "andalucia", "rubrica", "carne_porcino"),
    ("IPC298652", "andalucia", "rubrica", "carne_ave"),
    ("IPC298656", "andalucia", "rubrica", "otras_carnes"),
    ("IPC298780", "andalucia", "rubrica", "pescado_fresco_congelado"),
    ("IPC298784", "andalucia", "rubrica", "crustaceos_preparados_pescado"),
    ("IPC298660", "andalucia", "rubrica", "huevos"),
    ("IPC298664", "andalucia", "rubrica", "leche"),
    ("IPC298668", "andalucia", "rubrica", "productos_lacteos"),
    ("IPC298672", "andalucia", "rubrica", "aceites_grasas"),
    ("IPC298676", "andalucia", "rubrica", "frutas_frescas"),
    ("IPC298788", "andalucia", "rubrica", "frutas_conserva_frutos_secos"),
    ("IPC298680", "andalucia", "rubrica", "legumbres_hortalizas_frescas"),
    ("IPC298684", "andalucia", "rubrica", "preparados_legumbres_hortalizas"),
    ("IPC298792", "andalucia", "rubrica", "patatas_preparados"),
    ("IPC298688", "andalucia", "rubrica", "azucar"),
    ("IPC298692", "andalucia", "rubrica", "cafe_cacao_infusiones"),
    ("IPC298696", "andalucia", "rubrica", "agua_refrescos_zumos"),
    ("IPC298800", "andalucia", "rubrica", "bebidas_alcoholicas"),
]


def descargar(cod: str) -> dict:
    with urllib.request.urlopen(API.format(cod=cod, nult=NULT), timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    filas = []
    for cod, ambito, nivel, slug in SERIES:
        serie = descargar(cod)
        nombre = serie["Nombre"].strip()
        n = 0
        for v in serie["Data"]:
            mes = v["FK_Periodo"]
            if not 1 <= mes <= 12:  # descarta medias anuales si aparecieran
                continue
            n += 1
            filas.append(
                {
                    "anio": v["Anyo"],
                    "mes": mes,
                    "periodo": f"{v['Anyo']}-{mes:02d}",
                    "ambito": ambito,
                    "nivel": nivel,
                    "concepto": slug,
                    "indice": v["Valor"],
                    "serie_ine": cod,
                    "serie_nombre": nombre,
                }
            )
        print(f"  {cod}  {ambito:9s} {nivel:8s} {n:3d} obs.  {nombre[:52]}")

    df = pd.DataFrame(filas).sort_values(["ambito", "nivel", "concepto", "anio", "mes"])

    # Variación anual calculada por nosotros desde el índice, para no depender de
    # las series de variación del INE y poder recomputarla siempre igual.
    df["var_anual_pct"] = (
        df.groupby(["ambito", "concepto"])["indice"].pct_change(periods=12) * 100
    ).round(2)

    df.to_csv(SALIDA, index=False, encoding="utf-8", float_format="%.3f")

    print()
    print(f"Escrito {SALIDA.relative_to(RAIZ)}: {len(df)} filas, "
          f"{len(SERIES)} series, {df.periodo.min()} a {df.periodo.max()}")
    print()
    print(f"Integridad: la media de {BASE} debe ser 100.000 en todas las series.")
    med = df[df.anio == BASE].groupby(["ambito", "concepto"]).indice.mean().round(3)
    malas = med[(med - 100).abs() > 0.001]
    if len(malas):
        print("  *** Revisar, no cuadran: ***")
        print(malas.to_string())
    else:
        print(f"  OK en las {len(med)} series.")


if __name__ == "__main__":
    main()

"""
Clasifica las filas artículo×mes que el informe da por debajo de coste.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import comprobar_cifras as cc

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "docs" / "articulos_bajo_coste.csv"

# Salto de `tarifa_costo`, en absoluto, sobre la mediana de los meses sanos del
# propio artículo, que delata un coste realmente anómalo (pack o caja cruzado
# con la unidad).
UMBRAL_PACK = 1.8

# Artículos que la tienda confirma que compra por caja y vende a granel al
# kilo, con el factor caja→kilo.
CAJA_A_GRANEL = {
    "54933": 2,   # Magdalena aceite oliva - caja de 2 kg
    "55114": 2,   # Magdalena integral KG - caja de 2 kg (1 mes en la ventana, ya sano)
}

# Artículos donde el mismo cruce (coste de un lote contra precio de una unidad)
# se ve en los propios datos del artículo, sin hacer falta preguntar a la
# tienda: sus meses sanos alternan entre "vendido suelto" y "vendido en pack",
# y en los dos casos precio y tarifa se mueven juntos y mantienen la misma
# proporción.
PACK_INFERIDO = {
    "35061": 6,   # Agua con gas fonter 50 - pack de 6 botellas de 50 cl
}

# Artículos que la tienda confirma que se venden sueltos y en pack, pero sin un
# factor que se pueda medir: sus meses sanos no mantienen una proporción
# estable entre tarifa y precio.
PACK_CONFIRMADO = {
    "11479",      # Agua junior font vella
}


def _corregir_por_factor(base: pd.DataFrame, factores: dict[str, int]) -> pd.DataFrame:
    """Divide `tarifa_costo` por el factor de cada artículo de `factores`, y
    solo en las filas donde el informe cruzó el precio de la unidad con el
    coste del lote (las de margen negativo).
    """
    b = base.copy()
    b["cod_articulo"] = b["cod_articulo"].astype(str)
    for cod, factor in factores.items():
        m = (b["cod_articulo"] == cod) & (b["beneficio_tc"] < 0)
        if not m.any():
            continue
        b.loc[m, "tarifa_costo"] = b.loc[m, "tarifa_costo"] / factor
        b.loc[m, "beneficio_tc"] = (b.loc[m, "total_venta"]
                                    - b.loc[m, "unidades"] * b.loc[m, "tarifa_costo"])
        b.loc[m, "pct_rent_tc"] = 100 * b.loc[m, "beneficio_tc"] / b.loc[m, "total_venta"]
    return b


def corregir_caja_a_granel(base: pd.DataFrame) -> pd.DataFrame:
    """Aplica la corrección de `CAJA_A_GRANEL` (confirmada por la tienda) y de
    `PACK_INFERIDO` (derivada de los propios meses sanos del artículo).
    """
    return _corregir_por_factor(_corregir_por_factor(base, CAJA_A_GRANEL), PACK_INFERIDO)


def clasificar() -> pd.DataFrame:
    """Devuelve las filas artículo×mes en pérdida, con su causa asignada."""
    base = cc._base_margen(cc.Contexto()).copy()
    base["cod_articulo"] = base["cod_articulo"].astype(str)

    # La referencia de cada artículo es su propia tarifa en los meses sanos, no
    # la del catálogo: lo que importa es que el artículo se salga de su propia
    # norma, no de la de los demás.
    sanos = base[base["beneficio_tc"] >= 0]
    ref = sanos.groupby("cod_articulo").agg(tarifa_ref=("tarifa_costo", "median"))

    p = base[base["beneficio_tc"] < 0].merge(ref, on="cod_articulo", how="left")

    tarifa_salto = p["tarifa_costo"] / p["tarifa_ref"]
    p["tarifa_salto"] = tarifa_salto

    # El coste realmente pagado queda por debajo del precio: la venta es
    # rentable y la tarifa es la que está mal.
    coste_ficha = (p["costo_medio"] > 0) & (p["costo_medio"] < p["precio"])

    # `CAJA_A_GRANEL` y `PACK_INFERIDO` entran aquí a la fuerza: da igual si el
    # salto numérico de esa fila concreta llega a 1,8× o no, porque la prueba
    # de esos dos diccionarios (la tienda, o los propios meses sanos del
    # artículo) es más fuerte que el umbral genérico.
    forzados = (p["cod_articulo"].isin(CAJA_A_GRANEL) | p["cod_articulo"].isin(PACK_INFERIDO)
                | p["cod_articulo"].isin(PACK_CONFIRMADO))
    pack = ((tarifa_salto >= UMBRAL_PACK) & ~coste_ficha) | forzados

    promocion = p["descripcion"].str.startswith("(P)") | p["oferta"]

    p["causa"] = np.select(
        [pack, coste_ficha, promocion],
        ["pack_unidad", "coste_ficha", "promocion"],
        default="sin_firma",
    )
    return p


def por_articulo(p: pd.DataFrame) -> pd.DataFrame:
    """Una fila por artículo, con la causa de su mes de mayor pérdida."""
    peor = p.loc[p.groupby("cod_articulo")["beneficio_tc"].idxmin()]
    peor = peor.set_index("cod_articulo")[["descripcion", "departamento", "causa"]]

    agg = p.groupby("cod_articulo").agg(
        meses_en_perdida=("periodo", "size"),
        primer_mes=("periodo", "min"),
        ultimo_mes=("periodo", "max"),
        venta=("total_venta", "sum"),
        perdida=("beneficio_tc", "sum"),
    )
    art = peor.join(agg).reset_index()
    art["venta"] = art["venta"].round(2)
    art["perdida"] = art["perdida"].round(2)
    return art.sort_values("perdida").reset_index(drop=True)


# Los dos defectos de la columna de coste, que son los que la memoria cuenta
# aparte de las promociones y del residuo.
DEFECTOS_DE_COSTE = ["coste_ficha", "pack_unidad"]


def fuerza_de_la_prueba(p: pd.DataFrame) -> pd.DataFrame:
    """Reparte la pérdida de los dos defectos según lo demostrable que sea."""
    d = p[p["causa"].isin(DEFECTOS_DE_COSTE)].copy()

    prop = (d["tarifa_costo"] / d["precio"]).round(2)
    veces = d.assign(prop=prop).groupby(["cod_articulo", "prop"])["periodo"].transform("size")
    ficha = d["causa"] == "coste_ficha"
    forzados = (d["cod_articulo"].isin(CAJA_A_GRANEL) | d["cod_articulo"].isin(PACK_INFERIDO)
                | d["cod_articulo"].isin(PACK_CONFIRMADO))

    d["nivel"] = np.select(
        [(ficha & (veces >= 3))
         | (~ficha & ((d["tarifa_salto"] >= 3) | d["tarifa_salto"].isna() | forzados)),
         (ficha & (veces == 2)) | (~ficha & (d["tarifa_salto"] >= 2))],
        ["solido", "intermedio"], default="flojo")

    t = d.groupby("nivel").agg(filas=("periodo", "size"), perdida=("beneficio_tc", "sum"))
    t["pct"] = 100 * t["perdida"] / d["beneficio_tc"].sum()
    return t.reindex(["solido", "intermedio", "flojo"])


def pruebas_sin_costo_medio(p: pd.DataFrame, meses: int = 3) -> pd.DataFrame:
    """Las filas `coste_ficha`, con las dos pruebas que no usan `costo_medio`."""
    base = cc._base_margen(cc.Contexto()).copy()
    base["cod_articulo"] = base["cod_articulo"].astype(str)
    tarifa = (base.assign(m=pd.PeriodIndex(base["periodo"], freq="M"))
                  .groupby(["cod_articulo", "m"])["tarifa_costo"].first())
    cf = p[p["causa"] == "coste_ficha"].copy()
    m = pd.PeriodIndex(cf["periodo"], freq="M")

    def al_dia(i) -> bool:
        cod, mes, precio = cf.at[i, "cod_articulo"], m[cf.index.get_loc(i)], cf.at[i, "precio"]
        for k in range(1, meses + 1):
            t = tarifa.get((cod, mes + k))
            if t is not None and 0 < t <= precio:
                return True
        return False

    cf["prueba_coste_ultimo"] = (cf["costo_ultimo"] > 0) & (cf["costo_ultimo"] < cf["precio"])
    cf["prueba_tarifa_al_dia"] = [al_dia(i) for i in cf.index]
    cf["alguna_prueba"] = cf["prueba_coste_ultimo"] | cf["prueba_tarifa_al_dia"]
    return cf


def main() -> None:
    p = clasificar()
    art = por_articulo(p)
    art.to_csv(SALIDA, index=False, encoding="utf-8")

    filas = p.groupby("causa").agg(
        filas=("periodo", "size"), venta=("total_venta", "sum"),
        perdida=("beneficio_tc", "sum"))
    arts = art.groupby("causa").size().rename("articulos")
    tabla = filas.join(arts).sort_values("perdida")
    tabla["pct_perdida"] = (100 * tabla["perdida"] / p["beneficio_tc"].sum()).round(1)

    print(tabla.round(2).to_string())
    print("")
    print("Fuerza de la prueba en los dos defectos de coste:")
    print(fuerza_de_la_prueba(p).round(2).to_string())
    print(f"\nTotal: {len(p)} filas | {art.shape[0]} articulos | "
          f"{p['total_venta'].sum():.2f} EUR de venta | {p['beneficio_tc'].sum():.2f} EUR")
    print(f"Escrito: {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()

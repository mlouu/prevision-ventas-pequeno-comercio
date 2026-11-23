"""
Recalcula desde `datos_limpio/` las cifras clave de la memoria y avisa si alguna
deja de cuadrar (sale con código 1).

    python src/comprobar_cifras.py
    python src/comprobar_cifras.py --con-modelos   # reentrena los modelos, tarda minutos
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


LIMPIO = RAIZ / "datos_limpio"


BASE_COSTE = "tarifa_costo"


BOLSAS_CAJA = ["121601", "121602"]


FRIOS_BEBIDA_MARGEN = ["84887", "84891", "84888", "84890", "84886", "84889", "85695"]


FRIOS_SUPLEMENTO_MARGEN = ["80006", "80768"]


FRIOS_TODOS_MARGEN = FRIOS_BEBIDA_MARGEN + FRIOS_SUPLEMENTO_MARGEN


MESES_ENE_JUL = [1, 2, 3, 4, 5, 6, 7]


@dataclass
class Contexto:
    """Carga perezosa de los datos limpios."""

    _cache: dict = field(default_factory=dict)

    def _leer(self, nombre: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        if nombre not in self._cache:
            self._cache[nombre] = fn()
        return self._cache[nombre]

    @property
    def serie(self) -> pd.DataFrame:
        def cargar():
            s = pd.read_parquet(LIMPIO / "serie_semanal.parquet")
            s["periodo_inicio"] = pd.to_datetime(s["periodo_inicio"])
            return s.sort_values("periodo_inicio").reset_index(drop=True)
        return self._leer("serie", cargar)

    @property
    def completa(self) -> pd.DataFrame:
        """Solo las semanas naturales de 7 días."""
        return self.serie[self.serie["semana_completa"]].reset_index(drop=True)

    @property
    def consolidado(self) -> pd.DataFrame:
        return self._leer(
            "consolidado", lambda: pd.read_parquet(LIMPIO / "consolidado_mensual.parquet")
        )

    @property
    def articulos(self) -> pd.DataFrame:
        """Filas de artículo con nombre propio, que son la base del surtido."""
        def cargar():
            c = self.consolidado
            a = c[c["tipo_fila"] == "articulo"].copy()
            a["periodo"] = a["periodo_inicio"].dt.strftime("%Y-%m")
            return a
        return self._leer("articulos", cargar)


@dataclass
class Cifra:
    """Una cifra que la memoria afirma, con su cálculo y dónde está escrita."""

    clave: str
    escrito: float
    tolerancia: float
    donde: str
    calcular: Callable[[Contexto], float]
    grupo: str = "serie"
    unidad: str = ""


def _tendencia_pct(dato: np.ndarray) -> tuple[float, float]:
    """Tendencia en % sobre la media, con la fórmula del notebook 03 §5."""
    import statsmodels.api as sm

    x = sm.add_constant(np.arange(len(dato), dtype=float))
    aj = sm.OLS(dato, x).fit()
    return 100 * aj.params[1] * len(dato) / dato.mean(), aj.pvalues[1]


def _crecimiento(ctx: Contexto, col: str, a: int, b: int) -> float:
    d = ctx.completa
    d = d[d["periodo_inicio"].dt.month.isin(MESES_ENE_JUL)]
    xa = d.loc[d["periodo_inicio"].dt.year == a, col].mean()
    xb = d.loc[d["periodo_inicio"].dt.year == b, col].mean()
    return 100 * (xb / xa - 1)


def _crecimiento_log(ctx: Contexto, col: str, a: int, b: int) -> float:
    """Lo mismo que `_crecimiento`, en diferencias logarítmicas (Δln, en puntos)."""
    d = ctx.completa
    d = d[d["periodo_inicio"].dt.month.isin(MESES_ENE_JUL)]
    xa = d.loc[d["periodo_inicio"].dt.year == a, col].mean()
    xb = d.loc[d["periodo_inicio"].dt.year == b, col].mean()
    return 100 * np.log(xb / xa)


def _escalon_fiscal(ctx: Contexto) -> float:
    """Cuánto encarece la ley la misma cesta, en el tramo de 2024."""
    s = ctx.completa
    t = s[s["periodo_inicio"] < "2024-09-30"]
    return float((100 * ((1 + t["tipo_iva_referencia"]) / (1 + t["tipo_iva_efectivo"]) - 1)).mean())


def _base_margen(ctx: Contexto) -> pd.DataFrame:
    """La base de margen del notebook 02 (`CON_COSTE`), reproducida tal cual."""
    a = ctx.articulos
    roto_a15 = (a["precio"] > 0) & (a[BASE_COSTE] > 10 * a["precio"])
    return a[(a[BASE_COSTE] > 0) & (~roto_a15) & (a["total_venta"] > 0)
             & (~a["cod_articulo"].isin(BOLSAS_CAJA))]


def _margen(base: pd.DataFrame) -> float:
    """Margen en euros total / venta total, ponderado por venta."""
    return 100 * base["beneficio_tc"].sum() / base["total_venta"].sum()


def _surtido(ctx: Contexto) -> pd.DataFrame:
    """`SURTIDO`, la base de margen real del notebook 02 (§3), reproducida tal cual."""
    import clasificar_bajo_coste as cbc  # import diferido: ese módulo importa este

    pre = _base_margen(ctx)
    pre = pre[~pre["cod_articulo"].isin(FRIOS_TODOS_MARGEN)]
    corregido = cbc.corregir_caja_a_granel(pre)  # aplica también PACK_INFERIDO

    forzados = set(cbc.CAJA_A_GRANEL) | set(cbc.PACK_INFERIDO)
    bc = cbc.clasificar()
    bc["cod_articulo"] = bc["cod_articulo"].astype(str)
    defec = bc[bc["causa"].isin(cbc.DEFECTOS_DE_COSTE)
                & (~bc["cod_articulo"].isin(forzados))]
    claves_error = set(zip(defec["cod_articulo"], defec["periodo"]))
    clave = list(zip(corregido["cod_articulo"].astype(str), corregido["periodo"]))
    es_error = pd.Series([k in claves_error for k in clave], index=corregido.index)
    return corregido[~es_error]


# --- La nevera de bebidas frias ---
FRIOS_BEBIDA = ["84887", "84891", "84888", "84890", "84886", "84889", "85695"]


BANDA_TRAMO_FRIO = 0.30


POOL_FRIO = [
    ("84887", ["1623", "428016", "12783", "7460", "428012", "428030", "428017"]),
    ("84891", ["431002", "1988", "11206", "36026", "431018", "70697", "37723"]),
    ("84890", ["431003"]),
    ("84888", ["431001", "1936"]),
    ("84889", ["428033", "10526", "15987", "11394", "18730", "136212", "118551", "1440"]),
]


def _frio(ctx: Contexto) -> pd.DataFrame:
    a = ctx.articulos
    return a[a["cod_articulo"].isin(FRIOS_BEBIDA)]


def _frio_margenes(ctx: Contexto, banda: float = BANDA_TRAMO_FRIO) -> tuple[float, float, float]:
    """Margen del frio, del mismo envase y tramo en el lineal, y margen extra."""
    p = ctx.articulos.groupby("cod_articulo").agg(
        u=("unidades", "sum"), v=("total_venta", "sum"), c=(BASE_COSTE, "median"))
    uf = vf = ci = vl = cl = ul = 0.0
    for cod_frio, pool in POOL_FRIO:
        f = p.loc[cod_frio]
        L = p.loc[[c for c in pool if c in p.index]]
        sel = L[(L["c"] >= f["c"] * (1 - banda)) & (L["c"] <= f["c"] * (1 + banda))]
        if sel.empty:
            continue
        c_lineal = (sel["c"] * sel["u"]).sum() / sel["u"].sum()
        uf += f["u"]; vf += f["v"]; ci += c_lineal * f["u"]
        ul += sel["u"].sum(); vl += sel["v"].sum(); cl += (sel["c"] * sel["u"]).sum()
    return (100 * (vf - ci) / vf, 100 * (vl - cl) / vl,
            (vf - ci) - uf * (vl - cl) / ul)


def _frio_cuota_mes(ctx: Contexto, mes: int) -> float:
    """Peso de la bebida fria sobre la venta de articulos, en un mes del ano."""
    a = ctx.articulos
    f = _frio(ctx)
    num_ = f.loc[f["periodo_inicio"].dt.month == mes, "total_venta"].sum()
    den = a.loc[a["periodo_inicio"].dt.month == mes, "total_venta"].sum()
    return 100 * num_ / den


def _margen_global(ctx: Contexto) -> float:
    return _margen(_surtido(ctx))


def _concentracion_50(ctx: Contexto) -> float:
    """% de artículos que acumula la mitad de la venta."""
    v = ctx.articulos.groupby("cod_articulo")["total_venta"].sum().sort_values(ascending=False)
    acum = 100 * v.cumsum() / v.sum()
    return 100 * (int((acum <= 50).sum()) + 1) / len(v)


def _fijos(ctx: Contexto) -> tuple[int, float, float]:
    """Artículos fijos, presentes en los 31 meses."""
    a = ctx.articulos
    meses = a.groupby("cod_articulo")["periodo"].nunique()
    n_meses = a["periodo"].nunique()
    fijos = meses[meses == n_meses].index
    v = a.groupby("cod_articulo")["total_venta"].sum()
    return len(fijos), 100 * len(fijos) / len(meses), 100 * v[fijos].sum() / v.sum()


def _concentracion(ctx: Contexto, pct: float) -> float:
    """% de artículos que acumula `pct` % de la venta, agrupando por código."""
    v = ctx.articulos.groupby("cod_articulo")["total_venta"].sum().sort_values(ascending=False)
    acum = 100 * v.cumsum() / v.sum()
    return 100 * (int((acum < pct).sum()) + 1) / len(v)


def _lentos_pct(ctx: Contexto) -> float:
    r = []
    for per, g in ctx.articulos.groupby("periodo_inicio"):
        u = g.groupby("cod_articulo")["unidades"].sum()
        r.append(100 * ((u / (per.days_in_month / 7)) < 1).mean())
    return float(np.mean(r))


def _coviran(ctx: Contexto, grupo: str, col: str) -> float:
    import agrupacion as ag
    r = ag.resumen_recomendacion_coviran(ag.clasificar_basicos(_surtido(ctx)))
    fila = "básicos (leche, aceite)" if grupo == "basicos" else "resto del surtido"
    return float(r.loc[fila, col])


def _supervivencia_surtido(ctx: Contexto) -> float:
    """% de los artículos vendidos en 2024 que siguen vendiéndose en 2026."""
    a = ctx.articulos
    anio = a["periodo_inicio"].dt.year
    c24 = set(a.loc[anio == 2024, "cod_articulo"])
    c26 = set(a.loc[anio == 2026, "cod_articulo"])
    return 100 * len(c24 & c26) / len(c24)


def _bc(ctx: Contexto) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Las filas bajo coste de los artículos bajo coste y su reparto por fuerza de la prueba."""
    import clasificar_bajo_coste as cbc
    p = cbc.clasificar()
    return p, cbc.fuerza_de_la_prueba(p)


# --- La agrupación por rentabilidad ---
@lru_cache(maxsize=1)
def _rentab() -> dict:
    """La partición de la agrupación y lo que se deriva de ella, calculada una sola vez."""
    import agrupacion as ag  # import diferido: ese módulo importa este

    largo, meta = ag.cargar_fijos()
    r = ag.cargar_rentabilidad(meta.index)
    Z = ag.escalar(ag.matriz_rentabilidad(r))
    enlace = ag.enlace_jerarquico(Z)
    et = ag.ordenar_por_margen_euros(ag.jerarquico(Z, 4)[0], r)
    km = ag.emparejar(et, ag.agrupar(Z, 4)[0])
    sur = ag.surtido_de(r.index)
    cong = ag.tarifa_congelada(largo).reindex(r.index, fill_value=False)
    grupo0 = r.loc[et.index[et == 0]]
    return {
        "r": r,
        "et": et,
        "sur": sur,
        "cong": cong,
        "Z": Z,
        "perfil": ag.perfilar_rentabilidad(et, r, meta),
        "saltos": ag.saltos_de_union(enlace, 6),
        "ac": ag.coeficiente_aglomeracion(enlace),
        "coinc_km": 100 * ag.coincidencia(et, km),
        "km_se_queda": 100 * (et == km).groupby(et).mean(),
        "k2": ag.perfilar_rentabilidad(
            ag.ordenar_por_margen_euros(ag.jerarquico(Z, 2)[0], r), r, meta),
        "coinc_km_2": 100 * ag.coincidencia(
            ag.ordenar_por_margen_euros(ag.jerarquico(Z, 2)[0], r),
            ag.emparejar(ag.ordenar_por_margen_euros(ag.jerarquico(Z, 2)[0], r),
                         ag.agrupar(Z, 2)[0])),
        "evol": ag.margen_por_anio(sur, et),
        "evol_sin": ag.margen_por_anio(sur[~sur["cod"].map(cong).astype(bool)], et),
        "u12": ag.ultimos_12_meses(sur, et),
        "top10": 100 * grupo0.nlargest(10, "margen_euros")["margen_euros"].sum()
                 / r["margen_euros"].sum(),
        "congeladas": int(cong.sum()),
        "congeladas_g0": int(cong[et.index[et == 0]].sum()),
    }


CIFRAS: list[Cifra] = [
    # --- la serie ---
    Cifra("semanas", 135, 0, "I.3 · metodología · «135 semanas naturales»",
          lambda c: len(c.completa), unidad="semanas"),
    Cifra("venta_media", 16_038.89, 0.5, "I.1 · situación · «del orden de 16.000 € semanales»",
          lambda c: c.completa["venta_neta"].mean(), unidad="€"),
    Cifra("tickets_medios", 1_436.40, 0.5, "I.1 · situación · «unos 1.436 tickets»",
          lambda c: c.completa["tickets"].mean(), unidad="tickets"),
    Cifra("ticket_medio", 11.19, 0.005, "I.1 · situación · «ticket medio de 11,19 €»",
          lambda c: (c.completa["venta_neta"] / c.completa["tickets"]).mean(), unidad="€"),
    Cifra("cv_semanal", 8.8, 0.05, "I.5.D · «8,8 % el total» frente a los departamentos",
          lambda c: 100 * c.completa["venta_neta"].std() / c.completa["venta_neta"].mean(),
          unidad="%"),
    Cifra("lag52", 0.365, 0.0005, "la estacionalidad anual · autocorrelación anual",
          lambda c: c.completa["venta_neta"].autocorr(52)),

    # --- crecimiento y deflactado ---
    Cifra("crec_nominal", 3.56, 0.005, "I.5.A y II.5 · crecimiento nominal ene-jul 2025→2026",
          lambda c: _crecimiento(c, "venta_neta", 2025, 2026), unidad="%"),
    Cifra("crec_real", 1.80, 0.005, "I.5.A y II.5 · crecimiento real en volumen (Δln)",
          lambda c: _crecimiento_log(c, "venta_neta_real", 2025, 2026), unidad="pp"),
    Cifra("escalon_fiscal", 1.31, 0.005, "I.5.A y II.5 · «encareció la misma cesta un 1,31 %»",
          _escalon_fiscal, unidad="%"),
    Cifra("tend_tickets", 15.9, 0.05, "I.5.A y II.5 · «los tickets suben un 15,9 %»",
          lambda c: _tendencia_pct(c.completa["tickets"].to_numpy(float))[0], unidad="%"),
    Cifra("tend_tm_real", -4.9, 0.05, "I.5.A y II.5 · «el ticket medio real cae un 4,9 %»",
          lambda c: _tendencia_pct(
              (c.completa["venta_neta_real"] / c.completa["tickets"]).to_numpy(float))[0],
          unidad="%"),


    # --- lo que la plataforma perdió ---
    Cifra("perdidas_docs", 506, 0, "I.3 y II.3 · «506 documentos»",
          lambda c: int(
              pd.read_csv(RAIZ / "datos_externos" / "arqueos_caja_totales.csv")["operaciones"].sum()
          ), unidad="documentos"),
    Cifra("perdidas_euros", 5_751.82, 0.005, "I.3 y II.3 · «5.751,82 €»",
          lambda c: float(
              pd.read_csv(RAIZ / "datos_externos" / "arqueos_caja_totales.csv")["venta_total"].sum()
          ), unidad="€"),

    # --- surtido y margen ---
    Cifra("articulos_distintos", 6_696, 0, "I.1 y II.5 · «6.696 artículos»",
          lambda c: c.articulos["cod_articulo"].nunique(), grupo="surtido", unidad="artículos"),
    Cifra("concentracion", 5.2, 0.05, "I.5.A y II.5 · «el 5,2 % hace el 50 % de la venta»",
          _concentracion_50, grupo="surtido", unidad="%"),
    Cifra("lentos_pct", 47.5, 0.05, "I.1 · «el 47,5 %» vende menos de una unidad por semana",
          _lentos_pct, grupo="surtido", unidad="%"),
    Cifra("coviran_resto_mediano", 27.3, 0.05, "I.7 y II.7 · resto del surtido, mediano «27,3 %»",
          lambda c: _coviran(c, "resto", "margen_pct_mediano"), grupo="surtido", unidad="%"),
    Cifra("concentracion_80", 20.5, 0.05, "I.5.A y II.5 · «el 20,5 % hace el 80 %»",
          lambda c: _concentracion(c, 80), grupo="surtido", unidad="%"),
    Cifra("supervivencia_surtido", 65, 0.5, "I.5.A y II.5 · «sobrevive el 65 % de los artículos de 2024»",
          _supervivencia_surtido, grupo="surtido", unidad="%"),
    Cifra("fijos_n", 856, 0, "I.5.A y II.5 · «856 artículos»",
          lambda c: _fijos(c)[0], grupo="surtido", unidad="artículos"),
    Cifra("fijos_pct_venta", 55.7, 0.05, "I.5.A y II.5 · «hacen el 55,7 % de la venta»",
          lambda c: _fijos(c)[2], grupo="surtido", unidad="%"),
    Cifra("margen_global", 28.38, 0.005, "I.5.A · «28,38 % de margen teórico sobre tarifa»",
          _margen_global, grupo="surtido", unidad="%"),

    # --- la nevera de bebidas frías ---
    Cifra("frio_margen", 48.5, 0.05, "I.5.A y II.5 · margen de la nevera, 48,5 %",
          lambda c: _frio_margenes(c)[0], grupo="surtido", unidad="%"),
    Cifra("lineal_margen", 26.6, 0.05, "I.5.A y II.5 · margen del lineal comparable, 26,6 %",
          lambda c: _frio_margenes(c)[1], grupo="surtido", unidad="%"),
    Cifra("frio_margen_extra", 4_583.36, 0.5, "I.5.A y II.5 · «4.583,36 € de margen extra»",
          lambda c: _frio_margenes(c)[2], grupo="surtido", unidad="€"),


    # --- los artículos bajo coste ---
    Cifra("bc_filas", 391, 0, "I.3 y II.3 · «391 filas artículo×mes» bajo coste",
          lambda c: len(_bc(c)[0]), grupo="surtido", unidad="filas"),
    Cifra("bc_pct_venta", 0.31, 0.005, "I.3 y II.3 · «el 0,31 % de la venta»",
          lambda c: 100 * _bc(c)[0]["total_venta"].sum() / c.articulos["total_venta"].sum(),
          grupo="surtido", unidad="%"),

    # --- la agrupación por rentabilidad ---
    Cifra("rent_articulos", 849, 0, "I.5.C y II.5 · «849 artículos fijos» con margen",
          lambda c: len(_rentab()["r"]), grupo="surtido", unidad="artículos"),
    Cifra("rent_top10", 20.3, 0.05, "I.5.C y II.5 · 10 artículos, el 20,3 % del margen en euros",
          lambda c: _rentab()["top10"], grupo="surtido", unidad="%"),
]


def _cifras_modelo() -> list[Cifra]:
    """Las del modelo, que exigen reentrenar y por eso van aparte."""

    def tabla(ctx: Contexto) -> pd.DataFrame:
        """Corre el origen móvil una sola vez y cachea la tabla comparativa."""
        sys.path.insert(0, str(RAIZ / "src"))
        import modelos

        if not hasattr(tabla, "_cache"):
            detalle = modelos.evaluar(ctx.completa)
            tabla._cache = modelos.tabla_comparativa(detalle)
        return tabla._cache

    def col(nombre: str, columna: str) -> Callable[[Contexto], float]:
        def fn(ctx: Contexto) -> float:
            return float(tabla(ctx).loc[nombre, columna])
        return fn

    def futuro(ctx: Contexto) -> pd.DataFrame:
        """Pronóstico h=10, escenario 'cierre' - el mismo del notebook 03 §11."""
        import modelos

        if not hasattr(futuro, "_cache"):
            fut = modelos.predecir_futuro(ctx.completa, h=10, escenario="cierre")
            fut["periodo_inicio"] = pd.to_datetime(fut["periodo_inicio"])
            futuro._cache = fut.set_index(fut["periodo_inicio"].dt.strftime("%Y-%m-%d"))
        return futuro._cache

    def contraste(ctx: Contexto) -> pd.DataFrame:
        """La previsión de septiembre frente a los cierres Z que llegaron después."""
        import modelos

        if not hasattr(contraste, "_cache"):
            r = modelos.contrastar_futuro(ctx.completa)
            contraste._cache = r.set_index(r.index.strftime("%Y-%m-%d"))
        return contraste._cache


    def box_jenkins(ctx: Contexto) -> dict:
        """Los pasos 3, 4, 5 y 7 de Box-Jenkins (notebook 03, §6), una sola vez."""
        import modelos

        if not hasattr(box_jenkins, "_cache"):
            box_jenkins._cache = modelos.box_jenkins(ctx.completa)
        return box_jenkins._cache

    def modelos_cobertura(ctx: Contexto) -> dict:
        """Cobertura empírica del intervalo de predicción (modelos.cobertura_intervalo)."""
        import modelos
        if not hasattr(modelos_cobertura, "_cache"):
            modelos_cobertura._cache = modelos.cobertura_intervalo(ctx.completa)
        return modelos_cobertura._cache


    def ajuste_ols(ctx: Contexto):
        """El modelo elegido, ajustado sobre las 135 semanas para leer los coeficientes - igual que el notebook 03, celda 19 (I.6)."""
        import statsmodels.api as sm
        import modelos

        if not hasattr(ajuste_ols, "_cache"):
            serie = ctx.completa
            y = serie["venta_neta"].to_numpy(float)
            X = modelos.matrices(serie)["basico"]
            aj = sm.OLS(y, sm.add_constant(X.to_numpy(float))).fit()
            nombres = ["constante"] + list(X.columns)
            ajuste_ols._cache = {
                "coef": pd.Series(aj.params, index=nombres),
                "p": pd.Series(aj.pvalues, index=nombres),
                "se": pd.Series(aj.bse, index=nombres),
                "r2": aj.rsquared,
                "resid_std": float(np.std(aj.resid)),
            }
        return ajuste_ols._cache

    def coeficiente(nombre: str) -> Callable[[Contexto], float]:
        return lambda ctx: float(ajuste_ols(ctx)["coef"][nombre])

    def valor_ols(clave: str) -> Callable[[Contexto], float]:
        return lambda ctx: float(ajuste_ols(ctx)[clave])

    def diagnostico_residuo(ctx: Contexto):
        """Ljung-Box a los retardos 2 y 4 sobre el residuo del modelo elegido (notebook 03, celda 27) - donde aparece la p baja atribuida al término de acopio, no a memoria real de la serie."""
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import acorr_ljungbox
        import modelos

        if not hasattr(diagnostico_residuo, "_cache"):
            serie = ctx.completa
            y = serie["venta_neta"].to_numpy(float)
            X = modelos.matrices(serie)["basico"]
            resid = sm.OLS(y, sm.add_constant(X.to_numpy(float))).fit().resid
            from statsmodels.tsa.stattools import acf
            lb = acorr_ljungbox(resid, lags=[2, 4], return_df=True)["lb_pvalue"]
            diagnostico_residuo._cache = {"lb2": float(lb.iloc[0]), "lb4": float(lb.iloc[1]),
                                          "acf1": float(acf(resid, nlags=1)[1])}
        return diagnostico_residuo._cache

    def lb(retardo: str) -> Callable[[Contexto], float]:
        return lambda ctx: diagnostico_residuo(ctx)[retardo]


    def tabla_placebo(ctx: Contexto) -> pd.DataFrame:
        """El calendario de folleto y su placebo de fase corrida."""
        import modelos

        if not hasattr(tabla_placebo, "_cache"):
            detalle = modelos.evaluar(ctx.completa, ["ols_basico", "ols_promo_basico", "ols_promo_desfase_basico"])
            tabla_placebo._cache = modelos.tabla_comparativa(detalle)
        return tabla_placebo._cache


    def peor_sin_acopio(ctx: Contexto) -> float:
        """Peor partición del modelo elegido sin el término de acopio."""
        import modelos

        if not hasattr(peor_sin_acopio, "_cache"):
            detalle = modelos.evaluar(ctx.completa, ["ols_basico_sin_acopio"])
            peor_sin_acopio._cache = float(
                modelos.tabla_comparativa(detalle).loc["ols_basico_sin_acopio", "todas_peor"])
        return peor_sin_acopio._cache

    def h26(ctx: Contexto, modelo: str) -> float:
        """MAE mediano a 26 semanas, el horizonte anterior."""
        import modelos

        if not hasattr(h26, "_cache"):
            detalle = modelos.evaluar(ctx.completa, ["ses", "ols_sin_acopio"], horizonte=26)
            h26._cache = modelos.tabla_comparativa(detalle)
        return float(h26._cache.loc[modelo, "todas_mae"])


    return [
        Cifra("mae_modelo", 642, 1, "I.4 y II.4 · «642 € de error mediano»",
              col("ols_basico", "todas_mae"), grupo="modelo", unidad="€"),
        Cifra("peor_modelo", 1335, 5, "I.5.B y II.5 · «peor partición 1.335 €»",
              col("ols_basico", "todas_peor"), grupo="modelo", unidad="€"),
        Cifra("mae_liston", 790, 0.5, "I.5.B · «referencia declarada de antemano»",
              col("naive_1", "todas_mae"), grupo="modelo", unidad="€"),
        Cifra("mejora_vs_liston", 148, 1, "I.5.B y II.5 · el modelo se equivoca «148 € menos por semana» que la referencia",
              lambda c: col("naive_1", "todas_mae")(c) - col("ols_basico", "todas_mae")(c),
              grupo="modelo", unidad="€"),


        # --- I.6, coeficientes del modelo elegido ---
        Cifra("coef_tendencia", 17.17, 0.1, "I.6 · «la tienda crece 17,17 €/semana»",
              coeficiente("tendencia"), grupo="modelo", unidad="€/semana"),
        Cifra("coef_horas_festivo", -202.98, 0.5, "I.6 · «cada hora de cierre cuesta 202,98 €»",
              coeficiente("horas_cierre_festivo"), grupo="modelo", unidad="€/hora"),
        Cifra("coef_acopio", 3_846.26, 1, "I.6 y II.5 («unos 3.800 €») · «la semana de acopio vende 3.846,26 € demás»",
              coeficiente("fiesta_local_acopio"), grupo="modelo", unidad="€"),
        Cifra("modelo_r2", 0.678, 0.001, "I.6 · «R² = 0,678»",
              valor_ols("r2"), grupo="modelo"),


        # --- I.6, ARIMA con xreg ---
        Cifra("mae_arima_011", 641.66, 0.05, "I.5.B e I.6 · arima() con xreg (0,1,1), «641,66 €»",
              col("arima_xreg_011", "todas_mae"), grupo="modelo", unidad="€"),


    ]


def _cifras_supervivencia() -> list[Cifra]:
    """Las del modelo de artículos nuevos."""

    def resultado(ctx: Contexto) -> dict:
        if not hasattr(resultado, "_cache"):
            sys.path.insert(0, str(RAIZ / "src"))
            import supervivencia as sv

            d = sv.construir_grupo_articulos_nuevos()
            X, y = sv.matriz(d)
            tabla, probas = sv.evaluar(X, y)
            lift = sv.curva_lift(y, probas["logistica"])
            por_dpto = d.groupby("departamento")["sobrevive"].mean()

            # --- I.6, la objeción «esto es solo mirar el departamento» ---
            Xsin, ysin = sv.matriz(d, con_coste=False)
            tabla_sin, _ = sv.evaluar(Xsin, ysin)
            dcols = [c for c in X.columns if c.startswith("dpto_")]
            tabla_dpto, _ = sv.evaluar(X[dcols], y, {"logistica": sv.construir_modelos()["logistica"]})
            tabla_nodpto, _ = sv.evaluar(X.drop(columns=dcols), y,
                                          {"logistica": sv.construir_modelos()["logistica"]})

            from statsmodels.stats.outliers_influence import variance_inflation_factor
            Xf = X.astype(float)
            vif = pd.Series(
                [variance_inflation_factor(Xf.values, i) for i in range(Xf.shape[1])],
                index=Xf.columns,
            )

            coef = sv.coeficientes_logistica(X, y)
            robustez = sv.robustez_umbral((2, 3, 4))

            resultado._cache = {
                "n_estrenos": len(d),
                "tasa_base": 100 * d["sobrevive"].mean(),
                "auc": tabla.loc["logistica", "auc"],
                "cal_max_desvio": float(sv.calibracion(y, probas["logistica"])["desvio_puntos"].abs().max()),
                "auc_sin_coste": tabla_sin.loc["logistica", "auc"],
                "auc_solo_dpto": tabla_dpto.loc["logistica", "auc"],
                "auc_sin_dpto": tabla_nodpto.loc["logistica", "auc"],
                "decil10_tasa": 100 * lift.loc[10, "tasa"],
                "bazar": 100 * por_dpto["BAZAR"],
                "drogueria": 100 * por_dpto["DROGUERIA Y PERFUMERIA"],
                "drogueria_sobre_bazar": por_dpto["DROGUERIA Y PERFUMERIA"] / por_dpto["BAZAR"],
                "tasa_base_anterior": 100 * d["reaparece_6m"].mean(),
                "auc_anterior": sv.evaluar(X, d["reaparece_6m"].astype(int).to_numpy(),
                                           {"logistica": sv.construir_modelos()["logistica"]}
                                           )[0].loc["logistica", "auc"],
                "retirados_entre_reaparecidos": 100 * (d["reaparece_6m"] & ~d["sobrevive"]).sum()
                                                / d["reaparece_6m"].sum(),
                "vif_prefijo_promo": vif["prefijo_promo"],
                "vif_oferta": vif["oferta"],
                "vif_precio": vif["precio"],
                "vif_margen_unit": vif["margen_unit"],
                "or_ultramarinos": coef.loc["dpto_ULTRAMARINOS", "odds_ratio"],
                "or_precio": coef.loc["precio", "odds_ratio"],
                "rob_auc_min": robustez["auc"].min(),
                "rob_spearman_min": robustez["spearman_vs_3"].drop(3).min(),
                "rob_spearman_max": robustez["spearman_vs_3"].drop(3).max(),
            }
        return resultado._cache

    def val(clave: str) -> Callable[[Contexto], float]:
        return lambda ctx: resultado(ctx)[clave]

    return [
        Cifra("nuevos_n", 1_302, 0, "I.2, I.5.C y II.5 · «1.302 artículos» estrenados",
              val("n_estrenos"), grupo="modelo", unidad="artículos"),
        Cifra("nuevos_auc", 0.705, 0.0005, "I.5.C · área bajo la curva de la logística",
              val("auc"), grupo="modelo"),
    ]


def _cifras_supervivencia_censura() -> list[Cifra]:
    """Las del CART de supervivencia (notebook 04 §12)."""

    def resultado(ctx: Contexto) -> dict:
        if not hasattr(resultado, "_cache"):
            sys.path.insert(0, str(RAIZ / "src"))
            import supervivencia as sv

            dur = sv.construir_duraciones()
            d_q52 = sv.construir_grupo_articulos_nuevos()
            dv_q52 = sv.construir_ventana_venta(d_q52)
            km = sv.kaplan_meier(dur["tiempo"], dur["evento"])
            X = sv.matriz_x(dur)
            arbol = sv.ArbolSupervivencia().fit(X, dur["tiempo"], dur["evento"])
            hojas = sv.hojas_arbol_supervivencia(arbol, X, dur)
            tabla, _ = sv.evaluar_supervivencia(dur)
            resultado._cache = {
                "n": len(dur),
                "censurados": int((~dur["evento"]).sum()),
                "riesgo_mes3": 100 * km.loc[3, "riesgo_mes"],
                "riesgo_mes12": 100 * km.loc[12, "riesgo_mes"],
                "mejor_s6": 100 * hojas.iloc[0]["S_6m"],
                "mejor_s12": 100 * hojas.iloc[0]["S_12m"],
                "peor_s6": 100 * hojas.iloc[-1]["S_6m"],
                "corte_precio": arbol.arbol_.tree_.threshold[0],
                "candidatos": [sv.curva_producto_nuevo(
                    d_q52, dv_q52,
                    {"departamento": c["departamento"],
                     "precio": c["precio_con_iva"] / (1 + c["iva"] / 100)},
                    dur=dur, arbol=arbol) for c in _CANDIDATOS_PRODUCTO_NUEVO],
                **{f"{m}_{k}": tabla.loc[m, k]
                   for m in ("logistica", "arbol", "arbol_supervivencia")
                   for k in ("auc_6m", "indice_c")},
            }
        return resultado._cache

    def val(clave: str) -> Callable[[Contexto], float]:
        return lambda ctx: resultado(ctx)[clave]

    return [
        Cifra("sv_n", 1509, 0, "I.5.C y I.6 · «1.509 estrenos»", val("n"), grupo="modelo"),
        Cifra("sv_riesgo_mes3", 24.5, 0.05, "I.5.C y II · «en el mes 3 se retira el 24,5 %»",
              val("riesgo_mes3"), grupo="modelo", unidad="%"),
    ]


def _cifras_volumen() -> list[Cifra]:
    """Las del modelo de volumen."""

    def resultado(ctx: Contexto) -> dict:
        if not hasattr(resultado, "_cache"):
            sys.path.insert(0, str(RAIZ / "src"))
            import supervivencia as sv

            d = sv.construir_grupo_articulos_nuevos()
            dv = sv.construir_ventana_venta(d)
            tabla_venta, _ = sv.evaluar_venta(dv)
            mae_mejor = tabla_venta.loc["log_lineal", "mae_medio"]
            mae_nulo = tabla_venta.loc["media_global", "mae_medio"]
            ajuste, _ = sv.ajustar_venta_final(dv)
            resultado._cache = {
                "mae_log_lineal": mae_mejor,
                "mae_nulo": mae_nulo,
                "mejora_pct": 100 * (1 - mae_mejor / mae_nulo),
                "r2": ajuste["ajuste"].rsquared,
            }
        return resultado._cache

    def val(clave: str) -> Callable[[Contexto], float]:
        return lambda ctx: resultado(ctx)[clave]

    return [
        Cifra("volumen_mae", 14.55, 0.005, "I.5.C e I.6 · «MAE de 14,55 unidades»",
              val("mae_log_lineal"), grupo="modelo", unidad="uds"),
        Cifra("volumen_mejora", 22.3, 0.05, "I.5.C · «un 22,3 % mejor»; II.5 · «un 22 % más»",
              val("mejora_pct"), grupo="modelo", unidad="%"),
    ]


_CANDIDATOS_PRODUCTO_NUEVO = [
    dict(clave="claras_huevo", producto="Claras de huevo pasteurizadas 1 L",
         departamento="CHARCUTERIA L.S", precio_con_iva=2.80, iva=10),  # dato de la tienda, 2026-10-06
    dict(clave="hamburguesa_vegetal", producto="Hamburguesa vegetal/vegana congelada",
         departamento="CONGELADOS", precio_con_iva=2.35, iva=10),
    dict(clave="nata_sin_lactosa", producto="Nata para cocinar sin lactosa (200 ml)",
         departamento="LACTEOS", precio_con_iva=7.79 / 6, iva=10),
    dict(clave="masa_pizza_sgluten", producto="Masa de pizza sin gluten",
         departamento="CONGELADOS", precio_con_iva=3.35, iva=10),
    dict(clave="payes", producto="Payés harina sostenible",
         departamento="PANADERIA PASTELERIA", precio_con_iva=2.15, iva=4),
    dict(clave="barra_gallega", producto="Barra gallega",
         departamento="PANADERIA PASTELERIA", precio_con_iva=0.95, iva=4),
    dict(clave="campera", producto="Campera nature",
         departamento="PANADERIA PASTELERIA", precio_con_iva=1.80, iva=4),
]


def comprobar(grupos: set[str]) -> int:
    ctx = Contexto()
    cifras = [c for c in CIFRAS if c.grupo in grupos]
    if "modelo" in grupos:
        cifras += _cifras_modelo()
        cifras += _cifras_supervivencia()
        cifras += _cifras_supervivencia_censura()
        cifras += _cifras_volumen()

    print(f"Comprobando {len(cifras)} cifras contra datos_limpio/\n")
    print(f"{'cifra':<24}{'escrito':>12}{'calculado':>12}  {'':2} donde")
    print("-" * 100)

    desvian, rotas = [], []
    for c in cifras:
        try:
            valor = float(c.calcular(ctx))
        except Exception as exc:  # noqa: BLE001 - un fallo aquí es información
            rotas.append((c, exc))
            print(f"{c.clave:<24}{c.escrito:>12,.2f}{'ERROR':>12}  {'!!':2} {type(exc).__name__}: {exc}")
            continue
        ok = abs(valor - c.escrito) <= c.tolerancia
        if not ok:
            desvian.append((c, valor))
        print(f"{c.clave:<24}{c.escrito:>12,.2f}{valor:>12,.2f}  {'OK' if ok else '!!':2} {c.donde}")

    print()
    if not desvian and not rotas:
        print(f"Las {len(cifras)} cifras cuadran con lo escrito en la memoria.")
        return 0

    if desvian:
        print(f"{len(desvian)} cifra(s) ya no cuadran con lo escrito:\n")
        for c, valor in desvian:
            print(f"  {c.clave}: la memoria dice {c.escrito:,.4g} y sale {valor:,.4g} {c.unidad}".rstrip())
            print(f"      escrito en: {c.donde}")
        print("\n  Antes de tocar la memoria, mirar si cambió el dato o si el cálculo")
        print("  de este fichero no reproduce el de su notebook. No siempre manda el dato.")
    if rotas:
        print(f"\n{len(rotas)} cifra(s) no se han podido calcular; el cálculo está roto.")
    return 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    grupos = {"serie", "surtido"}
    if "--con-modelos" in sys.argv:
        grupos.add("modelo")
    raise SystemExit(comprobar(grupos))

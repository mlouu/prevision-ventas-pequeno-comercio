"""
Modelo de artículos nuevos: ¿seguirá un artículo nuevo en el lineal a los 6
meses?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]

# Grupo de artículos nuevos: estrenos de enero de 2025 a enero de 2026.
ESTRENO_MIN, ESTRENO_MAX = "2025-01", "2026-01"
MESES_SUPERVIVENCIA = 6

# La regla de retirada de la tienda (dato de la tienda): un artículo que pasa
# tres meses seguidos sin venderse se lleva a la zona de liquidación de caja
# con pegatina amarilla y después se retira del lineal.
MESES_SIN_VENTA_RETIRADA = 3

# Departamentos con menos artículos que esto se agrupan en otros.
MIN_POR_DEPARTAMENTO = 20


def cargar_panel(ruta=RAIZ / "datos_limpio/consolidado_mensual.parquet") -> pd.DataFrame:
    """El panel mensual, solo filas de artículo real."""
    m = pd.read_parquet(ruta)
    a = m[m["tipo_fila"] == "articulo"].copy()
    a["mes"] = a["periodo_inicio"].dt.to_period("M")
    return a


def mes_retirada(vendidos: set[int], meses_observados: int,
                 k: int = MESES_SIN_VENTA_RETIRADA) -> int | None:
    """Mes, contado desde el estreno (0), en el que salta la regla de retirada:
    el último de los primeros `k` meses seguidos sin venta.
    """
    for inicio in range(1, meses_observados - k + 2):
        if not any(inicio + j in vendidos for j in range(k)):
            return inicio + k - 1
    return None


def construir_grupo_articulos_nuevos(panel: pd.DataFrame | None = None,
                                     estreno_max: str = ESTRENO_MAX,
                                     meses_sin_venta: int = MESES_SIN_VENTA_RETIRADA
                                     ) -> pd.DataFrame:
    """Una fila por artículo estrenado en la ventana, con etiqueta y predictores."""
    a = cargar_panel() if panel is None else panel

    estreno = a.groupby("cod_articulo")["mes"].min()
    ultimo = a.groupby("cod_articulo")["mes"].max()

    en_ventana = (estreno >= pd.Period(ESTRENO_MIN)) & (estreno <= pd.Period(estreno_max))
    elegibles = estreno[en_ventana]

    fin = a["mes"].max()
    meses_con_venta = (a[a["cod_articulo"].isin(elegibles.index)]
                       .groupby("cod_articulo")["mes"].apply(set))
    retirada = pd.Series({
        cod: mes_retirada({(m - e).n for m in meses_con_venta[cod]}, (fin - e).n,
                          k=meses_sin_venta)
        for cod, e in elegibles.items()
    }, dtype="object")
    sobrevive = retirada.apply(lambda r: r is None or r > MESES_SUPERVIVENCIA)
    reaparece = (ultimo[elegibles.index] - elegibles).apply(lambda p: p.n) >= MESES_SUPERVIVENCIA

    # El mes de estreno, y solo ese: cualquier columna posterior sería trampa.
    primer = (a.merge(elegibles.rename("estreno"), on="cod_articulo")
              .query("mes == estreno")
              .set_index("cod_articulo"))

    d = pd.DataFrame(index=primer.index)
    d["sobrevive"] = sobrevive.reindex(d.index).astype(bool).to_numpy()
    d["reaparece_6m"] = reaparece.reindex(d.index).to_numpy()
    d["mes_estreno"] = primer["estreno"].dt.month.to_numpy()
    d["descripcion"] = primer["descripcion"].to_numpy()

    # --- Predictores del estreno ----------------------------------------
    d["unidades"] = primer["unidades"].to_numpy()
    d["precio"] = primer["precio"].to_numpy()
    d["venta"] = primer["total_venta"].to_numpy()
    d["oferta"] = primer["oferta"].astype(int).to_numpy()

    # El prefijo "(P)" en la descripción marca artículo de promoción de la
    # central.
    d["prefijo_promo"] = (primer["descripcion"].str.startswith("(P)")
                          .astype(int).to_numpy())

    # Margen unitario en euros sobre `tarifa_costo`, que es la base de mayor
    # cobertura.
    d["margen_unit"] = (primer["precio"] - primer["tarifa_costo"]).to_numpy()

    dep = primer["departamento"].to_numpy()
    conteo = pd.Series(dep).value_counts()
    raros = set(conteo[conteo < MIN_POR_DEPARTAMENTO].index)
    d["departamento"] = np.where(pd.Series(dep).isin(raros), "OTROS", dep)

    # El departamento sin agrupar, antes de meter los pequeños en otros.
    d["departamento_real"] = dep

    return d


# Las columnas que dependen de la base de coste bloqueada.
COLS_COSTE = ["margen_unit"]


def matriz_x(d: pd.DataFrame, con_coste: bool = True) -> pd.DataFrame:
    """Solo la parte `X` de `matriz()`: la usa también la supervivencia con
    censura (`construir_duraciones()`), que no tiene etiqueta sí/no.
    """
    numericas = ["unidades", "precio", "venta", "oferta", "prefijo_promo",
                 "mes_estreno"]
    if con_coste:
        numericas = numericas + COLS_COSTE

    return pd.concat(
        [d[numericas],
         pd.get_dummies(d["departamento"], prefix="dpto", drop_first=True).astype(int)],
        axis=1,
    )


def matriz(d: pd.DataFrame, con_coste: bool = True):
    """Matriz de diseño `(X, y)` para los modelos, con el departamento en dummies."""
    return matriz_x(d, con_coste=con_coste), d["sobrevive"].astype(int).to_numpy()


def curva_lift(y_real, puntuacion, n_grupos: int = 10) -> pd.DataFrame:
    """Tabla de lift por deciles de puntuación."""
    orden = np.argsort(-np.asarray(puntuacion))
    y = np.asarray(y_real)[orden]
    base = y.mean()

    grupos = np.array_split(np.arange(len(y)), n_grupos)
    filas = []
    for i, g in enumerate(grupos, start=1):
        tasa = y[g].mean()
        filas.append({
            "decil": i,
            "n": len(g),
            "sobreviven": int(y[g].sum()),
            "tasa": tasa,
            "lift": tasa / base if base else np.nan,
            "acumulado_capturado": y[np.concatenate(grupos[:i])].sum() / y.sum(),
        })
    return pd.DataFrame(filas).set_index("decil")


# ---------------------------------------------------------------------------
# Los modelos
# ---------------------------------------------------------------------------

# Semilla fija en todo el módulo: sin ella, CART y los pliegues cambian entre
# ejecuciones y las cifras de la memoria dejarían de ser reproducibles.
SEMILLA = 42
N_PLIEGUES = 5


def construir_modelos(profundidad_arbol: int = 3):
    """Los tres modelos, cada uno con su procedencia."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    return {
        "baseline": (DummyClassifier(strategy="prior"), "DummyClassifier"),
        "logistica": (
            Pipeline([("escala", StandardScaler()),
                      ("glm", LogisticRegression(max_iter=2000,
                                                 random_state=SEMILLA))]),
            "regresión logística",
        ),
        "arbol": (
            DecisionTreeClassifier(max_depth=profundidad_arbol,
                                   min_samples_leaf=20, random_state=SEMILLA),
            "CART",
        ),
    }


def evaluar(X, y, modelos=None, n_pliegues: int = N_PLIEGUES):
    """Validación cruzada estratificada, con ROC, AUC y lift."""
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    modelos = modelos or construir_modelos()
    cv = StratifiedKFold(n_splits=n_pliegues, shuffle=True, random_state=SEMILLA)

    filas, probas = [], {}
    for nombre, (modelo, procedencia) in modelos.items():
        p = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
        probas[nombre] = p
        filas.append({
            "modelo": nombre,
            "procedencia": procedencia,
            "auc": roc_auc_score(y, p),
            "acierto": accuracy_score(y, (p >= 0.5).astype(int)),
            "lift_decil_1": curva_lift(y, p).loc[1, "lift"],
        })

    tabla = pd.DataFrame(filas).set_index("modelo")
    return tabla, pd.DataFrame(probas, index=X.index)


def calibracion(y, p, n_grupos: int = 5) -> pd.DataFrame:
    """Probabilidad prevista frente a proporción observada, por tramos."""
    y = np.asarray(y)
    tramo = pd.qcut(pd.Series(np.asarray(p)), n_grupos, labels=False)
    t = pd.DataFrame({"prevista": np.asarray(p), "observada": y, "tramo": tramo})         .groupby("tramo").agg(prevista=("prevista", "mean"), observada=("observada", "mean"),
                              articulos=("observada", "size"))
    t["desvio_puntos"] = 100 * (t["observada"] - t["prevista"])
    return t


def coeficientes_logistica(X, y):
    """Los coeficientes en odds ratio, que es como se leen."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    escala = StandardScaler().fit(X)
    glm = LogisticRegression(max_iter=2000, random_state=SEMILLA).fit(
        escala.transform(X), y)

    return (pd.DataFrame({
        "coef": glm.coef_[0],
        "odds_ratio": np.exp(glm.coef_[0]),
    }, index=X.columns).sort_values("odds_ratio", ascending=False))

def oferta_posterior(d: pd.DataFrame, panel: pd.DataFrame | None = None,
                     meses: int = MESES_SUPERVIVENCIA) -> pd.Series:
    """¿Tuvo el artículo oferta en algún mes entre el 1 y `meses` después de su
    estreno?
    """
    a = cargar_panel() if panel is None else panel
    estreno = a.groupby("cod_articulo")["mes"].min()
    b = a[a["cod_articulo"].isin(d.index)].copy()
    b["offset"] = (b["mes"] - b["cod_articulo"].map(estreno)).apply(lambda p: p.n)
    return (b[(b["offset"] >= 1) & (b["offset"] <= meses)]
            .groupby("cod_articulo")["oferta"].any()
            .reindex(d.index, fill_value=False))


def robustez_umbral(umbrales=(2, 3, 4), panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """¿Dependen las conclusiones del modelo de artículos nuevos de que la
    regla sea de 3 meses?
    """
    from scipy.stats import spearmanr

    a = cargar_panel() if panel is None else panel
    filas, puntuaciones = [], {}
    for k in umbrales:
        d = construir_grupo_articulos_nuevos(a, meses_sin_venta=k)
        X, y = matriz(d)
        tabla, probas = evaluar(X, y)
        lift = curva_lift(y, probas["logistica"])
        arbol = construir_modelos()["arbol"][0].fit(X, y)
        por_dpto = d.groupby("departamento")["sobrevive"].agg(["size", "mean"])
        por_dpto = por_dpto[(por_dpto["size"] >= MIN_POR_DEPARTAMENTO)
                            & (por_dpto.index != "OTROS")]
        tamano, por_dpto = por_dpto["size"], por_dpto["mean"]
        puntuaciones[k] = probas["logistica"]
        filas.append({
            "meses_sin_venta": k,
            "tasa_base": d["sobrevive"].mean(),
            "auc": tabla.loc["logistica", "auc"],
            "auc_arbol": tabla.loc["arbol", "auc"],
            "decil1": lift.loc[1, "tasa"],
            "decil10": lift.loc[10, "tasa"],
            "peor_dpto": por_dpto.idxmin(),
            "peor_tasa": por_dpto.min(),
            "mejor_dpto": por_dpto.idxmax(),
            "mejor_tasa": por_dpto.max(),
            "mejor_n": int(tamano[por_dpto.idxmax()]),
            "primera_pregunta": f"{X.columns[arbol.tree_.feature[0]]} ≤ {arbol.tree_.threshold[0]:.2f}",
        })
    t = pd.DataFrame(filas).set_index("meses_sin_venta")
    if 3 in puntuaciones:
        t["spearman_vs_3"] = [spearmanr(puntuaciones[k], puntuaciones[3])[0] for k in t.index]
    return t


# ---------------------------------------------------------------------------
# Herramienta: puntuar un producto hipotético (no es el modelo del TFM, que ya
# está cerrado en el modelo de artículos nuevos - es una utilidad encima de él,
# para explorar casos concretos que describa la tienda).
# ---------------------------------------------------------------------------

def ajustar_logistica_final(X, y):
    """La logística ajustada sobre todo el grupo de artículos nuevos, para
    puntuar artículos nuevos.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    modelo = Pipeline([("escala", StandardScaler()),
                       ("glm", LogisticRegression(max_iter=2000, random_state=SEMILLA))])
    modelo.fit(X, y)
    return modelo


def predecir_nuevo(d: pd.DataFrame, nuevo: dict) -> float:
    """Probabilidad de supervivencia de un producto hipotético."""
    con_coste = "margen_unit" in nuevo
    X, y = matriz(d, con_coste=con_coste)
    modelo = ajustar_logistica_final(X, y)

    fila = dict(nuevo)
    fila.setdefault("venta", fila["precio"] * fila["unidades"])
    fila.setdefault("oferta", 0)
    fila.setdefault("prefijo_promo", 0)
    fila.setdefault("mes_estreno", 1)
    dpto = fila.pop("departamento")
    if dpto not in set(d["departamento"]):
        dpto = "OTROS"

    tabla = pd.DataFrame([fila]).reindex(columns=X.columns, fill_value=0)
    col_dpto = f"dpto_{dpto}"
    if col_dpto in tabla.columns:
        tabla[col_dpto] = 1

    return float(modelo.predict_proba(tabla)[0, 1])


def vecinos_mas_parecidos(d: pd.DataFrame, nuevo: dict, k: int = 5,
                          mismo_departamento: bool = True) -> pd.DataFrame:
    """Los `k` artículos ya estrenados más parecidos a `nuevo`, por distancia."""
    cols = ["precio", "unidades", "venta"]
    if "margen_unit" in nuevo and "margen_unit" in d.columns:
        cols = cols + ["margen_unit"]

    col_dpto = "departamento_real" if "departamento_real" in d.columns else "departamento"
    candidatos = d
    if mismo_departamento and (d[col_dpto] == nuevo["departamento"]).any():
        candidatos = d[d[col_dpto] == nuevo["departamento"]]

    medias, stds = d[cols].mean(), d[cols].std()
    venta = nuevo.get("venta", nuevo["precio"] * nuevo["unidades"])
    valores_nuevo = {**nuevo, "venta": venta}

    z_cand = (candidatos[cols] - medias) / stds
    z_nuevo = pd.Series({c: (valores_nuevo.get(c, medias[c]) - medias[c]) / stds[c]
                         for c in cols})
    dist = ((z_cand - z_nuevo) ** 2).sum(axis=1) ** 0.5

    cercanos = candidatos.loc[dist.sort_values().index[:k]].copy()
    cercanos["sobrevive"] = cercanos["sobrevive"].map({True: "sí", False: "no"})
    columnas = ["descripcion", col_dpto, "precio", "unidades", "venta", "sobrevive"]
    # Si `d` trae la ventana de venta (construir_ventana_venta), se enseña también:
    # da contexto real para juzgar la cifra del modelo de volumen.
    columnas += [c for c in ("unidades_ventana", "venta_ventana") if c in cercanos.columns]
    return cercanos[columnas].rename(columns={col_dpto: "departamento"})


# ---------------------------------------------------------------------------
# Ampliación: no solo si sobrevive, sino cuánto va a vender
# ---------------------------------------------------------------------------

MESES_VENTANA_VENTA = MESES_SUPERVIVENCIA  # misma ventana y la misma censura que el modelo de artículos nuevos


def construir_ventana_venta(d: pd.DataFrame, panel: pd.DataFrame | None = None,
                             meses_ventana: int = MESES_VENTANA_VENTA) -> pd.DataFrame:
    """Añade al grupo de artículos nuevos el total de unidades y venta en los
    `meses_ventana` meses siguientes al estreno, incluido el propio mes.
    """
    a = cargar_panel() if panel is None else panel
    estreno = a.groupby("cod_articulo")["mes"].min()

    a2 = (a[a["cod_articulo"].isin(d.index)]
          [["cod_articulo", "mes", "unidades", "total_venta"]].copy())
    a2["estreno"] = a2["cod_articulo"].map(estreno)
    a2["offset"] = (a2["mes"] - a2["estreno"]).apply(lambda p: p.n)
    ventana = a2[(a2["offset"] >= 0) & (a2["offset"] < meses_ventana)]

    agg = ventana.groupby("cod_articulo").agg(
        unidades_ventana=("unidades", "sum"),
        venta_ventana=("total_venta", "sum"),
        meses_con_venta=("mes", "count"),
    )
    return d.join(agg)


def matriz_venta(dv: pd.DataFrame, con_coste: bool = False):
    """Matriz de diseño para el modelo de volumen: solo lo que se conoce antes
    de vender una sola unidad.
    """
    numericas = ["precio", "oferta", "prefijo_promo", "mes_estreno"]
    if con_coste:
        numericas = numericas + COLS_COSTE

    X = pd.concat(
        [dv[numericas],
         pd.get_dummies(dv["departamento"], prefix="dpto", drop_first=True).astype(int)],
        axis=1,
    )
    return X


def _alpha_binomial_negativa(y: np.ndarray) -> float:
    """Estimador de alpha por el método de los momentos (Cameron y Trivedi,
    forma NB2: var(y) = media(y) + alpha·media(y)²).
    """
    media, var = float(y.mean()), float(y.var())
    return max((var - media) / media ** 2, 1e-4)


def evaluar_venta(dv: pd.DataFrame, n_pliegues: int = N_PLIEGUES,
                   con_coste: bool = False):
    """Compara, con la misma K-fold para todos, los candidatos a estimar
    `unidades_ventana`: dos modelos nulos, Poisson, Binomial Negativa y un
    log-lineal.
    """
    import statsmodels.api as sm
    from sklearn.model_selection import KFold

    X = matriz_venta(dv, con_coste=con_coste)
    y = dv["unidades_ventana"].to_numpy(dtype=float)
    dpto = dv["departamento"].to_numpy()
    Xc = sm.add_constant(X, has_constant="add").reset_index(drop=True)

    cv = KFold(n_splits=n_pliegues, shuffle=True, random_state=SEMILLA)
    resultados = {m: [] for m in
                  ["media_global", "media_departamento", "poisson",
                   "binomial_negativa", "log_lineal"]}

    for train, test in cv.split(Xc):
        y_tr, y_te = y[train], y[test]

        resultados["media_global"].append(float(np.abs(y_tr.mean() - y_te).mean()))

        medias_dpto = pd.Series(y_tr).groupby(dpto[train]).mean()
        pred_dpto = (pd.Series(dpto[test]).map(medias_dpto)
                     .fillna(y_tr.mean()).to_numpy())
        resultados["media_departamento"].append(float(np.abs(pred_dpto - y_te).mean()))

        m_pois = sm.GLM(y_tr, Xc.iloc[train], family=sm.families.Poisson()).fit()
        resultados["poisson"].append(
            float(np.abs(m_pois.predict(Xc.iloc[test]) - y_te).mean()))

        alpha = _alpha_binomial_negativa(y_tr)
        m_nb = sm.GLM(y_tr, Xc.iloc[train],
                      family=sm.families.NegativeBinomial(alpha=alpha)).fit()
        resultados["binomial_negativa"].append(
            float(np.abs(m_nb.predict(Xc.iloc[test]) - y_te).mean()))

        ylog_tr = np.log1p(y_tr)
        m_ols = sm.OLS(ylog_tr, Xc.iloc[train]).fit()
        pred_ols = np.exp(m_ols.predict(Xc.iloc[test])) - 1
        resultados["log_lineal"].append(float(np.abs(pred_ols - y_te).mean()))

    detalle = pd.DataFrame(resultados)
    procedencia = {
        "media_global": "modelo nulo",
        "media_departamento": "modelo nulo (por departamento)",
        "poisson": "regresión de Poisson",
        "binomial_negativa": "Binomial Negativa (alpha por método de momentos, ver _alpha_binomial_negativa)",
        "log_lineal": "log-lineal, misma familia que el modelo agregado",
    }
    tabla = pd.DataFrame({
        "procedencia": procedencia,
        "mae_medio": detalle.mean(),
        "mae_std": detalle.std(),
    }).sort_values("mae_medio")
    return tabla, detalle


def ajustar_venta_final(dv: pd.DataFrame, con_coste: bool = False,
                         modelo: str = "log_lineal"):
    """El modelo de volumen ajustado sobre todo el grupo de artículos nuevos,
    listo para puntuar un producto nuevo - análogo a
    `ajustar_logistica_final()` para el modelo de artículos nuevos.
    """
    import statsmodels.api as sm

    X = matriz_venta(dv, con_coste=con_coste)
    y = dv["unidades_ventana"].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")

    if modelo == "log_lineal":
        ylog = np.log1p(y)  # log(1+unidades): hay artículos con unidades_ventana = 0
        ajuste = sm.OLS(ylog, Xc).fit()
        return {"tipo": "log_lineal", "ajuste": ajuste}, X.columns
    if modelo == "binomial_negativa":
        alpha = _alpha_binomial_negativa(y)
        ajuste = sm.GLM(y, Xc, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
        return {"tipo": "binomial_negativa", "ajuste": ajuste, "alpha": alpha}, X.columns
    raise ValueError(f"modelo desconocido: {modelo!r} (usar 'log_lineal' o 'binomial_negativa')")


def predecir_venta_futura(dv: pd.DataFrame, nuevo: dict,
                           meses_ventana: int = MESES_VENTANA_VENTA,
                           modelo: str = "log_lineal") -> dict:
    """Unidades y venta esperadas de un producto hipotético en sus primeros
    `meses_ventana` meses - la ampliación de `predecir_nuevo()` que en vez de
    sí/no da una cifra.
    """
    import statsmodels.api as sm

    con_coste = "margen_unit" in nuevo
    ajuste, columnas = ajustar_venta_final(dv, con_coste=con_coste, modelo=modelo)

    fila = {k: v for k, v in nuevo.items() if k != "departamento"}
    fila.setdefault("oferta", 0)
    fila.setdefault("prefijo_promo", 0)
    fila.setdefault("mes_estreno", 1)
    dpto = nuevo["departamento"]
    if dpto not in set(dv["departamento"]):
        dpto = "OTROS"

    tabla = pd.DataFrame([fila]).reindex(columns=columnas, fill_value=0)
    col_dpto = f"dpto_{dpto}"
    if col_dpto in tabla.columns:
        tabla[col_dpto] = 1
    tabla_c = sm.add_constant(tabla, has_constant="add")

    if ajuste["tipo"] == "log_lineal":
        crudo = np.exp(ajuste["ajuste"].predict(tabla_c))[0] - 1
        unidades = float(max(crudo, 0.0))  # log1p puede devolver un residuo < 0 para precios/departamentos extremos
    else:
        unidades = float(ajuste["ajuste"].predict(tabla_c)[0])

    media_dpto = dv.loc[dv["departamento"] == dpto, "unidades_ventana"].mean()
    if np.isnan(media_dpto):
        media_dpto = dv["unidades_ventana"].mean()

    precio = nuevo["precio"]
    return {
        "unidades_total": unidades,
        "unidades_mes": unidades / meses_ventana,
        "venta_total_eur": unidades * precio,
        "venta_mes_eur": unidades / meses_ventana * precio,
        "media_departamento_total": float(media_dpto),
        "razon_vs_departamento": float(unidades / media_dpto) if media_dpto else float("nan"),
    }


def predecir_producto_nuevo(d: pd.DataFrame, dv: pd.DataFrame, nuevo: dict,
                             k: int = 5, meses_ventana: int = MESES_VENTANA_VENTA,
                             modelo_venta: str = "log_lineal") -> dict:
    """Las tres preguntas sobre un producto hipotético en una sola llamada:
    ¿sobrevive?, ¿cuánto vende? y ¿a qué se parece? - compone
    `predecir_nuevo()`, `predecir_venta_futura()` y `vecinos_mas_parecidos()`.
    """
    venta = predecir_venta_futura(dv, nuevo, meses_ventana=meses_ventana, modelo=modelo_venta)

    nuevo_completo = dict(nuevo)
    unidades_estimadas = "unidades" not in nuevo_completo
    if unidades_estimadas:
        nuevo_completo["unidades"] = venta["unidades_mes"]

    return {
        "probabilidad_supervivencia": predecir_nuevo(d, nuevo_completo),
        "unidades_estimadas_para_supervivencia": unidades_estimadas,
        **venta,
        "vecinos": vecinos_mas_parecidos(dv, nuevo_completo, k=k),
    }


# ---------------------------------------------------------------------------
# Supervivencia con censura: el CART de supervivencia no sustituye a la
# logística a 6 meses, que sigue siendo el modelo de artículos nuevos, sino que
# la amplía con la técnica específica para este tipo de dato: un árbol cuya
# variable respuesta es el tiempo hasta que la tienda retira el artículo, con
# la censura dentro del modelo en vez de fuera.
# ---------------------------------------------------------------------------

# Los estrenos más recientes que siguen aportando algo.
ESTRENO_MAX_CENSURA = "2026-04"


def construir_duraciones(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una fila por artículo estrenado desde enero de 2025, con `tiempo` y
    `evento` además de los mismos predictores del mes de estreno que el modelo
    de artículos nuevos.
    """
    a = cargar_panel() if panel is None else panel
    d = construir_grupo_articulos_nuevos(a, estreno_max=ESTRENO_MAX_CENSURA)
    d = d.drop(columns=["sobrevive", "reaparece_6m"])

    fin = a["mes"].max()
    estreno = a.groupby("cod_articulo")["mes"].min()

    tiempo, evento, vuelve, observados = [], [], [], []
    meses_con_venta = (a[a["cod_articulo"].isin(d.index)]
                       .groupby("cod_articulo")["mes"].apply(set))
    for cod in d.index:
        e = estreno[cod]
        vendidos = {(m - e).n for m in meses_con_venta[cod]}
        obs = (fin - e).n  # último mes observable, contado desde el estreno (0)
        retirada = mes_retirada(vendidos, obs)
        observados.append(obs)
        tiempo.append(obs if retirada is None else retirada)
        evento.append(retirada is not None)
        vuelve.append(retirada is not None and any(o > retirada for o in vendidos))

    d["tiempo"] = tiempo
    d["evento"] = evento
    d["vuelve_tras_hueco"] = vuelve
    d["meses_observados"] = observados
    return d


def estado_a(dur: pd.DataFrame, meses: int = MESES_SUPERVIVENCIA) -> pd.Series:
    """¿Seguía en el lineal a los `meses` meses, según la regla de la tienda?"""
    # Retirado en el mes t: ya no está en el lineal en t. Censurado en t: sí
    # lo está (la regla no ha saltado en t), así que llega a `meses` si t >= meses.
    vivo = (dur["tiempo"] > meses) | (~dur["evento"] & (dur["tiempo"] >= meses))
    sabido = dur["evento"] | (dur["tiempo"] >= meses)
    return vivo.where(sabido)


def kaplan_meier(tiempo, evento) -> pd.DataFrame:
    """Kaplan-Meier en tiempo discreto (meses): en cada mes, cuántos siguen en
    riesgo, cuántos mueren, el riesgo del mes y la supervivencia acumulada.
    """
    tiempo, evento = np.asarray(tiempo), np.asarray(evento, dtype=bool)
    filas, s, h_acum = [], 1.0, 0.0
    for t in range(1, int(tiempo.max()) + 1):
        en_riesgo = int((tiempo >= t).sum())
        muertes = int(((tiempo == t) & evento).sum())
        riesgo = muertes / en_riesgo if en_riesgo else 0.0
        s *= 1 - riesgo
        h_acum += riesgo
        filas.append({"mes": t, "en_riesgo": en_riesgo, "muertes": muertes,
                      "riesgo_mes": riesgo, "supervivencia": s,
                      "nelson_aalen": h_acum})
    return pd.DataFrame(filas).set_index("mes")


class ArbolSupervivencia:
    """CART de supervivencia de hojas exponenciales - el método `"exp"` de
    `rpart` (Therneau y Atkinson), el mismo paquete de la práctica de.
    """

    def __init__(self, profundidad: int = 3, min_hoja: int = 20):
        self.profundidad, self.min_hoja = profundidad, min_hoja

    def fit(self, X, tiempo, evento):
        from sklearn.tree import DecisionTreeRegressor

        tiempo, evento = np.asarray(tiempo), np.asarray(evento, dtype=float)
        self.km_ = kaplan_meier(tiempo, evento)
        exposicion = self._lambda0(tiempo)
        self.arbol_ = DecisionTreeRegressor(
            criterion="poisson", max_depth=self.profundidad,
            min_samples_leaf=self.min_hoja, random_state=SEMILLA,
        ).fit(X, evento / exposicion, sample_weight=exposicion)
        return self

    def _lambda0(self, t):
        """Riesgo acumulado del grupo de entrenamiento en el mes `t`, constante
        más allá del último mes observado.
        """
        na = self.km_["nelson_aalen"]
        t = np.clip(np.atleast_1d(t), 1, na.index.max())
        return na.reindex(t).to_numpy()

    def riesgo_relativo(self, X) -> np.ndarray:
        return self.arbol_.predict(X)

    def supervivencia(self, X, meses: int) -> np.ndarray:
        return np.exp(-self.riesgo_relativo(X) * self._lambda0(meses))


def indice_c(tiempo, evento, riesgo) -> float:
    """Índice C de Harrell: el AUC de la supervivencia."""
    tiempo, evento, riesgo = (np.asarray(tiempo), np.asarray(evento, dtype=bool),
                              np.asarray(riesgo, dtype=float))
    antes = (tiempo[:, None] < tiempo[None, :]) & evento[:, None]
    mas = riesgo[:, None] > riesgo[None, :]
    igual = riesgo[:, None] == riesgo[None, :]
    return float((mas[antes].sum() + 0.5 * igual[antes].sum()) / antes.sum())


def evaluar_supervivencia(dur: pd.DataFrame, meses: int = MESES_SUPERVIVENCIA,
                          n_pliegues: int = N_PLIEGUES, con_coste: bool = True):
    """Compara el árbol de supervivencia con los tres modelos sí/no del modelo
    de artículos nuevos, todos con la misma etiqueta: la regla de retirada de
    la tienda.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X = matriz_x(dur, con_coste=con_coste)
    estado = estado_a(dur, meses)
    sabido = (dur["meses_observados"] >= meses).to_numpy()
    assert estado[sabido].notna().all()
    y6 = estado.fillna(False).astype(int).to_numpy()
    t, e = dur["tiempo"].to_numpy(), dur["evento"].to_numpy()

    clasificadores = construir_modelos()
    riesgo = {nombre: np.zeros(len(dur)) for nombre in
              [*clasificadores, "arbol_supervivencia"]}
    cv = StratifiedKFold(n_splits=n_pliegues, shuffle=True, random_state=SEMILLA)
    for tr, te in cv.split(X, e):
        tr_s = tr[sabido[tr]]
        for nombre, (modelo, _) in clasificadores.items():
            modelo.fit(X.iloc[tr_s], y6[tr_s])
            riesgo[nombre][te] = 1 - modelo.predict_proba(X.iloc[te])[:, 1]
        arbol = ArbolSupervivencia().fit(X.iloc[tr], t[tr], e[tr])
        riesgo["arbol_supervivencia"][te] = arbol.riesgo_relativo(X.iloc[te])

    procedencia = {
        "baseline": "DummyClassifier",
        "logistica": "logística, solo el grupo del modelo de artículos nuevos",
        "arbol": "CART de clasificación, solo el grupo del modelo de artículos nuevos",
        "arbol_supervivencia": "CART de supervivencia (rpart 'exp'), con censurados",
    }
    tabla = pd.DataFrame([{
        "modelo": nombre,
        "procedencia": procedencia[nombre],
        "entrena_con": int(len(dur) if nombre == "arbol_supervivencia" else sabido.sum()),
        "auc_6m": roc_auc_score(1 - y6[sabido], r[sabido]),
        "indice_c": indice_c(t, e, r),
    } for nombre, r in riesgo.items()]).set_index("modelo")
    return tabla, pd.DataFrame(riesgo, index=dur.index)


def precio_tras_hueco(dur: pd.DataFrame, panel: pd.DataFrame | None = None) -> pd.Series:
    """Para los artículos que vuelven a vender después del hueco de retirada,
    precio medio después del hueco dividido por el precio medio antes.
    """
    a = cargar_panel() if panel is None else panel
    vuelven = dur[dur["vuelve_tras_hueco"]]
    estreno = a.groupby("cod_articulo")["mes"].min()
    b = a[a["cod_articulo"].isin(vuelven.index)].copy()
    b["offset"] = (b["mes"] - b["cod_articulo"].map(estreno)).apply(lambda p: p.n)
    b["despues"] = b["offset"] >= b["cod_articulo"].map(vuelven["tiempo"])
    suma = b.groupby(["cod_articulo", "despues"])[["total_venta", "unidades"]].sum()
    precio = (suma["total_venta"] / suma["unidades"]).unstack()
    return (precio[True] / precio[False]).replace([np.inf, -np.inf], np.nan).dropna()


def exportar_para_rpart(dur: pd.DataFrame, ruta) -> None:
    """Escribe el grupo en CSV para `src/supervivencia_rpart.R`, que ajusta el
    mismo árbol con `rpart(method="exp")` de R y permite contrastarlo.
    """
    X = matriz_x(dur)
    X.columns = [c.replace(" ", "_").replace(".", "_") for c in X.columns]
    (X.assign(tiempo=dur["tiempo"].to_numpy(),
              evento=dur["evento"].astype(int).to_numpy())
     .to_csv(ruta, index=False))


def curva_producto_nuevo(d: pd.DataFrame, dv: pd.DataFrame, nuevo: dict,
                         dur: pd.DataFrame | None = None,
                         arbol: ArbolSupervivencia | None = None,
                         meses=(3, 6, 12)) -> dict:
    """La curva del árbol de supervivencia para un producto hipotético: qué
    parte de los artículos de su mismo perfil sigue en el lineal a los `meses`
    meses.
    """
    if dur is None:
        dur = construir_duraciones()
    X = matriz_x(dur)
    if arbol is None:
        arbol = ArbolSupervivencia().fit(X, dur["tiempo"], dur["evento"])

    fila = {k: v for k, v in nuevo.items() if k != "departamento"}
    if "unidades" not in fila:
        fila["unidades"] = predecir_venta_futura(dv, nuevo)["unidades_mes"]
    fila.setdefault("venta", fila["precio"] * fila["unidades"])
    fila.setdefault("oferta", 0)
    fila.setdefault("prefijo_promo", 0)
    fila.setdefault("mes_estreno", 1)
    dpto = nuevo["departamento"] if nuevo["departamento"] in set(dur["departamento"]) else "OTROS"
    tabla = pd.DataFrame([fila]).reindex(columns=X.columns, fill_value=0)
    if f"dpto_{dpto}" in tabla.columns:
        tabla[f"dpto_{dpto}"] = 1

    hoja = int(arbol.arbol_.apply(tabla)[0])
    hojas = hojas_arbol_supervivencia(arbol, X, dur, meses).set_index("hoja")
    riesgo = float(arbol.riesgo_relativo(tabla)[0])
    return {
        "hoja": hoja,
        "regla": hojas.loc[hoja, "regla"],
        "articulos_perfil": int(hojas.loc[hoja, "articulos"]),
        **{f"S_{m}m": float(np.exp(-riesgo * arbol._lambda0(m))[0]) for m in meses},
    }


def hojas_arbol_supervivencia(arbol: ArbolSupervivencia, X: pd.DataFrame,
                              dur: pd.DataFrame, meses=(3, 6, 12)) -> pd.DataFrame:
    """Una fila por hoja: la regla que lleva a ella, cuántos artículos, su
    riesgo relativo y la supervivencia que da la exponencial a varios meses.
    """
    arb = arbol.arbol_.tree_
    hoja = arbol.arbol_.apply(X)

    def reglas(nid, camino):
        if arb.children_left[nid] == -1:
            yield nid, camino
            return
        var, umbral = X.columns[arb.feature[nid]], arb.threshold[nid]
        if var.startswith("dpto_") or var in ("oferta", "prefijo_promo"):
            izq, der = f"no {var}", var
        else:
            izq, der = f"{var} ≤ {umbral:.2f}", f"{var} > {umbral:.2f}"
        yield from reglas(arb.children_left[nid], camino + [izq])
        yield from reglas(arb.children_right[nid], camino + [der])

    m = meses[1]
    filas = []
    for nid, camino in reglas(0, []):
        en_hoja = hoja == nid
        rr = float(arb.value[nid][0][0])
        km = kaplan_meier(dur["tiempo"][en_hoja], dur["evento"][en_hoja])
        fila = {"hoja": int(nid), "regla": " y ".join(camino),
                "articulos": int(en_hoja.sum()),
                "mueren": int(dur["evento"][en_hoja].sum()),
                "riesgo_relativo": rr}
        for mm in meses:
            fila[f"S_{mm}m"] = float(np.exp(-rr * arbol._lambda0(mm))[0])
        # Si la hoja se queda sin artículos antes del mes m, su supervivencia
        # real es la última que alcanzó (0 si murieron todos).
        fila[f"S_{m}m_real"] = float(km["supervivencia"]
                                     .reindex(range(1, m + 1)).ffill().iloc[-1])
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("riesgo_relativo").reset_index(drop=True)


if __name__ == "__main__":
    d = construir_grupo_articulos_nuevos()
    print("=" * 70)
    print(f"Grupo de artículos nuevos · {len(d)} articulos estrenados "
          f"{ESTRENO_MIN} a {ESTRENO_MAX}")
    print("=" * 70)
    print(f"\nsobreviven >= {MESES_SUPERVIVENCIA} meses: {int(d['sobrevive'].sum())} "
          f"({100 * d['sobrevive'].mean():.1f} %)")

    print("\n--- Departamentos (tras agrupar los de menos de "
          f"{MIN_POR_DEPARTAMENTO}) ---")
    print(d.groupby("departamento")["sobrevive"]
          .agg(n="size", tasa="mean").sort_values("n", ascending=False)
          .assign(tasa=lambda t: (100 * t["tasa"]).round(1)).to_string())

    X, y = matriz(d)
    print(f"\n--- Matriz de diseno: {X.shape[0]} x {X.shape[1]} ---")
    print(", ".join(X.columns))

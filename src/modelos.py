"""
Modelos de predicción de `venta_neta` semanal: suavizado exponencial, ARIMA con
regresores y regresión lineal sobre el calendario. Todos se evalúan con origen
móvil en las mismas particiones que los baselines, para que los MAE sean
comparables.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import features

# Órdenes ARMA que se prueban.
ORDENES = {
    "arima_xreg_100": {"order": (1, 0, 0), "sin_tendencia": False},
    "arima_xreg_101": {"order": (1, 0, 1), "sin_tendencia": False},
    "arima_xreg_011": {"order": (0, 1, 1), "sin_tendencia": True},
}

# Descripción breve de cada modelo, para que la tabla de resultados se lea
# sin salir del notebook.
PROCEDENCIA = {
    # Los baselines dicen qué regla son, no solo de qué ficha salen: la columna
    # se lee en la tabla sin tener que ir a buscar la validación.
    "naive_1": "listón ingenuo: la semana anterior",
    "naive_52": "listón ingenuo: la misma semana de la temporada anterior",
    "constante_mediana": "listón ingenuo: el nivel del entrenamiento",
    "ses": "suavizado exponencial simple",
    "holt": "Holt (tendencia)",
    "holt_winters_52": "Holt-Winters aditivo",
    "holt_winters_52_mul": "Holt-Winters multiplicativo",
    "arima_xreg_100": "arima() con xreg",
    "arima_xreg_101": "arima() con xreg",
    "brown": "alisado doble de Brown",
    "arima_xreg_011": "arima() con xreg",
    "ols_spline": "lineal con spline cíclico",
    "ols_sin_ciclo": "calendario completo, 11 regresores",
    "ols_sin_acopio": "el completo sin el término de acopio (ablación histórica)",
    "ols_basico": "el modelo elegido",
    "ols_basico_sin_acopio": "el modelo elegido sin el término de acopio (ablación)",
    "ols_promo": "calendario de folleto sobre el completo",
    "ols_promo_desfase": "el mismo con la fase corrida: prueba de falsación (histórico)",
    "ols_promo_basico": "calendario de folleto sobre el modelo elegido",
    "ols_promo_desfase_basico": "el mismo con la fase corrida: prueba de falsación",
    "desc_ols": "tickets × ticket medio, cada uno con OLS",
    "desc_ses": "tickets × ticket medio, cada uno con ses",
}


# ---------------------------------------------------------------------------
# Suavizado exponencial
# ---------------------------------------------------------------------------

def predecir_suavizado(y_train, h: int, tipo: str,
                       estacional: str = "add") -> np.ndarray:
    """Familia de suavizado exponencial."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if tipo == "brown":
            return _brown(np.asarray(y_train, dtype=float), h)
        if tipo == "ses":
            ajuste = SimpleExpSmoothing(y_train, initialization_method="estimated").fit()
        elif tipo == "holt":
            ajuste = Holt(y_train, initialization_method="estimated").fit()
        elif tipo == "holt_winters":
            if estacional not in ("add", "mul"):
                raise ValueError(f"esquema estacional desconocido: {estacional!r}")
            ajuste = ExponentialSmoothing(
                y_train, trend="add", seasonal=estacional,
                seasonal_periods=baselines.PERIODO_ANUAL,
                initialization_method="estimated",
            ).fit()
        else:
            raise ValueError(f"tipo de suavizado desconocido: {tipo!r}")
        return np.asarray(ajuste.forecast(h))


def _brown(y_train, h: int) -> np.ndarray:
    """Alisado exponencial doble de Brown."""
    from scipy.optimize import minimize_scalar

    def alisar(alpha):
        s1p = s2p = float(np.mean(y_train))
        s1 = np.empty(len(y_train))
        s2 = np.empty(len(y_train))
        for i, x in enumerate(y_train):
            s1p = alpha * x + (1 - alpha) * s1p
            s2p = alpha * s1p + (1 - alpha) * s2p
            s1[i], s2[i] = s1p, s2p
        return s1, s2

    def coeficientes(s1, s2, alpha, i):
        return 2 * s1[i] - s2[i], alpha / (1 - alpha) * (s1[i] - s2[i])

    def sce(alpha):
        s1, s2 = alisar(alpha)
        b0, b1 = coeficientes(s1, s2, alpha, np.arange(len(y_train) - 1))
        return float(((y_train[1:] - (b0 + b1)) ** 2).sum())

    # El límite superior no llega a 1: `b1` lleva 1-alpha en el denominador.
    alpha = minimize_scalar(sce, bounds=(0.01, 0.95), method="bounded").x
    s1, s2 = alisar(alpha)
    b0, b1 = coeficientes(s1, s2, alpha, len(y_train) - 1)
    return b0 + b1 * np.arange(1, h + 1)

# ---------------------------------------------------------------------------
# Modelo lineal sobre el calendario
# ---------------------------------------------------------------------------

def _utiles(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Quita las columnas que no varían dentro de esta partición."""
    vivas = X_train.columns[X_train.std(numeric_only=True) > 0]
    return X_train[vivas], X_test[vivas]


def predecir_ols(y_train, X_train, X_test, devolver_ajuste: bool = False):
    """OLS con intercepto sobre los regresores de calendario."""
    import statsmodels.api as sm

    X_train, X_test = _utiles(X_train, X_test)
    ajuste = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
    pred = np.asarray(ajuste.predict(sm.add_constant(X_test, has_constant="add")))
    return (pred, ajuste) if devolver_ajuste else pred


def cobertura_intervalo(serie: pd.DataFrame, horizonte: int = baselines.HORIZONTE,
                        alfa: float = 0.05) -> dict:
    """Qué parte de las semanas reales cae dentro del intervalo de predicción."""
    import statsmodels.api as sm

    y = np.asarray(serie["venta_neta"], dtype=float)
    X = matrices(serie)[BASE_XREG]
    dentro, total = 0, 0
    for corte in baselines.origenes_particiones(len(y), horizonte, n_particiones=len(y)):
        X_tr, X_te = _utiles(X.iloc[:corte], X.iloc[corte:corte + horizonte])
        ajuste = sm.OLS(y[:corte], sm.add_constant(X_tr, has_constant="add")).fit()
        ic = ajuste.get_prediction(sm.add_constant(X_te, has_constant="add")) \
                   .summary_frame(alpha=alfa)
        real = y[corte:corte + horizonte]
        dentro += int(((real >= ic["obs_ci_lower"].to_numpy()) &
                       (real <= ic["obs_ci_upper"].to_numpy())).sum())
        total += len(real)
    return {"cobertura_pct": 100 * dentro / total, "semanas": total}


# ---------------------------------------------------------------------------
# arima() con regresores externos (xreg)
# ---------------------------------------------------------------------------

def predecir_arima_xreg(y_train, X_train, X_test, order, sin_tendencia: bool,
                     devolver_ajuste: bool = False):
    """`arima()` con los regresores de calendario como `xreg`."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if sin_tendencia:
        X_train = X_train.drop(columns=["tendencia"])
        X_test = X_test.drop(columns=["tendencia"])
    X_train, X_test = _utiles(X_train, X_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ajuste = SARIMAX(
            y_train,
            exog=X_train,
            order=order,
            trend="c" if not sin_tendencia else "n",
            # True por defecto; con False los d=0 fallan igual (ver
            # `diagnostico_raiz_unitaria`).
            enforce_stationarity=True,
            enforce_invertibility=True,
            # Con 1.000 convergen todas salvo una de `arima_xreg_101`.
        ).fit(disp=False, maxiter=1000)
    pred = np.asarray(ajuste.get_forecast(steps=len(X_test), exog=X_test).predicted_mean)
    return (pred, ajuste) if devolver_ajuste else pred


# ---------------------------------------------------------------------------
# Descomposición `tickets` × `ticket medio`
# ---------------------------------------------------------------------------

# Descomposición de la venta en número de tickets por ticket medio: separa un
# problema de afluencia de uno de gasto por visita.
DESCOMPOSICION = {
    "desc_ols": {"metodo": "ols"},
    "desc_ses": {"metodo": "ses"},
}


def predecir_descomposicion(serie: pd.DataFrame, corte: int, h: int,
                            X_train, X_test, metodo: str) -> np.ndarray:
    """Predice `tickets` y `ticket_medio` por separado y devuelve su producto."""
    tickets = serie["tickets"].to_numpy(float)[:corte]
    # El ticket medio se deriva, no se lee: no viene en el informe Z.
    ticket_medio = (serie["venta_neta"].to_numpy(float)[:corte] / tickets)

    if metodo == "ols":
        pred_t = predecir_ols(tickets, X_train, X_test)
        pred_tm = predecir_ols(ticket_medio, X_train, X_test)
    elif metodo == "ses":
        pred_t = predecir_suavizado(tickets, h, "ses")
        pred_tm = predecir_suavizado(ticket_medio, h, "ses")
    else:
        raise ValueError(f"método de descomposición desconocido: {metodo!r}")

    return np.asarray(pred_t) * np.asarray(pred_tm)


# ---------------------------------------------------------------------------
# Predicción fuera de muestra"prediction"`)
# ---------------------------------------------------------------------------

def predecir_futuro(serie: pd.DataFrame, h: int = baselines.HORIZONTE,
                    escenario: str = "cierre",
                    alpha: float = 0.05) -> pd.DataFrame:
    """Predice las `h` semanas siguientes al final de la serie, con intervalo."""
    import statsmodels.api as sm

    festivos = features.cargar_festivos_escenario(escenario)
    inicios_obs = serie["periodo_inicio"]
    origen = pd.to_datetime(inicios_obs).min()

    # Entrenamiento: las 135 semanas y el estado real de sus festivos.
    X_obs = features.construir(inicios_obs, features.cargar_festivos(), origen=origen,
                               leads=(), ventana_pascua=0)

    # Futuro: mismas columnas, mismo origen de tendencia.
    inicios_fut = features.semanas_futuras(serie, h)
    X_fut = features.construir(inicios_fut, festivos, origen=origen,
                               leads=(), ventana_pascua=0)[X_obs.columns]

    ajuste = sm.OLS(serie["venta_neta"].to_numpy(float),
                    sm.add_constant(X_obs, has_constant="add")).fit()
    resumen = (ajuste.get_prediction(sm.add_constant(X_fut, has_constant="add"))
               .summary_frame(alpha=alpha))

    return pd.DataFrame({
        "periodo_inicio": inicios_fut.to_numpy(),
        "periodo_fin": (inicios_fut + pd.Timedelta(days=6)).to_numpy(),
        "prediccion": resumen["mean"].to_numpy(),
        "lo": resumen["obs_ci_lower"].to_numpy(),
        "hi": resumen["obs_ci_upper"].to_numpy(),
        "horas_cierre_festivo": X_fut["horas_cierre_festivo"].to_numpy(),
    })


def futuro_por_escenario(serie: pd.DataFrame, h: int = baselines.HORIZONTE,
                         alpha: float = 0.05) -> pd.DataFrame:
    """Los dos escenarios de horario juntos, y la diferencia entre ellos."""
    a = predecir_futuro(serie, h, "cierre", alpha).set_index("periodo_inicio")
    b = predecir_futuro(serie, h, "manana", alpha).set_index("periodo_inicio")

    return pd.DataFrame({
        "periodo_fin": a["periodo_fin"],
        "cierre": a["prediccion"],
        "manana": b["prediccion"],
        "brecha": b["prediccion"] - a["prediccion"],
        "lo": a["lo"],
        "hi": b["hi"],
        "horas_cierre_festivo": a["horas_cierre_festivo"],
    })


CARPETA_POSTERIORES = Path(__file__).resolve().parents[1] / "datos_bruto" / "Informes Z septiembre 2026"


def contrastar_futuro(serie: pd.DataFrame, carpeta: Path = CARPETA_POSTERIORES,
                      h: int = 10) -> pd.DataFrame:
    """La previsión de `predecir_futuro` frente a la venta real que llegó después."""
    import parser_cierre_z as pz

    fut = predecir_futuro(serie, h=h, escenario="cierre")
    fut["periodo_inicio"] = pd.to_datetime(fut["periodo_inicio"])
    fut = fut.set_index("periodo_inicio")
    filas = []
    for ruta in sorted(Path(carpeta).glob("*.pdf")):
        datos = pz.parsear(ruta)
        hueco = pz.validar(datos)["continuidad_documentos"]["documentos_que_faltan"]
        if hueco:
            raise ValueError(f"{ruta.name}: faltan {hueco} documentos")
        inicio = pd.Timestamp(ruta.stem[:10])
        p = fut.loc[inicio]
        filas.append({"periodo_inicio": inicio, "real": datos["venta_neta"],
                      "tickets": datos["tickets"], "prediccion": p["prediccion"],
                      "lo": p["lo"], "hi": p["hi"],
                      "horas_cierre_festivo": p["horas_cierre_festivo"]})
    r = pd.DataFrame(filas).set_index("periodo_inicio")
    r["error"] = r["real"] - r["prediccion"]
    r["dentro"] = r["real"].between(r["lo"], r["hi"])
    return r

# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def matrices(serie: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Las dos bases estacionales, construidas una vez sobre la serie entera."""
    inicios = serie["periodo_inicio"]
    festivos = features.cargar_festivos()
    X = {
        base: features.construir(inicios, festivos, estacionalidad=base)
        for base in ("spline", "ninguna")
    }
    # Las dos versiones con calendario de folleto: la real y la de fase
    # corrida, que existe solo para falsar a la primera.
    for promo in ("campana", "campana_desfase"):
        X[promo] = features.construir(inicios, festivos, promocion=promo)
    # Ablación: el modelo sin el término de acopio de fiesta local, solo para
    # medir cuánto aporta ese término (§5 del notebook). El modelo lo lleva.
    X["sin_acopio"] = features.construir(inicios, festivos, fiesta_local=False)
    # El modelo elegido: solo las 5 variables significativas al 5 % de la tabla
    # de coeficientes (constante, tendencia, horas_cierre_festivo, la semana de
    # Semana Santa y fiesta_local_acopio) - quita las 6 que no lo eran (las dos
    # de proximidad a festivo y los otros cuatro términos de Pascua): medidas
    # en las mismas 76 particiones, esas 6 no mejoran la predicción fuera de
    # muestra, ni juntas ni por separado.
    X["basico"] = features.construir(inicios, festivos, leads=(), ventana_pascua=0)
    # Ablación del término de acopio dentro del modelo básico (para no depender
    # solo de la ablación equivalente sobre el modelo completo, más arriba).
    X["basico_sin_acopio"] = features.construir(
        inicios, festivos, leads=(), ventana_pascua=0, fiesta_local=False)
    # Las dos versiones del calendario de folleto, pero sobre el modelo básico
    # en vez del completo: mismo control de falsación, base consistente con el
    # modelo elegido.
    for promo in ("campana", "campana_desfase"):
        X[f"{promo}_basico"] = features.construir(
            inicios, festivos, promocion=promo, leads=(), ventana_pascua=0)
    return X


# Qué modelos entran en cada familia.
FAMILIAS = {
    "baselines": ["naive_1", "naive_52"],
    "suavizado": ["ses", "holt", "holt_winters_52", "holt_winters_52_mul"],
    # `ols_basico` es el modelo elegido; `ols_sin_ciclo` se queda como
    # comparación histórica (comprobar_cifras.py y la figura de comparación de
    # modelos lo necesitan en la misma familia).
    "regresion": ["ols_sin_ciclo", "ols_basico", "ols_spline"],
    "arima": list(ORDENES),
    "descomposicion": list(DESCOMPOSICION),
}

BASE_XREG = "basico"

# Con qué base estacional se ajusta cada modelo lineal.
BASE_OLS = {"ols_sin_ciclo": "ninguna", "ols_spline": "spline",
            "ols_promo": "campana", "ols_promo_desfase": "campana_desfase",
            "ols_sin_acopio": "sin_acopio", "ols_basico": "basico",
            "ols_basico_sin_acopio": "basico_sin_acopio",
            "ols_promo_basico": "campana_basico",
            "ols_promo_desfase_basico": "campana_desfase_basico"}
# Cada suavizado con su configuración completa, no solo el tipo: el esquema
# estacional es parte de la definición del modelo y tiene que viajar con él.
SUAVIZADO = {
    "ses": {"tipo": "ses"},
    "brown": {"tipo": "brown"},
    "holt": {"tipo": "holt"},
    "holt_winters_52": {"tipo": "holt_winters", "estacional": "add"},
    "holt_winters_52_mul": {"tipo": "holt_winters", "estacional": "mul"},
}


def evaluar(serie: pd.DataFrame, familias=tuple(FAMILIAS),
            horizonte: int = baselines.HORIZONTE,
            origenes=None) -> pd.DataFrame:
    """Origen móvil sobre las familias pedidas, en los mismos cortes de la
    validación.
    """
    queridos = []
    for f in familias:
        queridos.extend(FAMILIAS.get(f, [f]))

    y = np.asarray(serie["venta_neta"], dtype=float)
    X = matrices(serie)
    if origenes is None:
        origenes = baselines.origenes_particiones(len(y), horizonte,
                                                  n_particiones=len(y))

    filas, fallos = [], []
    for corte in origenes:
        y_train, y_test = y[:corte], y[corte : corte + horizonte]
        h = len(y_test)
        # Escala del MASE: MAE del naïve-1 dentro del train, nunca del test.
        escala = float(np.abs(np.diff(y_train)).mean())
        corte_x = {k: (v.iloc[:corte], v.iloc[corte : corte + horizonte]) for k, v in X.items()}

        pred = {n: p for n, p in baselines._predicciones(y, corte, h).items()
                if n in queridos}

        for nombre in queridos:
            try:
                if nombre in SUAVIZADO:                       #
                    pred[nombre] = predecir_suavizado(y_train, h, **SUAVIZADO[nombre])
                elif nombre in BASE_OLS:                      # -2
                    pred[nombre] = predecir_ols(y_train, *corte_x[BASE_OLS[nombre]])
                elif nombre in DESCOMPOSICION:
                    pred[nombre] = predecir_descomposicion(
                        serie, corte, h, *corte_x[BASE_XREG],
                        DESCOMPOSICION[nombre]["metodo"])
                elif nombre in ORDENES:                       #
                    cfg = ORDENES[nombre]
                    pred[nombre] = predecir_arima_xreg(
                        y_train, *corte_x[BASE_XREG], cfg["order"], cfg["sin_tendencia"])
            except Exception as e:
                fallos.append({"origen": int(corte), "modelo": nombre,
                               "n_train": int(corte), "error": type(e).__name__})

        for nombre, p in pred.items():
            filas.append({"origen": int(corte), "n_train": int(corte), "modelo": nombre,
                          **baselines._metricas(y_test, p, escala)})

    detalle = pd.DataFrame(filas)
    # Los fallos viajan como lista de diccionarios, no como DataFrame: pandas
    # compara los `.attrs` de los objetos al concatenar, y comparar dos
    # DataFrames ahí levanta "truth value is ambiguous".
    detalle.attrs["fallos"] = fallos
    return detalle


def fallos_de(detalle: pd.DataFrame) -> pd.DataFrame:
    """Los ajustes que no se pudieron hacer, como tabla."""
    fallos = detalle.attrs.get("fallos") or []
    if not fallos:
        return pd.DataFrame(columns=["origen", "modelo", "n_train", "error"])
    return (pd.DataFrame(fallos)
            .groupby(["modelo", "error"])
            .agg(particiones_fallidas=("origen", "size"),
                 n_train_maximo_que_falla=("n_train", "max")))


def comparables(detalle: pd.DataFrame, verboso: bool = True) -> pd.DataFrame:
    """Recorta el detalle a las particiones donde ajustaron todos los modelos."""
    presentes = detalle["modelo"].unique()
    por_modelo = detalle.groupby("modelo")["origen"].apply(set)
    comunes = set.intersection(*por_modelo)

    if verboso:
        total = detalle["origen"].nunique()
        if len(comunes) == total:
            print(f"Los {len(presentes)} modelos ajustaron en las {total} particiones: "
                  "la tabla ya era comparable.")
        else:
            culpables = sorted(m for m in presentes if len(por_modelo[m]) < total)
            print(f"Recortado de {total} a {len(comunes)} particiones para que las "
                  f"{len(presentes)} filas sean comparables.")
            print(f"  Manda el más limitado: {', '.join(culpables)}")
    return detalle[detalle["origen"].isin(comunes)].copy()


def tabla_comparativa(detalle: pd.DataFrame) -> pd.DataFrame:
    """Todos los modelos en una sola tabla, y comparables."""
    total = detalle["origen"].nunique()
    ajusta_en_todas = (detalle.groupby("modelo")["origen"].nunique() == total)

    todas = detalle.groupby("modelo").agg(
        todas_mae=("mae", "median"), todas_peor=("mae", "max"))
    # Se tapan las celdas de quien no ajustó en todas: enseñarlas invita a
    # compararlas con las de al lado, que es justo lo que no se puede hacer.
    todas.loc[~ajusta_en_todas, ["todas_mae", "todas_peor"]] = np.nan

    comun = comparables(detalle, verboso=False).groupby("modelo").agg(
        comun_mae=("mae", "median"), comun_peor=("mae", "max"),
        comun_rmse=("rmse", "median"))

    tabla = todas.join(comun)
    tabla["particiones"] = detalle.groupby("modelo")["origen"].nunique()
    tabla["procedencia"] = [PROCEDENCIA.get(m, "") for m in tabla.index]
    # Ordena por la columna que manda; los de NaN caen al final, que es
    # donde tienen que estar: no se han podido medir en igualdad.
    return tabla.sort_values("todas_mae", na_position="last")


def resumir(detalle: pd.DataFrame, n_particiones: int | None = None) -> pd.DataFrame:
    """La tabla que va a la memoria: mediana, media, peor partición y victorias."""
    if n_particiones is None:
        n_particiones = detalle["origen"].nunique()
    tabla = detalle.groupby("modelo").agg(
        mae_mediano=("mae", "median"), mae_medio=("mae", "mean"),
        peor_particion=("mae", "max"), rmse_mediano=("rmse", "median"),
        particiones=("mae", "size"),
    )
    ganados = detalle.loc[detalle.groupby("origen")["mae"].idxmin(), "modelo"].value_counts()
    tabla["ganados"] = ganados.reindex(tabla.index).fillna(0).astype(int)
    tabla["procedencia"] = [PROCEDENCIA.get(m, "") for m in tabla.index]
    return tabla.sort_values("mae_mediano")


def metricas(detalle: pd.DataFrame) -> pd.DataFrame:
    """Las tres métricas del TFM, mediana sobre las particiones."""
    orden = ["mae", "rmse", "mape"]
    tabla = detalle.groupby("modelo")[orden].median()
    tabla["particiones"] = detalle.groupby("modelo").size()
    return tabla.sort_values("mae")


def diagnostico_raiz_unitaria(serie: pd.DataFrame,
                              horizonte: int = baselines.HORIZONTE) -> pd.DataFrame:
    """El coeficiente AR(1) y el de tendencia del ARIMA (1,0,0) en cada partición."""
    y = np.asarray(serie["venta_neta"], dtype=float)
    X = matrices(serie)[BASE_XREG]
    filas = []
    for corte in baselines.origenes_particiones(len(y), horizonte, n_particiones=len(y)):
        _, ajuste = predecir_arima_xreg(
            y[:corte], X.iloc[:corte], X.iloc[corte : corte + horizonte],
            (1, 0, 0), False, devolver_ajuste=True,
        )
        p = dict(zip(ajuste.param_names, ajuste.params))
        filas.append({"origen": int(corte), "ar_L1": p.get("ar.L1"),
                      "coef_tendencia": p.get("tendencia")})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Metodología Box-Jenkins
# ---------------------------------------------------------------------------
# Box-Jenkins en ocho pasos: 1 leer, 2 dibujar, 3 decidir si hace falta
# transformar, 4 contrastar la estacionariedad, 5 identificar la estructura
# ARMA con la FAS y la FAP, 6 estimar, 7 diagnosticar y elegir, y 8 predecir.

def box_jenkins(serie: pd.DataFrame, nlags: int = 60,
                retardos_lb=(4, 8, 12, 24, 52)) -> dict:
    """Los pasos 3, 4, 5 y 7 de la metodología Box-Jenkins sobre `venta_neta`."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf, adfuller, pacf
    import statsmodels.api as sm

    y = np.asarray(serie["venta_neta"], dtype=float)
    n = len(y)
    fechas = pd.to_datetime(serie["periodo_inicio"])

    # Paso 3
    g = pd.Series(y).groupby(fechas.dt.to_period("M").to_numpy()).agg(["mean", "std", "count"])
    g = g[g["count"] >= 4]
    lx, ly = np.log(g["mean"]), np.log(g["std"])
    pendiente = float(np.polyfit(lx, ly, 1)[0])
    r_boxcox = float(np.corrcoef(lx, ly)[0, 1])

    # Paso 4
    adf_estad, adf_p = adfuller(y, autolag="AIC", regression="ct")[:2]

    # Paso 5
    t = sm.add_constant(np.arange(n, dtype=float))
    sin_tendencia = sm.OLS(y, t).fit().resid
    Xb = matrices(serie)["basico"]
    residuo = sm.OLS(y, sm.add_constant(Xb.to_numpy(float))).fit().resid
    lb_residuo = acorr_ljungbox(residuo, lags=[12, 24, 52], return_df=True)["lb_pvalue"]
    identificacion = pd.DataFrame({
        "fas_sin_tendencia": acf(sin_tendencia, nlags=nlags)[1:],
        "fap_sin_tendencia": pacf(sin_tendencia, nlags=nlags, method="ldb")[1:],
        "fas_residuo": acf(residuo, nlags=nlags)[1:],
        "fap_residuo": pacf(residuo, nlags=nlags, method="ldb")[1:],
    }, index=pd.RangeIndex(1, nlags + 1, name="retardo"))

    # Paso 7
    X = matrices(serie)[BASE_XREG]
    diag, ajustes = [], {}
    for nombre, cfg in ORDENES.items():
        _, aj = predecir_arima_xreg(y, X, X.iloc[-1:], cfg["order"], cfg["sin_tendencia"],
                                    devolver_ajuste=True)
        p, dd, q = cfg["order"]
        res = np.asarray(aj.resid)[dd:]
        lb = acorr_ljungbox(res, lags=list(retardos_lb), model_df=p + q, return_df=True)
        ajustes[nombre] = aj
        fila = {"modelo": nombre, "orden": str(cfg["order"]),
                "converge": bool(aj.mle_retvals.get("converged", True))}
        fila.update({f"ljung_box_p_{k}": float(v) for k, v in zip(retardos_lb, lb["lb_pvalue"])})
        diag.append(fila)

    return {
        "n": n,
        "banda": 1.96 / np.sqrt(n),
        "boxcox_pendiente": pendiente,
        "boxcox_r": r_boxcox,
        "boxcox_meses": len(g),
        "adf_estadistico": float(adf_estad),
        "adf_p": float(adf_p),
        "identificacion": identificacion,
        "sin_tendencia": sin_tendencia,
        "residuo": residuo,
        "diagnostico": pd.DataFrame(diag).set_index("modelo"),
        "ljung_box_residuo": {int(k): float(v) for k, v in lb_residuo.items()},
        "ajustes": ajustes,
    }


if __name__ == "__main__":
    serie = features.cargar_serie()

    print("=" * 78)
    print(f"Progresion de modelos · {len(serie)} semanas, horizonte "
          f"{baselines.HORIZONTE}")
    print("=" * 78)

    detalle = evaluar(serie)
    n = detalle["origen"].nunique()
    print(f"\nOrigen movil · {n} particiones, las mismas de la validación\n")
    tabla = resumir(detalle)
    print(tabla.round({"mae_mediano": 0, "mae_medio": 0, "peor_particion": 0,
                       "rmse_mediano": 0}).to_string())

    print("\n--- Métricas, mediana sobre las particiones ---")
    print(metricas(detalle).round(
        {"mae": 0, "rmse": 0, "mape": 2}).to_string())

    fallos = fallos_de(detalle)
    if len(fallos):
        print("\n--- Modelos que no se pudieron ajustar en alguna particion ---")
        print(fallos.to_string())

    print("\n--- Por que fallan los ARIMA con xreg en niveles (d=0) ---")
    d = diagnostico_raiz_unitaria(serie)
    print(f"  ar.L1 por particion:            {d['ar_L1'].min():+.3f} a {d['ar_L1'].max():+.3f}"
          "    <- pegado a la raiz unitaria")
    print(f"  coef. tendencia (EUR/semana): {d['coef_tendencia'].min():+.0f} a "
          f"{d['coef_tendencia'].max():+.0f}    <- inestable, y se extrapola")

    detalle.to_csv("datos_limpio/evaluacion_paso1.csv", index=False)
    print("\nDetalle por particion -> datos_limpio/evaluacion_paso1.csv")

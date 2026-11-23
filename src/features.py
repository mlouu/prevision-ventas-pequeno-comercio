"""
Regresores deterministas de la serie semanal: tendencia, ciclo anual, horas de
apertura perdidas por festivos y Semana Santa. Solo entra lo que se conoce de
antemano por el calendario, así que se pueden generar filas futuras.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.easter import easter
from patsy import build_design_matrices, dmatrix

# Raíz del proyecto, deducida de la posición de este fichero.
RAIZ = Path(__file__).resolve().parents[1]

# Horario de la tienda, constante desde la apertura: lunes a viernes
# de 9:00 a 14:00 y de 17:00 a 20:00, sábados de 9:00 a 14:00, domingos cerrado.
HORAS_NORMALES = {0: 8, 1: 8, 2: 8, 3: 8, 4: 8, 5: 5, 6: 0}  # 0 = lunes
HORAS_SEMANA_COMPLETA = sum(HORAS_NORMALES.values())  # 45 h

# Horas que abre un festivo según su estado en datos_externos/festivos_tienda.csv.
# `manana` son las 4 horas de 9:30 a 13:30, caiga en el día que caiga.
HORAS_FESTIVO = {"cerrado": 0.0, "cerrado_fijo": 0.0, "manana": 4.0}

# Periodo anual en semanas: 365,2425 / 7. no es 52.
PERIODO_ANUAL = 365.2425 / 7


def cargar_serie(ruta=RAIZ / "datos_limpio/serie_semanal.parquet") -> pd.DataFrame:
    """La serie semanal ordenada, con `periodo_inicio` ya como fecha."""
    s = pd.read_parquet(ruta).sort_values("periodo_inicio").reset_index(drop=True)
    s["periodo_inicio"] = pd.to_datetime(s["periodo_inicio"])
    return s


def cargar_festivos(ruta=RAIZ / "datos_externos/festivos_tienda.csv") -> pd.DataFrame:
    """El calendario de festivos, solo con las fechas de estado conocido."""
    f = pd.read_csv(ruta, parse_dates=["fecha"])
    return f[f["apertura"].isin(HORAS_FESTIVO)].copy()


def horas_cierre_festivo_por_semana(inicios, festivos: pd.DataFrame) -> pd.Series:
    """Horas de apertura que cada semana pierde por festivos."""
    inicios = pd.to_datetime(pd.Series(inicios)).reset_index(drop=True)

    # Lo que cuesta un festivo = lo que abriría un día normal en ese día de la
    # semana, menos lo que abre de hecho.
    coste = [
        HORAS_NORMALES[fecha.weekday()] - HORAS_FESTIVO[estado]
        for fecha, estado in zip(festivos["fecha"], festivos["apertura"])
    ]
    lunes = festivos["fecha"] - pd.to_timedelta(festivos["fecha"].dt.weekday, unit="D")
    por_semana = pd.Series(coste, index=lunes).groupby(level=0).sum()

    return pd.Series(
        por_semana.reindex(inicios).fillna(0.0).to_numpy(), name="horas_cierre_festivo"
    )


def semanas_a_semana_santa(inicios) -> pd.Series:
    """Semanas con signo desde la que contiene el Domingo de Resurrección."""
    inicios = pd.to_datetime(pd.Series(inicios)).reset_index(drop=True)

    def lunes_de_pascua(anio: int) -> pd.Timestamp:
        d = pd.Timestamp(easter(anio))
        return d - pd.Timedelta(days=d.weekday())

    ref = inicios.dt.year.map(lunes_de_pascua)
    return ((inicios - ref).dt.days / 7).round().astype(int)


def semana_del_anio(inicios) -> pd.Series:
    """Semana ISO del año, 1-53."""
    inicios = pd.to_datetime(pd.Series(inicios)).reset_index(drop=True)
    return inicios.dt.isocalendar().week.astype(int)


def semanas_acopio_fiesta_local(inicios, festivos: pd.DataFrame,
                                semanas_antes: int = 2) -> pd.Series:
    """Marca 0/1 las semanas de acopio previas a la primera fiesta local del año."""
    inicios = pd.to_datetime(pd.Series(inicios)).reset_index(drop=True)
    marca = np.zeros(len(inicios), dtype=int)
    if "ambito" not in festivos.columns:
        return pd.Series(marca, name="fiesta_local_acopio")
    locales = festivos[festivos["ambito"] == "local"]
    if locales.empty:
        return pd.Series(marca, name="fiesta_local_acopio")
    for _, grupo in locales.groupby(locales["fecha"].dt.year):
        primera = grupo["fecha"].min()
        lunes_fiesta = primera - pd.Timedelta(days=primera.weekday())
        for k in range(1, semanas_antes + 1):
            marca[(inicios == lunes_fiesta - pd.Timedelta(weeks=k)).to_numpy()] = 1
    return pd.Series(marca, name="fiesta_local_acopio")


# Calendario de campañas promocionales (reconstruido y corregido
# con los folletos oficiales).
#
# La regla se cumple, pero no con una sola fase. Los folletos publicados
# (`datos_externos/folletos_coviran.csv`) muestran que la Navidad de 2024
# corrió el ciclo una semana: el folleto oficial de la zona suroriental (la de
# Almería) del 28 de mayo de 2024 va una semana por detrás del de 12 de agosto
# de 2025. Así que el calendario son tres tramos:
#
# - hasta el 25 de noviembre de 2024, el ciclo anclado en el folleto oficial
#   del 28 de mayo de 2024;
# - del 26 de noviembre de 2024 al 13 de enero de 2025, los folletos de
#   Navidad tal como se publicaron (fuentes secundarias, las menos firmes);
# - desde el 14 de enero de 2025, el ciclo anclado en el folleto oficial del
#   12 de agosto de 2025.
#
# `verificar_folletos()` comprueba que cada folleto del CSV cae exactamente en
# una campaña del calendario.
DIAS_CAMPANA = 12
CICLO_CAMPANA = 14
RUTA_FOLLETOS = RAIZ / "datos_externos/folletos_coviran.csv"
ANCLA_2024 = pd.Timestamp("2024-05-28")      # folleto oficial, suroriental
ANCLA_ACTUAL = pd.Timestamp("2025-08-12")    # folleto oficial, suroriental
INICIO_NAVIDAD_2024 = pd.Timestamp("2024-11-26")
FIN_NAVIDAD_2024 = pd.Timestamp("2025-01-13")
# Compatibilidad: el ancla del ciclo vigente.
ANCLA_CAMPANA = ANCLA_ACTUAL


def cargar_folletos(ruta=RUTA_FOLLETOS) -> pd.DataFrame:
    """Los folletos publicados que se han podido fechar, con su fuente."""
    return pd.read_csv(ruta, parse_dates=["inicio", "fin"])


def _en_ciclo(dias: pd.DatetimeIndex, ancla: pd.Timestamp) -> np.ndarray:
    fase = np.mod((dias - ancla).days, CICLO_CAMPANA)
    return np.asarray(fase) <= DIAS_CAMPANA - 1


def dias_de_campana(dias, desfase: int = 0) -> np.ndarray:
    """Máscara booleana: qué días caen dentro de una campaña de folleto."""
    dias = pd.DatetimeIndex(pd.to_datetime(pd.Series(dias))) - pd.Timedelta(days=desfase)
    navidad = np.zeros(len(dias), dtype=bool)
    f = cargar_folletos()
    f = f[(f["inicio"] >= INICIO_NAVIDAD_2024) & (f["inicio"] <= FIN_NAVIDAD_2024)]
    for ini, fin in zip(f["inicio"], f["fin"]):
        navidad |= np.asarray((dias >= ini) & (dias <= fin))
    return np.where(dias < INICIO_NAVIDAD_2024, _en_ciclo(dias, ANCLA_2024),
                    np.where(dias <= FIN_NAVIDAD_2024, navidad,
                             _en_ciclo(dias, ANCLA_ACTUAL)))


def verificar_folletos() -> pd.DataFrame:
    """Cada folleto publicado contra el calendario: ¿coincide día a día?"""
    f = cargar_folletos().copy()
    ok = []
    for ini, fin in zip(f["inicio"], f["fin"]):
        fin = fin if pd.notna(fin) else ini + pd.Timedelta(days=DIAS_CAMPANA - 1)
        m = dias_de_campana(pd.date_range(ini - pd.Timedelta(days=1), fin + pd.Timedelta(days=1)))
        ok.append(bool((not m[0]) and m[1:-1].all() and (not m[-1])))
    f["coincide"] = ok
    return f


def campana_por_semana(inicios, desfase: int = 0,
                       solo_abiertos: bool = True) -> pd.Series:
    """Días de campaña de cada semana natural, sobre 6 abiertos o sobre 7."""
    inicios = pd.to_datetime(pd.Series(inicios)).reset_index(drop=True)
    fuera = []
    for ini in inicios:
        dias = pd.date_range(ini, periods=7, freq="D")
        dentro = dias_de_campana(dias, desfase)
        if solo_abiertos:
            dentro = dentro & (dias.dayofweek != 6)   # domingo cerrado
        fuera.append(int(dentro.sum()))
    return pd.Series(fuera, index=inicios, name="dias_campana")


def base_estacional_spline(inicios, df: int = 5, diseno=None):
    """Spline cúbico cíclico sobre la semana del año (`patsy.cc`)."""
    semana = semana_del_anio(inicios).to_numpy()
    if diseno is None:
        X = dmatrix(f"cc(s, df={df}) - 1", {"s": semana}, return_type="dataframe")
        return X, X.design_info
    X = build_design_matrices([diseno], {"s": semana}, return_type="dataframe")[0]
    return X, diseno


def construir(
    inicios,
    festivos: pd.DataFrame | None = None,
    leads: tuple[int, ...] = (1, 2),
    origen: pd.Timestamp | None = None,
    ventana_pascua: int = 2,
    estacionalidad: str = "ninguna",
    promocion: str = "ninguna",
    fiesta_local: bool = True,
    df_spline: int = 5,
    diseno_spline=None,
    devolver_diseno: bool = False,
):
    """Matriz de regresores deterministas para las semanas dadas."""
    inicios = pd.to_datetime(pd.Series(inicios)).reset_index(drop=True)
    if festivos is None:
        festivos = cargar_festivos()
    if origen is None:
        origen = inicios.min()

    X = pd.DataFrame(index=pd.DatetimeIndex(inicios, name="periodo_inicio"))
    dias = (inicios - origen).dt.days.to_numpy()

    # --- Tendencia -------------------------------------------------------
    # En semanas desde el origen, no un contador de filas: así una matriz
    # construida para el futuro sigue la misma recta que la de entrenamiento.
    X["tendencia"] = dias / 7

    # --- Ciclo anual -----------------------------------------------------
    diseno = None
    if estacionalidad == "spline":
        base, diseno = base_estacional_spline(inicios, df_spline, diseno_spline)
        for i, col in enumerate(base.columns):
            X[f"spl{i + 1}"] = base[col].to_numpy()
    elif estacionalidad != "ninguna":
        raise ValueError(f"estacionalidad desconocida: {estacionalidad!r}")

    # --- Exposición ------------------------------------------------------
    perdidas = horas_cierre_festivo_por_semana(inicios, festivos).to_numpy()
    X["horas_cierre_festivo"] = perdidas
    # Adelantos: el acopio ocurre antes del festivo (ver docstring del módulo).
    for salto in leads:
        X[f"horas_cierre_festivo_prox{salto}"] = np.concatenate(
            [perdidas[salto:], np.zeros(salto)]
        )

    # --- Semana Santa ----------------------------------------------------
    dist = semanas_a_semana_santa(inicios).to_numpy()
    for d in range(-ventana_pascua, ventana_pascua + 1):
        etiqueta = f"pascua_m{abs(d)}" if d < 0 else f"pascua_p{d}"
        X[etiqueta] = (dist == d).astype(int)

    # --- Acopio de las fiestas locales de septiembre -------------------
    # Un término propio para la semana previa a la fiesta del pueblo, que
    # `horas_cierre_festivo_prox*` infravalora mucho. Ver el helper: ajustado a 2 años.
    if fiesta_local:
        X["fiesta_local_acopio"] = semanas_acopio_fiesta_local(
            inicios, festivos).to_numpy()

    # --- Promoción -------------------------------------------------------
    # Solo la versión en días de apertura tiene varianza; la de 7 días es
    # constante en las 135 semanas. Ver `campana_por_semana`.
    if promocion != "ninguna":
        if promocion == "campana":
            desfase = 0
        elif promocion == "campana_desfase":
            desfase = 7
        else:
            raise ValueError(f"promoción desconocida: {promocion!r}")
        X["dias_campana"] = campana_por_semana(inicios, desfase).to_numpy()

    return (X, diseno) if devolver_diseno else X


# ---------------------------------------------------------------------------
# Predicción fuera de muestra: las semanas que todavía no han ocurrido
# ---------------------------------------------------------------------------
# El backtesting evalúa los modelos en pliegues dentro de la ventana observada;
# no da ninguna cifra para las semanas posteriores al final de la serie.

# Estado que se supone a los festivos `pendiente` en cada escenario.
ESCENARIOS = {
    "cierre": "cerrado",   # extremo pesimista: el festivo cierra el día entero
    "manana": "manana",    # extremo optimista: abre de 9:30 a 13:30
}


def cargar_festivos_escenario(escenario: str, ruta=None) -> pd.DataFrame:
    """El calendario completo, resolviendo las `pendiente` con un supuesto."""
    if escenario not in ESCENARIOS:
        raise ValueError(f"escenario desconocido: {escenario!r}. "
                         f"Opciones: {sorted(ESCENARIOS)}")
    ruta = ruta or RAIZ / "datos_externos/festivos_tienda.csv"
    f = pd.read_csv(ruta, parse_dates=["fecha"])
    f.loc[~f["apertura"].isin(HORAS_FESTIVO), "apertura"] = ESCENARIOS[escenario]
    return f


def semanas_futuras(serie: pd.DataFrame, h: int = 4) -> pd.Series:
    """Los `h` lunes siguientes al final de la serie (`h=4` por defecto, el
    horizonte al que el modelo está validado).
    """
    ultimo_fin = pd.to_datetime(serie["periodo_fin"]).max()
    primer_lunes = ultimo_fin + pd.Timedelta(days=1)
    if primer_lunes.weekday() != 0:
        raise ValueError(
            f"la serie no termina en domingo: {ultimo_fin.date()}. "
            "Las semanas deben ser naturales de lunes a domingo."
        )
    return pd.Series(pd.date_range(primer_lunes, periods=h, freq="7D"),
                     name="periodo_inicio")


if __name__ == "__main__":
    serie = cargar_serie()
    festivos = cargar_festivos()
    X = construir(serie["periodo_inicio"], festivos)

    print("=" * 70)
    print(f"Matriz de regresores · {X.shape[0]} semanas x {X.shape[1]} columnas")
    print("=" * 70)
    print("\ncolumnas:", ", ".join(X.columns))

    print(f"\n--- Exposicion (semana completa = {HORAS_SEMANA_COMPLETA} h) ---")
    afectadas = X[X["horas_cierre_festivo"] > 0]
    print(f"semanas con horas perdidas: {len(afectadas)} de {len(X)} "
          f"({100 * len(afectadas) / len(X):.1f} %)")
    print(afectadas["horas_cierre_festivo"].value_counts().sort_index().to_string())

    print("\n--- La Pascua se mueve, y Fourier no la sigue ---")
    for a in sorted(serie["periodo_inicio"].dt.year.unique()):
        print(f"  {a}: Domingo de Resurreccion {easter(a)}")

    print("\n--- Correlacion con el objetivo (descriptivo, no evaluativo) ---")
    y = serie["venta_neta"].to_numpy()
    corr = X.apply(lambda c: np.corrcoef(c, y)[0, 1] if c.std() > 0 else np.nan)
    print(corr.sort_values(key=abs, ascending=False).round(3).to_string())

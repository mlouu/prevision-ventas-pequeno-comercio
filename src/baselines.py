"""
Baselines de la serie semanal (naive_1, naive_52 y nivel constante), medidos en
muestra y con origen móvil. Son el listón con el que se compara cualquier
modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Horizonte de 4 semanas: la tienda pide a una semana o a un mes vista, no a
# seis meses.
HORIZONTE = 4

# Estacionalidad anual en datos semanales.
PERIODO_ANUAL = 52


def _metricas(real: np.ndarray, pred: np.ndarray, mae_naive_train: float | None) -> dict:
    """Las seis medidas de error habituales."""
    error = real - pred
    mase = (float(np.abs(error).mean() / mae_naive_train)
            if mae_naive_train else float("nan"))
    return {
        "me": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mpe": float(100 * np.mean(error / real)),
        "mape": float(100 * np.mean(np.abs(error / real))),
        "mase": mase,
    }


def en_muestra(y: pd.Series) -> pd.DataFrame:
    """Error de los tres baselines sobre la serie entera, sin separar test."""
    y = pd.Series(y).reset_index(drop=True)
    errores = {
        "naive_1": (y - y.shift(1)).abs(),
        "naive_52": (y - y.shift(PERIODO_ANUAL)).abs(),
        "constante_mediana": (y - y.median()).abs(),
    }
    filas = [
        {
            "baseline": nombre,
            "mae": float(e.mean()),
            "pct_media": float(100 * e.mean() / y.mean()),
            "n": int(e.notna().sum()),
        }
        for nombre, e in errores.items()
    ]
    return pd.DataFrame(filas)


def _predicciones(y: np.ndarray, corte: int, h: int) -> dict[str, np.ndarray]:
    """Predicción de cada baseline para las `h` semanas siguientes a `corte`."""
    train = y[:corte]
    return {
        "naive_1": np.repeat(train[-1], h),
        "naive_52": y[corte - PERIODO_ANUAL: corte - PERIODO_ANUAL + h],
        "constante_mediana": np.repeat(np.median(train), h),
    }


def origenes_particiones(n: int, horizonte: int = HORIZONTE, n_particiones: int = 5) -> np.ndarray:
    """Los cortes de la validación de origen móvil."""
    ultimo = n - horizonte
    # El primer origen tiene que dejar sitio al naïve-52 y a un train con
    # fondo: 52 semanas para que el ingenuo estacional tenga con qué comparar,
    # más el horizonte.
    primero = PERIODO_ANUAL + horizonte
    if primero > ultimo:
        raise ValueError(
            f"serie demasiado corta: {n} semanas no dan para horizonte "
            f"{horizonte} con estacionalidad {PERIODO_ANUAL}"
        )
    return np.unique(np.linspace(primero, ultimo, n_particiones).astype(int))


def origen_movil(
    y: pd.Series,
    horizonte: int = HORIZONTE,
    n_particiones: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling origin con ventana expansiva, el protocolo de la validación."""
    y = np.asarray(pd.Series(y).reset_index(drop=True), dtype=float)
    origenes = origenes_particiones(len(y), horizonte, n_particiones)

    filas = []
    for corte in origenes:
        train, test = y[:corte], y[corte: corte + horizonte]
        # Escala de MASE: MAE del naïve-1 dentro del entrenamiento, nunca del test.
        mae_naive_train = float(np.abs(np.diff(train)).mean())
        for nombre, pred in _predicciones(y, corte, len(test)).items():
            filas.append({
                "origen": int(corte),
                "n_train": int(corte),
                "baseline": nombre,
                **_metricas(test, pred, mae_naive_train),
            })

    detalle = pd.DataFrame(filas)
    resumen = (
        detalle.groupby("baseline")
        .agg(mae_medio=("mae", "mean"), mae_min=("mae", "min"),
             mae_max=("mae", "max"), mape_medio=("mape", "mean"),
             particiones=("mae", "size"))
        .sort_values("mae_medio")
        .reset_index()
    )
    return detalle, resumen


if __name__ == "__main__":
    serie = pd.read_parquet("datos_limpio/serie_semanal.parquet").sort_values("periodo_inicio")
    y = serie["venta_neta"].reset_index(drop=True)

    print("=" * 68)
    print(f"Baselines · {len(y)} semanas, {serie['periodo_inicio'].min().date()} -> "
          f"{serie['periodo_fin'].max().date()}")
    print("=" * 68)

    print("\nEn muestra (descriptivo, no evaluativo)")
    print(en_muestra(y).round(2).to_string(index=False))

    # La validación usa todos los cortes posibles, no una muestra de 5: con
    # n_particiones alto, `origen_movil` los agota y devuelve los 32 que caben.
    detalle, resumen = origen_movil(y, n_particiones=len(y))
    n_particiones = detalle["origen"].nunique()
    print(f"\nOrigen movil (horizonte {HORIZONTE} semanas, {n_particiones} particiones)")

    # La ficha de la validación se cita por mediana, media, peor particion y particiones
    # ganados: se imprime en ese formato para poder cotejarla de un vistazo.
    tabla = (detalle.groupby("baseline")["mae"]
             .agg(mae_mediano="median", mae_medio="mean", peor_particion="max"))
    ganados = detalle.loc[detalle.groupby("origen")["mae"].idxmin(), "baseline"].value_counts()
    tabla["particiones_ganadas"] = ganados.reindex(tabla.index).fillna(0).astype(int)
    print(tabla.sort_values("mae_mediano").round(0).to_string())

    peor = detalle.loc[detalle["mae"].idxmax()]
    print(f"\nPeor particion de todas: {peor['baseline']} con {peor['mae']:,.0f} EUR "
          f"(origen {int(peor['origen'])}).")
    print("Es el argumento de la validación: el nivel constante no es el mas certero,")
    print("es el unico que nunca revienta.")

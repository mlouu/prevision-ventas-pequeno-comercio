"""
Procesa en lote todos los informes brutos del TPV en `datos_bruto/Diario de
ventas por artículo acumulado mensual/`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parser_tpv import parsear, validar

# Umbral de tolerancia frente a los totales declarados por el propio informe.
TOLERANCIA_UNIDADES = 1.0
TOLERANCIA_VENTA = 1.0


def _granularidad(meta: dict) -> str:
    """Distingue semanal de mensual por la duración del periodo cubierto."""
    inicio, fin = meta.get("periodo_inicio"), meta.get("periodo_fin")
    if not inicio or not fin:
        return "desconocida"
    dias = (
        pd.to_datetime(fin, format="%d/%m/%Y") - pd.to_datetime(inicio, format="%d/%m/%Y")
    ).days
    return "semanal" if dias <= 10 else "mensual"


def procesar_todos(
    carpeta_bruto: str | Path = "datos_bruto/Diario de ventas por artículo acumulado mensual",
    carpeta_limpio: str | Path = "datos_limpio",
) -> dict[str, pd.DataFrame]:
    """Parsea todos los CSV de `carpeta_bruto` y puebla `carpeta_limpio`."""
    carpeta_bruto = Path(carpeta_bruto)
    carpeta_limpio = Path(carpeta_limpio)
    carpeta_limpio.mkdir(exist_ok=True)

    ficheros = sorted(f for f in carpeta_bruto.iterdir() if f.suffix.lower() == ".csv")
    if not ficheros:
        raise FileNotFoundError(f"No hay CSV en {carpeta_bruto}")

    por_granularidad: dict[str, list[pd.DataFrame]] = {}
    informe: list[dict] = []

    for ruta in ficheros:
        try:
            df, meta = parsear(ruta)
        except Exception as exc:
            informe.append({"fichero": ruta.name, "error": str(exc), "ok": False})
            print(f"  [Error] {ruta.name}: {exc}")
            continue

        gran = _granularidad(meta)
        df["granularidad"] = gran
        por_granularidad.setdefault(gran, []).append(df)

        val = validar(df, meta)
        diff_unidades = (val.get("unidades") or {}).get("diferencia")
        diff_venta = (val.get("total_venta") or {}).get("diferencia")
        n_anomalias = len(meta.get("anomalias", []))
        ok = (
            diff_unidades is not None and abs(diff_unidades) <= TOLERANCIA_UNIDADES
            and diff_venta is not None and abs(diff_venta) <= TOLERANCIA_VENTA
            and n_anomalias == 0
        )
        informe.append({
            "fichero": ruta.name,
            "periodo_inicio": meta.get("periodo_inicio"),
            "periodo_fin": meta.get("periodo_fin"),
            "granularidad": gran,
            "almacen": meta.get("almacen"),
            "articulos": len(df),
            "diff_unidades": diff_unidades,
            "diff_venta": diff_venta,
            "anomalias": n_anomalias,
            "ok": ok,
        })
        if not ok:
            print(f"  [Aviso] {ruta.name}: diff_unidades={diff_unidades} "
                  f"diff_venta={diff_venta} anomalias={n_anomalias}")

        df.to_csv(carpeta_limpio / f"{ruta.stem}.csv", index=False, encoding="utf-8-sig")

    tablas: dict[str, pd.DataFrame] = {}
    for gran, dfs in por_granularidad.items():
        tabla = pd.concat(dfs, ignore_index=True)
        almacenes = tabla["almacen"].dropna().unique()
        if len(almacenes) > 1:
            raise ValueError(
                f"El lote '{gran}' mezcla almacenes distintos: {sorted(almacenes)}. "
                "Revisa datos_bruto/: la tienda tiene más de un almacén y "
                "consolidarlos juntos falsearía la serie."
            )
        tabla.to_parquet(carpeta_limpio / f"consolidado_{gran}.parquet", index=False)
        tablas[gran] = tabla

    informe_df = pd.DataFrame(informe)
    informe_df.to_csv(carpeta_limpio / "informe_procesado.csv", index=False, encoding="utf-8-sig")

    n_ok = int(informe_df["ok"].sum()) if "ok" in informe_df else 0
    print(f"\nProcesados {len(ficheros)} ficheros: {n_ok} OK, {len(ficheros) - n_ok} con avisos/errores.")
    print(f"Detalle en {carpeta_limpio / 'informe_procesado.csv'}")

    return tablas


if __name__ == "__main__":
    procesar_todos()

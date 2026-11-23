"""
Agrupación de los 856 artículos fijos (presentes los 31 meses): por
rentabilidad y por la forma de su serie mensual. No es análisis de cesta: el
dato no tiene tickets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]

MESES_COMPLETOS = 31
# Duplicada a propósito en src/supervivencia.py (misma constante, mismo valor):
# si cambia aquí, cambiar también allí.
SEMILLA = 42
# Arranques de k-means: `kmeans(datos, centers = 4, nstart = 25)` en R.
INICIOS_KMEANS = 25


def cargar_fijos(ruta=RAIZ / "datos_limpio/consolidado_mensual.parquet"):
    """El panel de los artículos presentes en los 31 meses (artículos fijos)."""
    m = pd.read_parquet(ruta)
    a = m[m["tipo_fila"] == "articulo"].copy()
    a["mes"] = a["periodo_inicio"].dt.to_period("M")

    presencia = a.groupby("cod_articulo")["mes"].nunique()
    fijos = presencia[presencia == MESES_COMPLETOS].index
    largo = a[a["cod_articulo"].isin(fijos)].copy()

    meta = (largo.sort_values("mes")
            .groupby("cod_articulo")
            .agg(descripcion=("descripcion", "last"),
                 departamento=("departamento", "last"),
                 venta_total=("total_venta", "sum"),
                 unidades_total=("unidades", "sum")))
    return largo, meta


def matriz_formas(largo: pd.DataFrame, variable: str = "unidades"):
    """Matriz 856 x 31 con cada serie tipificada - el paso que decide todo."""
    crudas = (largo.pivot_table(index="cod_articulo", columns="mes",
                                values=variable, aggfunc="sum")
              .sort_index(axis=1))

    media = crudas.mean(axis=1).to_numpy()[:, None]
    desv = crudas.std(axis=1).to_numpy()[:, None]
    # Una serie plana tiene desviación 0 y su forma es indefinible: se deja en
    # ceros en vez de dividir por cero, y la comprobación de abajo la cuenta.
    desv = np.where(desv == 0, 1.0, desv)

    formas = pd.DataFrame((crudas.to_numpy() - media) / desv,
                          index=crudas.index, columns=crudas.columns)
    return formas, crudas


def elegir_k(formas, ks=range(2, 11)):
    """Codo y silueta para elegir k."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = formas.to_numpy()
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=INICIOS_KMEANS, random_state=SEMILLA).fit(X)
        filas.append({"k": k, "inercia": km.inertia_,
                      "silueta": silhouette_score(X, km.labels_)})
    return pd.DataFrame(filas).set_index("k")


def agrupar(formas, k: int):
    """k-means sobre las formas tipificadas."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=INICIOS_KMEANS, random_state=SEMILLA).fit(formas.to_numpy())
    etiquetas = pd.Series(km.labels_, index=formas.index, name="grupo")
    centros = pd.DataFrame(km.cluster_centers_, columns=formas.columns)
    centros.index.name = "grupo"
    return etiquetas, centros


def pca(formas, n: int = 2):
    """PCA sobre las formas."""
    from sklearn.decomposition import PCA

    p = PCA(n_components=n, random_state=SEMILLA).fit(formas.to_numpy())
    proy = pd.DataFrame(p.transform(formas.to_numpy()),
                        index=formas.index,
                        columns=[f"CP{i + 1}" for i in range(n)])
    return proy, p


def perfilar(etiquetas, meta, crudas) -> pd.DataFrame:
    """Qué es cada grupo en términos de negocio."""
    d = meta.loc[etiquetas.index].copy()
    d["grupo"] = etiquetas
    d["mes_pico"] = crudas.loc[etiquetas.index].idxmax(axis=1).astype(str)

    perfil = d.groupby("grupo").agg(
        articulos=("descripcion", "size"),
        venta=("venta_total", "sum"),
        venta_media=("venta_total", "mean"),
        dpto_dominante=("departamento", lambda s: s.value_counts().index[0]),
        pico_mas_comun=("mes_pico", lambda s: s.value_counts().index[0]),
    )
    perfil["cuota_venta_pct"] = 100 * perfil["venta"] / perfil["venta"].sum()
    return perfil


# ---------------------------------------------------------------------------
# La segunda vía: agrupar por características, no por la forma cruda
# ---------------------------------------------------------------------------
# `matriz_formas` trata cada serie como un punto en 31 dimensiones.

MESES_VERANO = (6, 7, 8)
MES_NAVIDAD = (12,)


def matriz_caracteristicas(crudas: pd.DataFrame) -> pd.DataFrame:
    """Cinco descriptores por artículo, todos con lectura directa."""
    X = crudas.to_numpy(float)
    meses = pd.PeriodIndex(crudas.columns)
    media = X.mean(axis=1)
    seguro = np.where(media == 0, 1.0, media)
    t = np.arange(X.shape[1])

    car = pd.DataFrame(index=crudas.index)
    car["cv"] = X.std(axis=1) / seguro
    car["tendencia"] = np.array([np.polyfit(t, fila, 1)[0] for fila in X]) / seguro
    car["idx_verano"] = X[:, np.isin(meses.month, MESES_VERANO)].mean(axis=1) / seguro
    car["idx_navidad"] = X[:, np.isin(meses.month, MES_NAVIDAD)].mean(axis=1) / seguro
    top3 = np.sort(X, axis=1)[:, -3:].sum(axis=1)
    total = X.sum(axis=1)
    car["concentracion"] = top3 / np.where(total == 0, 1.0, total)

    return car.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def escalar(car: pd.DataFrame) -> pd.DataFrame:
    """Tipificado antes de agrupar."""
    from sklearn.preprocessing import StandardScaler

    return pd.DataFrame(StandardScaler().fit_transform(car),
                        index=car.index, columns=car.columns)


# ---------------------------------------------------------------------------
# Los 14 departamentos - el dendrograma que sí se lee entero
# ---------------------------------------------------------------------------

def cargar_departamentos(ruta=RAIZ / "datos_limpio/serie_semanal_departamentos.parquet"):
    """Las series semanales de departamento, solo las 14 completas."""
    s = pd.read_parquet(ruta)
    completos = s.groupby("departamento").size()
    completos = completos[completos == completos.max()].index
    ancho = (s[s["departamento"].isin(completos)]
             .pivot_table(index="departamento", columns="periodo_inicio",
                          values="importe", aggfunc="sum")
             .sort_index(axis=1))
    return ancho


def dendrograma(ancho: pd.DataFrame, metodo: str = "ward"):
    """Enlace jerárquico sobre las series tipificadas."""
    from scipy.cluster.hierarchy import linkage

    z = ancho.sub(ancho.mean(axis=1), axis=0).div(ancho.std(axis=1), axis=0)
    return linkage(z.to_numpy(), method=metodo)


# ---------------------------------------------------------------------------
# Parte 1 · Agrupación por rentabilidad
# ---------------------------------------------------------------------------

def surtido_de(codigos) -> pd.DataFrame:
    """Las filas artículo-mes de `SURTIDO` de esos artículos, con el código
    como texto en `cod`.
    """
    import comprobar_cifras as cc  # import diferido: ese módulo importa este

    sur = cc._surtido(cc.Contexto())
    sur = sur.assign(cod=sur["cod_articulo"].astype(str))
    return sur[sur["cod"].isin(pd.Index(codigos).astype(str))]


def cargar_rentabilidad(codigos) -> pd.DataFrame:
    """Venta, margen en euros y margen % de cada artículo sobre `SURTIDO`, en
    los 31 meses.
    """
    sur = surtido_de(codigos)
    g = sur.groupby("cod")
    r = pd.DataFrame({
        "meses_con_margen": g["periodo_inicio"].nunique(),
        "venta": g["total_venta"].sum(),
        "margen_euros": g["beneficio_tc"].sum(),
    })
    r["margen_pct"] = 100 * r["margen_euros"] / r["venta"]
    r.index.name = "cod_articulo"
    return r


def matriz_rentabilidad(r: pd.DataFrame) -> pd.DataFrame:
    """Las dos variables de la agrupación por rentabilidad."""
    if (r["margen_euros"] <= 0).any():
        raise ValueError("Hay artículos con margen en euros no positivo: "
                         "el logaritmo no está definido.")
    return pd.DataFrame({"margen_pct": r["margen_pct"],
                         "log_margen_euros": np.log(r["margen_euros"])}, index=r.index)


def enlace_jerarquico(Z, metodo: str = "ward"):
    """Matriz de enlace del jerárquico aglomerativo."""
    from scipy.cluster.hierarchy import linkage

    return linkage(np.asarray(Z, dtype=float), method=metodo)


def saltos_de_union(enlace, hasta: int = 10) -> pd.DataFrame:
    """Dónde cortar el dendrograma: el salto en las distancias de unión."""
    alturas = np.sort(enlace[:, 2])[::-1]          # la última unión, primero
    filas = [{"k": k,
              "union_que_los_junta": alturas[k - 2],
              "union_anterior": alturas[k - 1],
              "salto": alturas[k - 2] - alturas[k - 1]}
             for k in range(2, hasta + 1)]
    return pd.DataFrame(filas).set_index("k")


def coeficiente_aglomeracion(enlace) -> float:
    """Coeficiente de aglomeración."""
    n = enlace.shape[0] + 1
    primera = np.full(n, np.nan)
    for a, b, h, _ in enlace:
        for x in (int(a), int(b)):
            if x < n and np.isnan(primera[x]):
                primera[x] = h
    return float(np.mean(1 - primera / enlace[:, 2].max()))


def colores_dendrograma(enlace, etiquetas, paleta, gris):
    """Cada rama del color de su grupo, como `fviz_dend(k = ...)` en R."""
    etiquetas = np.asarray(etiquetas)
    n = len(etiquetas)
    grupos = {i: {int(etiquetas[i])} for i in range(n)}
    for i, (a, b, _, _) in enumerate(enlace):
        grupos[n + i] = grupos[int(a)] | grupos[int(b)]

    def color(nodo):
        g = grupos[nodo]
        return paleta[next(iter(g))] if len(g) == 1 else gris

    return color


def altura_de_cierre(enlace, etiquetas) -> pd.Series:
    """A qué altura del dendrograma se completa cada grupo."""
    etiquetas = np.asarray(etiquetas)
    n = len(etiquetas)
    grupos = {i: {int(etiquetas[i])} for i in range(n)}
    cierre = {}
    for i, (a, b, h, _) in enumerate(enlace):
        g = grupos[int(a)] | grupos[int(b)]
        grupos[n + i] = g
        if len(g) == 1:
            k = next(iter(g))
            cierre[k] = max(cierre.get(k, 0.0), float(h))
    return pd.Series(cierre).sort_index().rename("altura_de_cierre")


def atipicos_caja(datos: pd.DataFrame) -> pd.DataFrame:
    """Atípicos por el criterio del diagrama de caja."""
    q1, q3 = datos.quantile(0.25), datos.quantile(0.75)
    iqr = q3 - q1
    return (datos > q3 + 1.5 * iqr).astype(int) - (datos < q1 - 1.5 * iqr).astype(int)


def gap_estadistico(X, ks=range(1, 11), B=100, semilla=SEMILLA):
    """Gap statistic."""
    from scipy.spatial.distance import pdist
    from sklearn.cluster import KMeans

    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(semilla)

    def log_w(M, k):
        if k == 1:
            et = np.zeros(len(M), dtype=int)
        else:
            et = KMeans(n_clusters=k, n_init=INICIOS_KMEANS, random_state=SEMILLA).fit(M).labels_
        return np.log(0.5 * sum(pdist(M[et == j]).sum() / (et == j).sum()
                                for j in np.unique(et)))

    centro = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - centro, full_matrices=False)
    rotados = (X - centro) @ vt.T
    lo, hi = rotados.min(axis=0), rotados.max(axis=0)

    filas = []
    for k in ks:
        obs = log_w(X, k)
        ref = np.array([log_w(rng.uniform(lo, hi, rotados.shape) @ vt + centro, k)
                        for _ in range(B)])
        filas.append({"k": k, "log_W": obs, "E_log_W": ref.mean(),
                      "gap": ref.mean() - obs,
                      "error_tipico": ref.std(ddof=1) * np.sqrt(1 + 1 / B)})
    tabla = pd.DataFrame(filas).set_index("k")

    f, se = tabla["gap"].to_numpy(), tabla["error_tipico"].to_numpy()
    decrece = np.diff(f) <= 0
    nc = int(np.argmax(decrece)) if decrece.any() else len(f) - 1
    elegido = int(np.argmax(f[: nc + 1] >= f[nc] - se[nc]))
    return tabla, int(tabla.index[elegido])


def jerarquico(Z, k: int, metodo: str = "ward"):
    """Enlace jerárquico y corte en k grupos."""
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage
    from scipy.spatial.distance import pdist

    X = np.asarray(Z, dtype=float)
    enlace = linkage(X, method=metodo)
    etiquetas = pd.Series(fcluster(enlace, k, criterion="maxclust") - 1,
                          index=getattr(Z, "index", None), name="grupo")
    return etiquetas, cophenet(enlace, pdist(X))[0]


def coincidencia(a: pd.Series, b: pd.Series) -> float:
    """Qué parte de los artículos cae en el mismo grupo en dos particiones."""
    comunes = a.index.intersection(b.index)
    return float((a.loc[comunes] == b.loc[comunes]).mean())


def emparejar(referencia: pd.Series, otra: pd.Series) -> pd.Series:
    """Renumera `otra` para que cada grupo lleve el número del grupo de
    `referencia` con el que comparte más artículos.
    """
    from scipy.optimize import linear_sum_assignment

    comunes = referencia.index.intersection(otra.index)
    tabla = pd.crosstab(otra.loc[comunes], referencia.loc[comunes])
    filas, columnas = linear_sum_assignment(-tabla.to_numpy())
    mapa = {tabla.index[f]: tabla.columns[c] for f, c in zip(filas, columnas)}
    return otra.map(mapa).rename(otra.name)


TIPOS_MIXTURA = ("full", "tied", "diag", "spherical")


def mixturas_bic(Z, ks=range(1, 10), tipos=TIPOS_MIXTURA, semilla: int = SEMILLA):
    """BIC de una mixtura de normales para cada número de grupos y forma -
    `Mclust` de R.
    """
    from sklearn.mixture import GaussianMixture

    X = np.asarray(Z, dtype=float)
    tabla = pd.DataFrame(
        {tipo: [GaussianMixture(k, covariance_type=tipo, n_init=10,
                                random_state=semilla).fit(X).bic(X) for k in ks]
         for tipo in tipos},
        index=pd.Index(list(ks), name="k"),
    )
    k, tipo = tabla.stack().idxmin()
    return tabla, (int(k), tipo)


def mixtura(Z, k: int, tipo: str = "full", semilla: int = SEMILLA):
    """Mixtura de `k` normales: el grupo de cada artículo y la probabilidad con
    la que pertenece a él.
    """
    from sklearn.mixture import GaussianMixture

    X = np.asarray(Z, dtype=float)
    modelo = GaussianMixture(k, covariance_type=tipo, n_init=10,
                             random_state=semilla).fit(X)
    indice = getattr(Z, "index", None)
    etiquetas = pd.Series(modelo.predict(X), index=indice, name="grupo")
    probabilidad = pd.Series(modelo.predict_proba(X).max(axis=1), index=indice,
                             name="probabilidad")
    return etiquetas, probabilidad


def particion_ward(r: pd.DataFrame, k: int = 4) -> pd.Series:
    """La partición de la agrupación sobre una base de rentabilidad `r`
    cualquiera: tipificar, Ward, cortar en `k` y numerar por margen en euros.
    """
    Z = escalar(matriz_rentabilidad(r))
    return ordenar_por_margen_euros(jerarquico(Z, k)[0], r)


def estabilidad_quitando_uno(r: pd.DataFrame, k: int = 4, n: int = 20,
                             semilla: int = 0) -> pd.Series:
    """Sensibilidad de la partición: se quita un artículo al azar, se rehace
    Ward y se mide la coincidencia con la partición original, `n` veces.
    """
    base = particion_ward(r, k)
    rng = np.random.default_rng(semilla)
    out = []
    for _ in range(n):
        quita = rng.choice(r.index, 1)
        e1 = base.drop(quita)
        e2 = particion_ward(r.drop(quita), k)
        out.append(coincidencia(e1, emparejar(e1, e2)))
    return pd.Series(out, name="coincidencia")


def estabilidad_sin_negativos(r: pd.DataFrame, k: int = 4) -> dict:
    """Sensibilidad de la partición a la regla de los artículos bajo coste: se
    rehace la base de margen quitando todas las filas con margen negativo sobre
    la tarifa (la lectura más directa de que la tienda nunca vende por debajo
    de coste, sin ninguna regla y sin `costo_medio`) y se compara con la
    partición vigente.
    """
    import clasificar_bajo_coste as cbc
    import comprobar_cifras as cc

    base = cbc.corregir_caja_a_granel(cc._base_margen(cc.Contexto()).copy())
    base["cod_articulo"] = base["cod_articulo"].astype(str)
    base = base[(base["beneficio_tc"] >= 0) & base["cod_articulo"].isin(r.index.astype(str))]
    g = base.groupby("cod_articulo").agg(venta=("total_venta", "sum"),
                                         margen_euros=("beneficio_tc", "sum"))
    r2 = r.copy()
    r2["venta"] = g["venta"].reindex(r.index.astype(str)).to_numpy()
    r2["margen_euros"] = g["margen_euros"].reindex(r.index.astype(str)).to_numpy()
    r2["margen_pct"] = 100 * r2["margen_euros"] / r2["venta"]
    e1, e2 = particion_ward(r, k), particion_ward(r2, k)
    return {
        "articulos_cambian": int(((r2["margen_pct"] - r["margen_pct"]).abs() > 0.01).sum()),
        "coincidencia": coincidencia(e1, emparejar(e1, e2)),
    }


def ordenar_por_margen_euros(etiquetas: pd.Series, r: pd.DataFrame) -> pd.Series:
    """Renumera los grupos de mayor a menor margen medio en euros por artículo."""
    orden = r.loc[etiquetas.index, "margen_euros"].groupby(etiquetas).mean()
    mapa = {g: i for i, g in enumerate(orden.sort_values(ascending=False).index)}
    return etiquetas.map(mapa).rename("grupo")


def perfilar_rentabilidad(etiquetas: pd.Series, r: pd.DataFrame,
                          meta: pd.DataFrame) -> pd.DataFrame:
    """Qué es cada grupo de rentabilidad en términos de tienda."""
    d = r.loc[etiquetas.index].copy()
    d["departamento"] = meta.set_axis(meta.index.astype(str)).loc[d.index, "departamento"]
    d["grupo"] = etiquetas
    p = d.groupby("grupo").agg(
        articulos=("venta", "size"),
        venta=("venta", "sum"),
        margen_euros=("margen_euros", "sum"),
        margen_mediano=("margen_pct", "median"),
        margen_euros_mediano=("margen_euros", "median"),
        dpto_dominante=("departamento", lambda s: s.value_counts().index[0]),
    )
    p["margen_pct"] = 100 * p["margen_euros"] / p["venta"]
    p["cuota_margen_euros_pct"] = 100 * p["margen_euros"] / p["margen_euros"].sum()
    p["cuota_venta_pct"] = 100 * p["venta"] / p["venta"].sum()
    return p[["articulos", "margen_pct", "margen_mediano", "margen_euros_mediano",
              "margen_euros", "cuota_margen_euros_pct", "cuota_venta_pct",
              "dpto_dominante"]]


# ---------------------------------------------------------------------------
# Parte 1 · Contraste con el margen que recomienda Covirán (§9bis)
# ---------------------------------------------------------------------------
# Dato de la tienda: la plataforma Covirán da un margen de referencia de 20-40 %
# en general y de solo ~10 % en básicos (leche y aceite, los ejemplos que dio
# la tienda).

PATRON_BASICOS = r"^(?:LECHE (?:ENTERA|SEMIDESNAT|DESNATAD|SEMI )|ACEITE (?:DE |OLIVA|GIRASOL))"
# Derivados y conservas: otro producto, con otro margen - no son el básico.
PATRON_NO_BASICO = (r"ATUN|SARDINA|CABALLA|MEJILLON|PALOTE|PAN DE ACEITE|"
                    r"BOLLO|MAGDALENA|ARROZ CON LECHE|CHOCOLATE")


def clasificar_basicos(sur: pd.DataFrame) -> pd.DataFrame:
    """Margen por artículo de todo `SURTIDO`, con la marca `basico`
    (leche/aceite sin derivados).
    """
    g = sur.groupby("cod_articulo" if "cod_articulo" in sur.columns else "cod").agg(
        descripcion=("descripcion", "last"),
        departamento=("departamento", "last"),
        venta=("total_venta", "sum"),
        margen_euros=("beneficio_tc", "sum"),
    )
    g["margen_pct"] = 100 * g["margen_euros"] / g["venta"]
    g["basico"] = (g["descripcion"].str.contains(PATRON_BASICOS, case=False, regex=True, na=False)
                   & ~g["descripcion"].str.contains(PATRON_NO_BASICO, case=False, regex=True, na=False))
    return g


def resumen_recomendacion_coviran(g: pd.DataFrame) -> pd.DataFrame:
    """Básicos contra el resto: mediana, margen ponderado y cobertura de la
    banda 20-40 %.
    """
    filas = []
    for nombre, sub in [("básicos (leche, aceite)", g[g["basico"]]),
                        ("resto del surtido", g[~g["basico"]])]:
        filas.append({
            "grupo": nombre,
            "articulos": len(sub),
            "venta": sub["venta"].sum(),
            "margen_pct_mediano": sub["margen_pct"].median(),
            "margen_pct_ponderado": 100 * sub["margen_euros"].sum() / sub["venta"].sum(),
            "en_banda_20_40_pct": 100 * sub["margen_pct"].between(20, 40).mean(),
        })
    return pd.DataFrame(filas).set_index("grupo")


def representativos_por_banda(g: pd.DataFrame, venta_minima: float = 500, n: int = 5) -> pd.DataFrame:
    """Los `n` artículos de más venta en cada banda de margen - la lista para
    la tienda.
    """
    sub = g[g["venta"] >= venta_minima].copy()
    bandas = [("básico (~10 %)", sub["basico"]),
              ("bajo, no básico (<20 %)", (~sub["basico"]) & (sub["margen_pct"] < 20)),
              ("banda recomendada (20-40 %)", (~sub["basico"]) & sub["margen_pct"].between(20, 40)),
              ("alto (>40 %)", (~sub["basico"]) & (sub["margen_pct"] > 40))]
    piezas = []
    for nombre, mask in bandas:
        top = sub[mask].nlargest(n, "venta").copy()
        top.insert(0, "banda", nombre)
        piezas.append(top)
    return pd.concat(piezas)[["banda", "descripcion", "departamento", "venta", "margen_pct"]]


# ---------------------------------------------------------------------------
# Parte 1 · De la agrupación a la tienda (notebook 05 §10)
# ---------------------------------------------------------------------------

def tarifa_congelada(largo: pd.DataFrame) -> pd.Series:
    """Artículos con la tarifa de coste quieta los 31 meses y sin ninguna
    compra ligada.
    """
    g = largo.assign(cod=largo["cod_articulo"].astype(str)).groupby("cod")
    return ((g["tarifa_costo"].nunique() == 1)
            & g["costo_ultimo"].apply(lambda s: (s.fillna(0) == 0).all())).rename("tarifa_congelada")


def margen_por_anio(sur: pd.DataFrame, grupo: pd.Series, meses=range(1, 8)) -> pd.DataFrame:
    """Margen % de cada grupo por año, sobre los mismos meses de cada año."""
    d = sur[sur["periodo_inicio"].dt.month.isin(list(meses))]
    d = d.assign(grupo=d["cod"].map(grupo), anio=d["periodo_inicio"].dt.year)
    s = d.groupby(["grupo", "anio"])[["beneficio_tc", "total_venta"]].sum()
    return (100 * s["beneficio_tc"] / s["total_venta"]).unstack()


def ultimos_12_meses(sur: pd.DataFrame, grupo: pd.Series) -> pd.DataFrame:
    """Venta y margen de cada grupo en los últimos 12 meses de la ventana."""
    desde = sur["periodo_inicio"].max() - pd.DateOffset(months=11)
    d = sur[sur["periodo_inicio"] >= desde]
    d = d.assign(grupo=d["cod"].map(grupo))
    t = d.groupby("grupo").agg(meses=("periodo_inicio", "nunique"),
                               venta=("total_venta", "sum"),
                               margen_euros=("beneficio_tc", "sum"))
    t["margen_pct"] = 100 * t["margen_euros"] / t["venta"]
    t["cuota_margen_euros_pct"] = 100 * t["margen_euros"] / t["margen_euros"].sum()
    t["un_punto_mas_euros"] = t["venta"] / 100
    return t

def articulos_revision_precio(sur: pd.DataFrame, grupo: pd.Series, g: int = 1,
                              n: int = 10, congeladas: pd.Series | None = None) -> pd.DataFrame:
    """Los `n` artículos de un grupo donde un punto de margen vale más."""
    desde = sur["periodo_inicio"].max() - pd.DateOffset(months=11)
    d = sur[(sur["periodo_inicio"] >= desde) & (sur["cod"].map(grupo) == g)]
    t = (d.sort_values("periodo_inicio")
          .groupby("cod")
          .agg(descripcion=("descripcion", "last"), departamento=("departamento", "last"),
               venta=("total_venta", "sum"), margen_euros=("beneficio_tc", "sum")))
    t["margen_pct"] = 100 * t["margen_euros"] / t["venta"]
    t["un_punto_mas_euros"] = t["venta"] / 100
    m = t["margen_pct"] / 100
    t["subida_precio_pct"] = 100 * ((1 - m) / (1 - m - 0.01) - 1)
    t["ficha_congelada"] = t.index.map(congeladas).fillna(False).astype(bool)         if congeladas is not None else False
    return t.sort_values("venta", ascending=False).head(n)


if __name__ == "__main__":
    largo, meta = cargar_fijos()
    formas, crudas = matriz_formas(largo)

    print("=" * 70)
    print(f"Articulos fijos  · {formas.shape[0]} articulos x {formas.shape[1]} meses")
    print("=" * 70)
    print(f"\ncuota de venta de los articulos fijos: "
          f"{100 * meta['venta_total'].sum() / largo['total_venta'].sum():.1f} % "
          "del propio grupo (control: debe dar 100)")
    print(f"series planas (desviacion 0): {int((crudas.std(axis=1) == 0).sum())}")

    print("\n--- Codo y silueta ---")
    print(elegir_k(formas).round(3).to_string())

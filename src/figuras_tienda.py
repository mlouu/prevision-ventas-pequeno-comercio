"""
Figuras del documento para la tienda, dibujadas como SVG elemento a elemento,
sin librería de gráficos, para que se regeneren desde `datos_limpio/`.
"""
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import comprobar_cifras as cc
import baselines
import modelos

RAIZ = Path(__file__).resolve().parents[1]
FIGURAS = RAIZ / "reports/figuras"

# --- Paleta, validada con scripts/validate_palette.js de la guía de visualización ---
PAPEL = "#f4f1ea"
TINTA = "#221f1a"
TINTA_SEC = "#544e42"
REJILLA = "#e2ddd0"
ACENTO = "#0b8466"      # 2026 / positivo / semana de acopio
ACENTO2 = "#c0562b"     # contraste / semana que cierra por fiesta
NEUTRO = "#93897a"      # referencia / contexto, siempre con etiqueta de valor
RAMPA_ANIOS = {2024: "#70a99b", 2025: "#2c967c", 2026: "#0b8466"}  # rampa ordinal, un solo tono

FUENTE = "Segoe UI, -apple-system, Helvetica Neue, Arial, sans-serif"
ANCHO, ALTO = 1600, 900
MARGEN = {"izq": 90, "der": 60, "arr": 190, "abj": 90}


def num(valor, dec=1, signo=False):
    formato = f"{{:{'+' if signo else ''},.{dec}f}}"
    crudo = formato.format(valor)
    return crudo.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def esc(texto):
    return html.escape(str(texto), quote=True)


class Lienzo:
    """Acumula elementos SVG y resuelve el encabezado editorial común a las
    seis figuras: mismo tipo, mismo tamaño, mismo sitio - la homogeneidad que
    pide visinf.
    """

    def __init__(self):
        self.el = []

    def add(self, s):
        self.el.append(s)

    def rect(self, x, y, w, h, fill, rx=0, opacity=1):
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" opacity="{opacity}"/>'
        )

    def linea(self, x1, y1, x2, y2, stroke, ancho=2, dash=None, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{ancho}"{d} opacity="{opacity}" '
            f'stroke-linecap="round"/>'
        )

    def polilinea(self, puntos, stroke, ancho=2.5, dash=None, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in puntos)
        self.add(
            f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
            f'stroke-width="{ancho}"{d} opacity="{opacity}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def circulo(self, cx, cy, r, fill, stroke=None, sw=0):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"{s}/>')

    def texto(self, x, y, contenido, size=15, color=TINTA, peso=400, ancla="start",
              familia=FUENTE, spacing=None, opacity=1):
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{familia}" '
            f'font-size="{size}" font-weight="{peso}" fill="{color}" '
            f'text-anchor="{ancla}"{ls} opacity="{opacity}">{esc(contenido)}</text>'
        )

    def texto_con_halo(self, x, y, contenido, size=15, color=TINTA, peso=700, ancla="middle"):
        """Como `texto`, pero con un respaldo del color de fondo detrás: evita
        que una línea de referencia u otra marca atraviese visualmente la
        etiqueta y parezca un tachado.
        """
        ancho_aprox = size * 0.56 * len(contenido)
        cx = {"start": x + ancho_aprox / 2, "middle": x, "end": x - ancho_aprox / 2}[ancla]
        self.rect(cx - ancho_aprox / 2 - 4, y - size * 0.78, ancho_aprox + 8, size * 1.0,
                   PAPEL, rx=3)
        self.texto(x, y, contenido, size=size, color=color, peso=peso, ancla=ancla)

    def texto_multilinea(self, x, y, lineas, size, color, peso, salto, ancla="start"):
        for i, linea in enumerate(lineas):
            self.texto(x, y + i * salto, linea, size=size, color=color, peso=peso, ancla=ancla)

    def encabezado(self, eyebrow, titular, subtitulo, ancho_max=64):
        """Ojo de mosca + titular grande + subtítulo, siempre en el mismo sitio."""
        self.texto(MARGEN["izq"], 44, eyebrow.upper(), size=13, color=ACENTO, peso=700,
                   spacing="1.6px")
        self.texto(MARGEN["izq"], 88, titular, size=34, color=TINTA, peso=700)
        palabras = subtitulo.split(" ")
        lineas, actual = [], ""
        for palabra in palabras:
            candidato = f"{actual} {palabra}".strip()
            if len(candidato) > ancho_max:
                lineas.append(actual)
                actual = palabra
            else:
                actual = candidato
        if actual:
            lineas.append(actual)
        self.texto_multilinea(MARGEN["izq"], 122, lineas, size=18, color=TINTA_SEC,
                              peso=400, salto=26)
        return 122 + len(lineas) * 26

    def nota(self, texto_nota, y=None):
        y = y or (ALTO - 16)
        self.texto(MARGEN["izq"], y, texto_nota, size=12.5, color=TINTA_SEC)

    def svg(self):
        cuerpo = "\n  ".join(self.el)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ANCHO} {ALTO}" '
            f'width="{ANCHO}" height="{ALTO}" font-family="{FUENTE}">\n'
            f'  <rect x="0" y="0" width="{ANCHO}" height="{ALTO}" fill="{PAPEL}"/>\n'
            f'  {cuerpo}\n'
            f'</svg>\n'
        )


def guardar(lienzo, nombre):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta_svg = FIGURAS / f"{nombre}.svg"
    ruta_svg.write_text(lienzo.svg(), encoding="utf-8")
    print(f"  {ruta_svg.name} escrito")
    return ruta_svg


def eje_y_lineal(c, y0, y1, minimo, maximo, n_ticks, fmt=lambda v: num(v, 0)):
    """Rejilla horizontal recesiva con dos o tres etiquetas, sin eje dibujado."""
    for i in range(n_ticks):
        valor = minimo + (maximo - minimo) * i / (n_ticks - 1)
        y = y1 - (y1 - y0) * i / (n_ticks - 1)
        c.linea(MARGEN["izq"], y, ANCHO - MARGEN["der"], y, REJILLA, ancho=1)
        c.texto(MARGEN["izq"] - 14, y + 5, fmt(valor), size=13, color=TINTA_SEC, ancla="end")


# =============================================================================
# Figura 1 · Venta nominal frente a venta real
# =============================================================================

def figura_1(ctx):
    s = ctx.completa[["periodo_inicio", "venta_neta", "venta_neta_real"]].copy()
    crec_nom = cc._crecimiento(ctx, "venta_neta", 2025, 2026)
    crec_real = cc._crecimiento_log(ctx, "venta_neta_real", 2025, 2026)

    c = Lienzo()
    y_ini = c.encabezado(
        "Evolución",
        "La tienda vende más, aunque parte es solo precio",
        f"Enero-julio de 2025 frente a enero-julio de 2026: +{num(crec_nom, 2)} % de venta "
        f"nominal; descontada la subida de precios, el crecimiento real es de +{num(crec_real)} puntos.",
    )

    x0, x1 = MARGEN["izq"], ANCHO - MARGEN["der"] - 130
    y0, y1 = ALTO - MARGEN["abj"], y_ini + 60
    fechas = pd.to_datetime(s["periodo_inicio"])
    t = (fechas - fechas.min()).dt.days.to_numpy(float)
    t_max = t.max()

    v_min = min(s["venta_neta"].min(), s["venta_neta_real"].min()) * 0.95
    v_max = max(s["venta_neta"].max(), s["venta_neta_real"].max()) * 1.05

    def px(ti):
        return x0 + (x1 - x0) * ti / t_max

    def py(v):
        return y0 - (y0 - y1) * (v - v_min) / (v_max - v_min)

    eje_y_lineal(c, y1, y0, v_min, v_max, 3, fmt=lambda v: f"{num(v, 0)} €")

    # Marcas de año
    for anio in [2024, 2025, 2026]:
        fecha_anio = pd.Timestamp(f"{anio}-01-01")
        if fecha_anio < fechas.min() or fecha_anio > fechas.max():
            continue
        ti = (fecha_anio - fechas.min()).days
        c.linea(px(ti), y0, px(ti), y1, REJILLA, ancho=1, dash="2,4")
        c.texto(px(ti), y0 + 26, str(anio), size=13, color=TINTA_SEC, ancla="middle")

    puntos_nom = [(px(ti), py(v)) for ti, v in zip(t, s["venta_neta"])]
    puntos_real = [(px(ti), py(v)) for ti, v in zip(t, s["venta_neta_real"])]
    c.polilinea(puntos_nom, TINTA_SEC, ancho=2, opacity=0.85)
    c.polilinea(puntos_real, ACENTO, ancho=3.5)

    c.circulo(*puntos_nom[-1], 4.5, TINTA_SEC)
    c.circulo(*puntos_real[-1], 4.5, ACENTO)
    # Las dos series terminan casi al mismo nivel (el crecimiento real es
    # modesto), así que sus etiquetas chocan si se colocan a ciegas junto a
    # cada punto.
    y_nom, y_real = puntos_nom[-1][1], puntos_real[-1][1]
    separacion_min = 34
    if abs(y_nom - y_real) < separacion_min:
        centro = (y_nom + y_real) / 2
        if y_nom <= y_real:
            y_nom, y_real = centro - separacion_min / 2, centro + separacion_min / 2
        else:
            y_nom, y_real = centro + separacion_min / 2, centro - separacion_min / 2
    lx = puntos_nom[-1][0] + 14
    c.texto(lx, y_nom + 5, "Nominal", size=15, color=TINTA_SEC, peso=600)
    c.texto(lx, y_real + 5, "Real", size=15, color=ACENTO, peso=700)
    c.texto(lx, y_real + 24, "(sin la inflación)", size=12.5, color=TINTA_SEC)

    c.nota("Venta semanal neta de impuestos · deflactada con el IPC de alimentos de la provincia, a impuestos constantes")
    return guardar(c, "06-01-nominal-vs-real")


# =============================================================================
# Figura 2 · Peso de cada familia, 2024 / 2025 / 2026
# =============================================================================

NOMBRES_DPTO = {
    "ULTRAMARINOS": "Ultramarinos", "BEBIDAS": "Bebidas", "CARNICERIA": "Carnicería",
    "FRUTAS Y VERDURA": "Frutas y\nverdura", "PANADERIA PASTEL": "Panadería",
    "LACTEOS": "Lácteos", "DROGUERIA Y PERF": "Droguería", "CHARCUTERIA CORT": "Charcutería",
    "CONGELADOS": "Congelados", "BAZAR": "Bazar", "OTROS": "Otros",
}


def figura_2():
    dep = pd.read_parquet(RAIZ / "datos_limpio/serie_semanal_departamentos.parquet")
    dep["periodo_inicio"] = pd.to_datetime(dep["periodo_inicio"])
    dep["anio"] = dep["periodo_inicio"].dt.year
    peso = dep.groupby(["anio", "departamento"])["importe"].sum().unstack("departamento")
    peso = peso.div(peso.sum(axis=1), axis=0) * 100
    orden = peso.loc[2026].sort_values(ascending=False)
    top, otros = orden.index[:6], orden.index[6:]
    tabla = peso[top].copy()
    tabla["OTROS"] = peso[otros].sum(axis=1)
    tabla = tabla[list(top) + ["OTROS"]]

    c = Lienzo()
    y_ini = c.encabezado(
        "Surtido",
        "La cesta apenas cambia — salvo panadería",
        f"Panadería y pastelería pasa del {num(tabla.loc[2024, 'PANADERIA PASTEL'])} % al "
        f"{num(tabla.loc[2026, 'PANADERIA PASTEL'])} % de la venta y gana peso todos "
        "los años: es el cambio más claro del periodo.",
    )

    # Una barra por familia que sale de una línea central, todas a la misma
    # escala, ordenadas de más ganancia a más pérdida.
    cambio = (tabla.loc[2026] - tabla.loc[2024]).sort_values(ascending=False)
    x_nombre = MARGEN["izq"]
    x_centro = 760
    medio_ancho = 300
    x_pesos = x_centro + medio_ancho + 140
    y_cab = y_ini + 62
    y_primera = y_cab + 52
    alto_fila = (ALTO - MARGEN["abj"] - 10 - y_primera) / len(cambio)
    alto_barra = alto_fila * 0.52
    escala = medio_ancho / float(cambio.abs().max())

    c.texto(x_centro, y_cab, "Cambio de 2024 a 2026, en puntos", size=14, color=TINTA_SEC,
            peso=600, ancla="middle")
    c.texto(x_pesos, y_cab, "Peso en la venta: 2024 → 2025 → 2026", size=14,
            color=TINTA_SEC, peso=600)
    c.linea(x_centro, y_primera - 18, x_centro, y_primera + alto_fila * len(cambio) - 18,
            TINTA_SEC, ancho=1.5)

    for i, (col, d) in enumerate(cambio.items()):
        yc = y_primera + alto_fila * i + alto_fila / 2 - 18
        resaltar = col == "PANADERIA PASTEL"
        peso_txt = 700 if resaltar else 600
        color = ACENTO if d >= 0.5 else ACENTO2 if d <= -0.5 else NEUTRO
        largo = abs(d) * escala
        bx = x_centro if d >= 0 else x_centro - largo
        c.rect(bx, yc - alto_barra / 2, largo, alto_barra, color, rx=3)
        etiqueta_cambio = f"{num(d, signo=True).replace('-', '−')}"
        if d >= 0:
            c.texto(x_centro + largo + 10, yc + 6, etiqueta_cambio, size=17, color=color,
                    peso=700)
        else:
            c.texto(x_centro - largo - 10, yc + 6, etiqueta_cambio, size=17, color=color,
                    peso=700, ancla="end")
        nombre = NOMBRES_DPTO.get(col, col.title()).replace("\n", " ")
        c.texto(x_nombre, yc + 6, nombre, size=18, color=TINTA, peso=peso_txt)
        pesos = " → ".join(f"{num(tabla.loc[a, col])} %" for a in (2024, 2025, 2026))
        c.texto(x_pesos, yc + 6, pesos, size=16, color=TINTA if resaltar else TINTA_SEC,
                peso=peso_txt)

    c.nota("% de la venta con IVA de la tienda, cierre Z semanal · 2026 llega hasta el 2 de agosto · "
           "verde, gana peso; naranja, lo pierde; gris, se mueve menos de medio punto")
    return guardar(c, "06-02-peso-familias")


# =============================================================================
# Figura 3 · Peso de la bebida fría sobre la venta, mes a mes
# =============================================================================

def figura_3(ctx):
    cuota = {m: cc._frio_cuota_mes(ctx, m) for m in range(1, 13)}
    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    valores = [cuota[m] for m in range(1, 13)]

    c = Lienzo()
    c.encabezado(
        "La nevera de bebidas frías",
        "Un negocio de verano, con la nevera puesta todo el año",
        "En julio se vende casi el triple de frío que en enero — y como la nevera está "
        "enchufada los doce meses, la curva mide al cliente, no la disponibilidad.",
    )

    x0, x1 = MARGEN["izq"], ANCHO - MARGEN["der"]
    y0, y1 = ALTO - MARGEN["abj"], 300
    n = 12
    barra_w = (x1 - x0) / n * 0.58
    paso = (x1 - x0) / n
    v_max = max(valores) * 1.32

    eje_y_lineal(c, y1, y0, 0, v_max, 3, fmt=lambda v: f"{num(v, 2)} %")

    for i, (mes, v) in enumerate(zip(meses, valores)):
        cx = x0 + paso * (i + 0.5)
        bh = (y0 - y1) * v / v_max
        color = ACENTO if i + 1 in (6, 7, 8, 9) else NEUTRO
        c.rect(cx - barra_w / 2, y0 - bh, barra_w, bh, color, rx=4)
        c.texto(cx, y0 - bh - 10, f"{num(v, 2)}", size=12.5, color=TINTA, peso=600, ancla="middle")
        c.texto(cx, y0 + 26, mes, size=13.5, color=TINTA_SEC, ancla="middle")

    c.nota("Peso sobre la venta total de artículos de cada mes del año, sumando los 31 meses (2024-01 a 2026-07) · verano (jun-sep) resaltado")
    return guardar(c, "06-03-frio-estacionalidad")


# =============================================================================
# Figura 4 · Predicción frente a venta real (backtest a un paso)
# =============================================================================

def figura_4(ctx):
    serie = ctx.completa.reset_index(drop=True)
    y = serie["venta_neta"].to_numpy(float)
    # El modelo elegido: 5 variables.
    X = modelos.matrices(serie)["basico"]
    origenes = baselines.origenes_particiones(len(y), horizonte=10, n_particiones=len(y))
    filas = []
    for corte in origenes:
        y_train = y[:corte]
        X_train, X_test = X.iloc[:corte], X.iloc[corte:corte + 1]
        pred = modelos.predecir_ols(y_train, X_train, X_test)
        filas.append({"fecha": serie["periodo_inicio"].iloc[corte], "real": y[corte],
                      "prediccion": float(pred[0])})
    bt = pd.DataFrame(filas)
    bt["error_abs"] = (bt["real"] - bt["prediccion"]).abs()
    mae = bt["error_abs"].median()
    media = bt["real"].mean()
    # La cifra oficial (4 semanas, 76 particiones) se calcula, no se escribe.
    oficial = modelos.tabla_comparativa(
        modelos.evaluar(ctx.completa, ["ols_basico"])).loc["ols_basico", "todas_mae"]

    c = Lienzo()
    y_ini = c.encabezado(
        "Predicción",
        "Así de bien sigue el modelo a la venta real",
        f"Cada punto se predice con una semana de antelación, sin ver el futuro: en una "
        f"semana típica se equivoca en {num(mae, 0)} €, sobre una venta media de "
        f"{num(media, 0)} €, en {len(bt)} semanas de prueba. A cuatro semanas vista, "
        f"el error típico es de {num(oficial, 0)} €.",
        ancho_max=78,
    )

    x0, x1 = MARGEN["izq"], ANCHO - MARGEN["der"] - 40
    y0, y1 = ALTO - MARGEN["abj"], y_ini + 50
    fechas = pd.to_datetime(bt["fecha"])
    t = (fechas - fechas.min()).dt.days.to_numpy(float)
    t_max = t.max()
    v_min = min(bt["real"].min(), bt["prediccion"].min()) * 0.95
    v_max = max(bt["real"].max(), bt["prediccion"].max()) * 1.05

    def px(ti):
        return x0 + (x1 - x0) * ti / t_max

    def py(v):
        return y0 - (y0 - y1) * (v - v_min) / (v_max - v_min)

    eje_y_lineal(c, y1, y0, v_min, v_max, 3, fmt=lambda v: f"{num(v, 0)} €")

    # Eje x: una marca al principio de cada trimestre que cae dentro del tramo.
    meses_abr = {1: "ene", 4: "abr", 7: "jul", 10: "oct"}
    for marca in pd.date_range(fechas.min().to_period("Q").to_timestamp(),
                               fechas.max(), freq="QS"):
        if marca < fechas.min():
            continue
        ti = (marca - fechas.min()).days
        c.linea(px(ti), y0, px(ti), y1, REJILLA, ancho=1, dash="2,4")
        c.texto(px(ti), y0 + 26, f"{meses_abr[marca.month]} {marca.year}", size=13,
                color=TINTA_SEC, ancla="middle")
    c.texto((x0 + x1) / 2, y0 + 52, "Semana de venta", size=13.5, color=TINTA_SEC,
            peso=600, ancla="middle")

    puntos_real = [(px(ti), py(v)) for ti, v in zip(t, bt["real"])]
    puntos_pred = [(px(ti), py(v)) for ti, v in zip(t, bt["prediccion"])]
    c.polilinea(puntos_real, TINTA_SEC, ancho=2.2, opacity=0.9)
    c.polilinea(puntos_pred, ACENTO, ancho=2.6, dash="7,5")

    c.circulo(*puntos_real[-1], 4.5, TINTA_SEC)
    c.circulo(*puntos_pred[-1], 4.5, ACENTO)
    c.texto(puntos_real[-1][0] + 12, puntos_real[-1][1] + 5, "Venta real", size=15,
            color=TINTA_SEC, peso=600)
    c.texto(puntos_pred[-1][0] + 12, puntos_pred[-1][1] - 8, "Predicción", size=15,
            color=ACENTO, peso=700)

    c.nota("Modelo de calendario · una predicción por semana, entrenada solo con lo anterior a ella")
    return guardar(c, "06-04-prediccion-vs-real")


# =============================================================================
# Figura 5 · Previsión de septiembre de 2026
# =============================================================================

def figura_5(ctx):
    fut = modelos.predecir_futuro(ctx.completa, h=10, escenario="cierre")
    fut["periodo_inicio"] = pd.to_datetime(fut["periodo_inicio"])
    sep = fut[(fut["periodo_inicio"] >= "2026-09-01") & (fut["periodo_inicio"] < "2026-10-05")].copy()
    # «Semana normal»: lo que el modelo prevé para esas mismas semanas sin
    # festivos ni acopio.
    fuera = fut[~fut.index.isin(sep.index) & (fut["horas_cierre_festivo"] == 0)]
    normal = np.interp(sep["periodo_inicio"].map(pd.Timestamp.toordinal),
                       fuera["periodo_inicio"].map(pd.Timestamp.toordinal),
                       fuera["prediccion"])
    media_normal = float(normal.mean())

    categorias = ["Semana normal\n(sin fiesta)", "7-13 sep\n(aprovisionamiento)", "14-20 sep\n(aprovisionamiento)",
                  "21-27 sep\n(cierra lunes)", "28 sep-4 oct\n(cierra lunes)"]
    valores = [media_normal] + list(sep["prediccion"])
    colores = [NEUTRO, ACENTO, ACENTO, ACENTO2, ACENTO2]

    c = Lienzo()
    c.encabezado(
        "Previsión",
        "Septiembre: dos semanas fuertes y dos de fiesta local",
        "Las de aprovisionamiento de fiestas locales, muy por encima de lo normal; las que cierran el lunes de fiesta, "
        "por debajo. El modelo reparte el aprovisionamiento a partes iguales: conviene cargar el pedido en la segunda semana.",
    )

    x0, x1 = MARGEN["izq"], ANCHO - MARGEN["der"]
    y0, y1 = ALTO - MARGEN["abj"], 300
    n = len(categorias)
    paso = (x1 - x0) / n
    barra_w = paso * 0.5
    v_max = max(valores) * 1.22

    eje_y_lineal(c, y1, y0, 0, v_max, 3, fmt=lambda v: f"{num(v, 0)} €")

    y_ref = y0 - (y0 - y1) * media_normal / v_max
    c.linea(x0, y_ref, x1, y_ref, TINTA_SEC, ancho=1.4, dash="3,5", opacity=0.7)

    for i, (cat, v, color) in enumerate(zip(categorias, valores, colores)):
        cx = x0 + paso * (i + 0.5)
        bh = (y0 - y1) * v / v_max
        c.rect(cx - barra_w / 2, y0 - bh, barra_w, bh, color, rx=5)
        # Halo, no etiqueta suelta: dos de las cinco barras terminan casi a la
        # altura de la línea de referencia, y sin respaldo el texto se cruza
        # con el guion de la línea y parece tachado.
        c.texto_con_halo(cx, y0 - bh - 12, f"{num(v, 0)} €", size=15, color=TINTA, peso=700)
        for li, linea in enumerate(cat.split("\n")):
            c.texto(cx, y0 + 26 + li * 17, linea, size=13, color=TINTA_SEC, ancla="middle")

    c.nota("Previsión hecha con los datos hasta el 2 de agosto de 2026 · validada a 1, 2, 4 y 10 semanas vista; estas cuatro caen entre la 6 y la 9 · "
           "«semana normal»: lo que prevé el modelo para esas semanas sin festivos ni aprovisionamiento de fiestas")
    return guardar(c, "06-05-prevision-septiembre")


# =============================================================================
# Figura 6 · Los cuatro grupos de artículos, según lo que dejan
# =============================================================================

GRUPOS_RENT = {
    0: "Margen alto y\nmucha aportación",
    1: "Margen medio y\nmucha aportación",
    2: "Margen alto y\npoca aportación",
    3: "Margen bajo",
}


def figura_6():
    perfil = cc._rentab()["perfil"]

    c = Lienzo()
    c.encabezado(
        "Rentabilidad",
        "Cuatro grupos, y uno de ellos deja más de un tercio del margen",
        "El eje horizontal es cuánto deja cada venta; el vertical, cuánto suma en total. "
        f"{num(perfil.loc[0, 'articulos'], 0)} artículos —casi todos frescos— están arriba "
        "a la derecha: dejan mucho y aportan mucho.",
        ancho_max=72,
    )

    x0, x1 = MARGEN["izq"] + 20, ANCHO - MARGEN["der"] - 40
    y0, y1 = ALTO - MARGEN["abj"] - 10, 300
    x_min, x_max = 8, 42
    y_min, y_max = 0, 45

    def px(v):
        return x0 + (x1 - x0) * (v - x_min) / (x_max - x_min)

    def py(v):
        return y0 - (y0 - y1) * (v - y_min) / (y_max - y_min)

    # Rejilla en las dos direcciones, más ligera que en las de barras: aquí es
    # un plano de posiciones, no una escala que haya que leer con precisión.
    for v in [10, 20, 30, 40]:
        c.linea(px(v), y1, px(v), y0, REJILLA, ancho=1)
        c.texto(px(v), y0 + 26, f"{v} %", size=13, color=TINTA_SEC, ancla="middle")
    for v in [0, 15, 30, 45]:
        c.linea(x0, py(v), x1, py(v), REJILLA, ancho=1)
        c.texto(x0 - 14, py(v) + 5, f"{v} %", size=13, color=TINTA_SEC, ancla="end")

    c.texto((x0 + x1) / 2, y0 + 58, "De cada 100 € vendidos, quedan (margen)",
            size=14, color=TINTA_SEC, ancla="middle", peso=600)
    # Título del eje Y como rótulo horizontal sobre la esquina del área de
    # trazado, no girado ni apretado en el margen izquierdo: con un texto
    # largo, un `text-anchor="middle"` centrado en un margen de 90px se sale
    # del lienzo por los dos lados.
    c.texto(x0, y1 - 16, "Parte del margen de los artículos fijos (%)", size=13.5,
            color=TINTA_SEC, peso=600)

    ns = perfil["articulos"]
    r_px = {g: 30 + (110 - 30) * ((ns[g] ** 0.5 - ns.pow(0.5).min())
                                    / (ns.pow(0.5).max() - ns.pow(0.5).min()))
            for g in perfil.index}

    for g in perfil.index:
        cx, cy = px(perfil.loc[g, "margen_pct"]), py(perfil.loc[g, "cuota_margen_euros_pct"])
        color = ACENTO if g == 0 else NEUTRO
        c.circulo(cx, cy, r_px[g], color, )
        c.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_px[g]:.1f}" fill="none" '
              f'stroke="{PAPEL}" stroke-width="3"/>')
        c.texto(cx, cy - 4, f"{num(perfil.loc[g, 'articulos'], 0)}", size=17, color="white",
                peso=700, ancla="middle")
        c.texto(cx, cy + 15, "artículos", size=11, color="white", ancla="middle")
        etiqueta = GRUPOS_RENT[g]
        # Solo la etiqueta del grupo 3 va debajo de su burbuja: la del grupo 2,
        # debajo, pisaba las marcas del eje horizontal, y encima cabe en el
        # hueco que deja la burbuja del grupo 1.
        ly = (cy + r_px[g] + 24) if g == 3 else (cy - r_px[g] - 34)
        for li, linea in enumerate(etiqueta.split("\n")):
            c.texto(cx, ly + li * 17, linea, size=14, color=TINTA, peso=700 if g == 0 else 600,
                    ancla="middle")

    c.nota(f"{num(perfil['articulos'].sum(), 0)} artículos fijos con margen, agrupados con el "
           "jerárquico de Ward · datos de 31 meses")
    return guardar(c, "06-06-grupos-rentabilidad")


# =============================================================================
# Render a PNG vía navegador sin cabecera
# =============================================================================

NAVEGADORES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def buscar_navegador():
    for ruta in NAVEGADORES:
        if Path(ruta).exists():
            return ruta
    encontrado = shutil.which("msedge") or shutil.which("chrome")
    if encontrado:
        return encontrado
    raise RuntimeError("No se encontró Edge ni Chrome para exportar a PNG")


def svg_a_png(ruta_svg: Path, escala=3):
    """Envuelve el SVG en un html mínimo y lo captura con el navegador en modo
    sin cabecera.
    """
    navegador = buscar_navegador()
    html_tmp = ruta_svg.with_suffix(".tmp.html")
    svg_contenido = ruta_svg.read_text(encoding="utf-8")
    html_tmp.write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><style>'
        f'*{{margin:0;padding:0}} body{{width:{ANCHO}px;height:{ALTO}px}}'
        f'</style></head><body>{svg_contenido}</body></html>',
        encoding="utf-8",
    )
    ruta_png = ruta_svg.with_suffix(".png")
    # `--window-size` es en píxeles css (viewport lógico);
    # `--force-device-scale-factor` multiplica sobre eso para el supermuestreo.
    subprocess.run(
        [
            navegador, "--headless", "--disable-gpu",
            f"--window-size={ANCHO},{ALTO}",
            "--force-device-scale-factor=" + str(escala),
            "--hide-scrollbars",
            f"--screenshot={ruta_png}",
            html_tmp.resolve().as_uri(),
        ],
        check=True, capture_output=True,
    )
    html_tmp.unlink()
    print(f"  {ruta_png.name} exportado ({ANCHO * escala}x{ALTO * escala}px)")


# =============================================================================
# Figura 7 · Qué mira el modelo (II.4)
# =============================================================================

def figura_7(ctx):
    """Esquema del modelo para la dirección: las cuatro cosas que mira, la
    previsión que sale y lo que no necesita.
    """
    oficial = modelos.tabla_comparativa(
        modelos.evaluar(ctx.completa, ["ols_basico"])).loc["ols_basico", "todas_mae"]
    pct = 100 * oficial / ctx.completa["venta_neta"].mean()

    c = Lienzo()
    c.encabezado(
        "El modelo",
        "Qué mira el modelo para prever la venta de la semana",
        "Solo cosas que se conocen de antemano. Con ellas, la previsión de cada "
        "semana se calcula sin esperar a saber cuántos clientes vendrán.",
        ancho_max=82,
    )

    cajas = [
        ("Tendencia", ["La tienda crece", "poco a poco"]),
        ("Festivos", ["Las horas que", "la tienda cierra"]),
        ("Semana Santa", ["Su semana,", "cada año distinta"]),
        ("Fiestas locales", ["Las dos semanas", "de antes"]),
    ]
    x0, x1 = MARGEN["izq"] + 20, ANCHO - MARGEN["der"] - 20
    n = len(cajas)
    hueco = 36
    w = (x1 - x0 - hueco * (n - 1)) / n
    y_caja, h_caja = 260, 170
    y_res, h_res = 560, 150
    w_res = 620
    x_res = (ANCHO - w_res) / 2

    for i, (titulo, lineas) in enumerate(cajas):
        x = x0 + i * (w + hueco)
        c.rect(x, y_caja, w, h_caja, "#ffffff", rx=12)
        c.rect(x, y_caja, w, 8, ACENTO, rx=4)
        c.texto(x + w / 2, y_caja + 62, titulo, size=24, color=TINTA, peso=700, ancla="middle")
        c.texto_multilinea(x + w / 2, y_caja + 102, lineas, size=17, color=TINTA_SEC,
                           peso=400, salto=24, ancla="middle")
        # Flecha de cada caja al resultado.
        xa, ya = x + w / 2, y_caja + h_caja
        xb, yb = x_res + w_res * (i + 0.5) / n, y_res
        c.linea(xa, ya + 6, xb, yb - 14, NEUTRO, ancho=2.2)
        c.add(f'<polygon points="{xb - 7:.1f},{yb - 16:.1f} {xb + 7:.1f},{yb - 16:.1f} '
              f'{xb:.1f},{yb - 4:.1f}" fill="{NEUTRO}"/>')

    c.rect(x_res, y_res, w_res, h_res, ACENTO, rx=14)
    c.texto(ANCHO / 2, y_res + 58, "PREVISIÓN DE LA VENTA SEMANAL", size=24,
            color="#ffffff", peso=700, ancla="middle", spacing="1px")
    c.texto(ANCHO / 2, y_res + 102,
            f"En una semana típica se equivoca en unos {num(oficial, 0)} € ({num(pct, 0)} %)",
            size=19, color="#ffffff", peso=400, ancla="middle")

    c.texto(ANCHO / 2, y_res + h_res + 62,
            "Lo que no necesita: el número de tickets, la venta con tarjeta Club Familia "
            "ni la venta de las semanas anteriores",
            size=17, color=TINTA_SEC, peso=400, ancla="middle")

    c.nota("Previsión validada 76 veces, a cuatro semanas vista, sobre semanas cuya venta ya se conocía")
    return guardar(c, "06-07-modelo")


def main():
    ctx = cc.Contexto()
    rutas = [
        figura_1(ctx),
        figura_2(),
        figura_3(ctx),
        figura_4(ctx),
        figura_5(ctx),
        figura_6(),
        figura_7(ctx),
    ]
    print("\nExportando a PNG...")
    for ruta in rutas:
        svg_a_png(ruta)


if __name__ == "__main__":
    main()

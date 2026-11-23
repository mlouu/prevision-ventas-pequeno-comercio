"""
Regenera `datos_limpio/` a partir de los informes originales de `datos_bruto/`.

    python actualizar_datos.py
"""
import sys

sys.path.insert(0, "src")

import procesar_lote
import procesar_lote_cierre_z

fallos = 0

print("=" * 70)
print("Diario de ventas por artículo (mensual)")
print("=" * 70)
try:
    procesar_lote.procesar_todos()
except Exception as exc:
    fallos += 1
    print(f"  [FALLO] no se pudo procesar el diario por artículo: {exc}")

print()
print("=" * 70)
print("Cierres Z semanales (la serie del modelo)")
print("=" * 70)
try:
    procesar_lote_cierre_z.procesar_todos()
except Exception as exc:
    fallos += 1
    print(f"  [FALLO] no se pudieron procesar los cierres Z: {exc}")

print()
if fallos:
    print(f"Terminado con {fallos} pipeline(s) fallido(s). Revisa los mensajes de arriba.")
    sys.exit(1)
print("Los dos pipelines han terminado correctamente.")

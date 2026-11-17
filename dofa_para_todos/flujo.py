"""El flujo completo en una sola función: imagen → DOFA → tarea → resultados."""
from __future__ import annotations

import os
from datetime import datetime

from . import imagenes, modelo as mod, resultados, tareas

AGRUPAR = "Agrupar zonas parecidas (sin etiquetas)"
CLASIFICAR = "Clasificar coberturas (con mis ejemplos)"
MAX_CELDAS = 300_000   # límite para no agotar la memoria de Colab


def ejecutar(ruta_imagen: str, ondas: list[float], tarea: str = AGRUPAR, radar: bool = False,
             detalle: int = 1, tamano: str = "base", grupos: int = 5, ruta_ejemplos: str | None = None,
             columna: str | None = None, carpeta: str = "resultados", progreso=None,
             modelo=None, extractor=None, suavizado: int = 1) -> dict:
    """Ejecuta una tarea completa y devuelve las rutas de los archivos y el informe."""
    aviso = progreso or (lambda fraccion, texto="": None)

    aviso(0.02, "Leyendo la imagen")
    img = imagenes.leer_imagen(ruta_imagen)
    if len(ondas) != img.bandas:
        raise ValueError(f"La imagen tiene {img.bandas} bandas, pero elegiste {len(ondas)}. "
                         "Marca exactamente las bandas que tiene el archivo, en el mismo orden.")
    filas, columnas, transform = imagenes.malla_resultados(img, detalle)
    if filas * columnas > MAX_CELDAS:
        raise ValueError("La imagen es muy grande para este nivel de detalle. Baja el nivel de detalle "
                         "o recorta una zona más pequeña.")

    aviso(0.05, "Preparando las bandas")
    datos = imagenes.preparar(img, radar=radar)

    aviso(0.08, "Cargando DOFA (la primera vez descarga los pesos)")
    modelo = modelo or mod.cargar_modelo(tamano)
    kwargs = {"extractor": extractor} if extractor else {}
    resumenes = mod.resumir_imagen(modelo, datos, ondas, detalle,
                                   progreso=lambda f: aviso(0.1 + 0.7 * f, "DOFA está analizando la imagen"),
                                   **kwargs)[:filas, :columnas]
    validas = imagenes.celdas_validas(img, detalle)

    aviso(0.85, "Ejecutando la tarea")
    if tarea == CLASIFICAR:
        if not ruta_ejemplos or not columna:
            raise ValueError("Para clasificar necesitas subir tus ejemplos y elegir la columna con el nombre de la cobertura.")
        gdf = tareas.leer_ejemplos(ruta_ejemplos)
        res = tareas.clasificar(resumenes, validas, gdf, columna, transform, img.crs)
        titulo = "Coberturas según DOFA"
    else:
        res = tareas.agrupar(resumenes, validas, grupos)
        titulo = f"{grupos} grupos encontrados por DOFA"
    if suavizado > 0:
        res.mapa = tareas.suavizar(res.mapa, len(res.nombres), pasadas=suavizado)

    aviso(0.95, "Guardando resultados")
    os.makedirs(carpeta, exist_ok=True)
    base = os.path.join(carpeta, f"dofa_{datetime.now():%Y%m%d_%H%M%S}")
    vista = imagenes.vista_color(img)
    archivos = {
        "figura": resultados.figura(vista, res, base + "_mapa.png", titulo),
        "geotiff": resultados.guardar_geotiff(res, transform, img.crs, base + "_mapa.tif"),
        "leyenda": base + "_mapa_leyenda.csv",
    }
    tabla = resultados.tabla_areas(res, imagenes.area_celda_ha(img, detalle))
    tabla.to_csv(base + "_areas.csv", index=False)
    archivos["areas"] = base + "_areas.csv"
    aviso(1.0, "Listo")
    return {"archivos": archivos, "tabla": tabla, "informe": res.informe, "resultado": res}

"""El flujo completo en una sola función: imagen(es) → DOFA → tarea → resultados."""
from __future__ import annotations

import os
from datetime import datetime

from . import imagenes, modelo as mod, resultados, tareas

AGRUPAR = "Agrupar zonas parecidas (sin etiquetas)"
CLASIFICAR = "Clasificar coberturas (con mis ejemplos)"
CAMBIOS = "Detectar cambios entre dos fechas"
TAREAS = [AGRUPAR, CLASIFICAR, CAMBIOS]
MAX_CELDAS = 300_000   # límite para no agotar la memoria de Colab


def _revisar(img, ondas, detalle, cual=""):
    if len(ondas) != img.bandas:
        raise ValueError(f"La imagen{cual} tiene {img.bandas} bandas, pero elegiste {len(ondas)}. "
                         "Marca exactamente las bandas que tiene el archivo, en el mismo orden.")
    filas, columnas, _ = imagenes.malla_resultados(img, detalle)
    if filas * columnas > MAX_CELDAS:
        raise ValueError("La imagen es muy grande para este nivel de detalle. Baja el nivel de detalle "
                         "o recorta una zona más pequeña.")


def ejecutar(ruta_imagen: str, ondas: list[float], tarea: str = AGRUPAR, radar: bool = False,
             detalle: int = 1, tamano: str = "base", grupos: int = 5, ruta_ejemplos: str | None = None,
             columna: str | None = None, carpeta: str = "resultados", progreso=None,
             modelo=None, extractor=None, suavizado: int = 1, ruta_imagen2: str | None = None,
             ondas2: list[float] | None = None, sensibilidad: float = 1.0) -> dict:
    """Ejecuta una tarea completa y devuelve las rutas de los archivos y el informe.

    Para detectar cambios, `ruta_imagen` es la imagen de antes y `ruta_imagen2` la de después.
    `ondas2` permite que la segunda imagen sea de otro sensor; si no se da, se usan las mismas longitudes de onda.
    """
    aviso = progreso or (lambda fraccion, texto="": None)

    aviso(0.02, "Leyendo la imagen")
    img = imagenes.leer_imagen(ruta_imagen)
    _revisar(img, ondas, detalle, " de antes" if tarea == CAMBIOS else "")
    filas, columnas, transform = imagenes.malla_resultados(img, detalle)
    img2 = None
    if tarea == CAMBIOS:
        if not ruta_imagen2:
            raise ValueError("Para detectar cambios necesitas subir también la imagen de después.")
        ondas2 = ondas2 or ondas
        img2 = imagenes.alinear(imagenes.leer_imagen(ruta_imagen2), img)
        _revisar(img2, ondas2, detalle, " de después")
        if not img2.valida.any():
            raise ValueError("Las dos imágenes no se superponen. Revisa que sean de la misma zona.")

    aviso(0.05, "Preparando las bandas")
    # Con dos fechas del mismo sensor, ambas se normalizan con las mismas estadísticas para que sean comparables
    mismo_sensor = img2 is not None and list(ondas2) == list(ondas)
    stats = imagenes.estadisticas([img, img2], radar) if mismo_sensor else None
    datos = imagenes.preparar(img, radar=radar, stats=stats)

    aviso(0.08, "Cargando DOFA (la primera vez descarga los pesos)")
    modelo = modelo or mod.cargar_modelo(tamano)
    kwargs = {"extractor": extractor} if extractor else {}
    tramo = 0.7 if img2 is None else 0.35
    resumenes = mod.resumir_imagen(modelo, datos, ondas, detalle,
                                   progreso=lambda f: aviso(0.1 + tramo * f, "DOFA está analizando la imagen"),
                                   **kwargs)[:filas, :columnas]
    validas = imagenes.celdas_validas(img, detalle)

    if img2 is not None:
        datos2 = imagenes.preparar(img2, radar=radar, stats=stats)
        resumenes2 = mod.resumir_imagen(modelo, datos2, ondas2, detalle,
                                        progreso=lambda f: aviso(0.45 + 0.35 * f, "DOFA está analizando la imagen de después"),
                                        **kwargs)[:filas, :columnas]
        validas &= imagenes.celdas_validas(img2, detalle)

    aviso(0.85, "Ejecutando la tarea")
    if tarea == CLASIFICAR:
        if not ruta_ejemplos or not columna:
            raise ValueError("Para clasificar necesitas subir tus ejemplos y elegir la columna con el nombre de la cobertura.")
        gdf = tareas.leer_ejemplos(ruta_ejemplos)
        res = tareas.clasificar(resumenes, validas, gdf, columna, transform, img.crs)
        titulo = "Coberturas según DOFA"
    elif tarea == CAMBIOS:
        res = tareas.detectar_cambios(resumenes, resumenes2, validas, sensibilidad)
        titulo = "Cambios detectados por DOFA"
    else:
        res = tareas.agrupar(resumenes, validas, grupos)
        titulo = f"{grupos} grupos encontrados por DOFA"
    if suavizado > 0:
        res.mapa = tareas.suavizar(res.mapa, len(res.nombres), pasadas=suavizado)

    aviso(0.95, "Guardando resultados")
    os.makedirs(carpeta, exist_ok=True)
    base = os.path.join(carpeta, f"dofa_{datetime.now():%Y%m%d_%H%M%S}")
    if img2 is not None:
        vista, vista2 = imagenes.vistas_comparables(img, img2, ondas, ondas2)
        figura = resultados.figura_cambios(vista, vista2, res, base + "_mapa.png", imagenes.tamano_celda(detalle))
    else:
        figura = resultados.figura(imagenes.vista_color(img, ondas=ondas), res, base + "_mapa.png", titulo)
    archivos = {
        "figura": figura,
        "geotiff": resultados.guardar_geotiff(res, transform, img.crs, base + "_mapa.tif"),
        "leyenda": base + "_mapa_leyenda.csv",
    }
    tabla = resultados.tabla_areas(res, imagenes.area_celda_ha(img, detalle))
    tabla.to_csv(base + "_areas.csv", index=False)
    archivos["areas"] = base + "_areas.csv"
    aviso(1.0, "Listo")
    return {"archivos": archivos, "tabla": tabla, "informe": res.informe, "resultado": res,
            "transform": transform, "crs": img.crs}

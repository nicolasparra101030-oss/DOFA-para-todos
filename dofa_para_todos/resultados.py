"""Mapas, tablas y archivos de resultados."""
from __future__ import annotations

import os

import numpy as np

from .tareas import SIN_DATO, Resultado

PALETA = ["#1b7837", "#2166ac", "#e6ab02", "#d95f02", "#7570b3", "#e7298a",
          "#66a61e", "#a6761d", "#1f78b4", "#b2df8a", "#fb9a99", "#666666"]


COLOR_POR_PALABRA = [
    (("agua", "rio", "río", "lago", "laguna", "mar", "water", "humedal", "cienaga", "ciénaga"), "#2166ac"),
    (("bosque", "selva", "forest", "arbol", "árbol", "manglar", "verde", "parque", "vegeta", "jard"), "#1b7837"),
    (("vía", "via ", "vias", "vías", "carretera", "calle", "avenida", "road"), "#d7301f"),
    (("pasto", "potrero", "pastizal", "pradera", "sabana", "herbazal", "grass"), "#a6d96a"),
    (("cultivo", "crop", "arroz", "cafe", "café", "palma", "maiz", "maíz", "agric"), "#e6ab02"),
    (("urbano", "ciudad", "construido", "edific", "vivienda", "urban", "industrial"), "#7f7f7f"),
    (("suelo", "desnudo", "arena", "mineria", "minería", "bare", "roca"), "#a6761d"),
    (("nube", "cloud", "sombra"), "#d9d9d9"),
    (("paramo", "páramo", "arbusto", "matorral"), "#b8a2d6"),
]


def colores(n: int, nombres: list[str] | None = None) -> list[str]:
    """Colores del mapa. Si el nombre de la cobertura es conocido (agua, bosque...), usa un color que lo evoque."""
    resultado, usados = [], set()
    for i in range(n):
        nombre = (nombres[i] if nombres else "").lower()
        color = next((c for palabras, c in COLOR_POR_PALABRA if any(p in nombre for p in palabras)), None)
        if color is None or color in usados:
            color = next(c for c in PALETA + [PALETA[i % len(PALETA)]] if c not in usados or c == PALETA[i % len(PALETA)])
        usados.add(color)
        resultado.append(color)
    return resultado


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mapa_a_color(mapa: np.ndarray, nombres: list[str]) -> np.ndarray:
    n = len(nombres)
    tabla = np.array([_rgb(c) for c in colores(n, nombres)] + [(0, 0, 0)], dtype="float32") / 255
    idx = np.where(mapa == SIN_DATO, n, mapa)
    return tabla[idx]


def tabla_areas(res: Resultado, area_celda_ha: float):
    import pandas as pd

    validas = res.mapa != SIN_DATO
    total = validas.sum()
    filas = []
    for i, nombre in enumerate(res.nombres):
        celdas = int((res.mapa == i).sum())
        filas.append({"Cobertura": nombre, "Hectáreas": round(celdas * area_celda_ha, 1),
                      "Porcentaje": round(100 * celdas / total, 1) if total else 0.0})
    return pd.DataFrame(filas)


def figura(vista_rgb: np.ndarray, res: Resultado, ruta: str, titulo: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(res.nombres)
    mapa_rgb = mapa_a_color(res.mapa, res.nombres)
    paneles = 3 if res.confianza is not None else 2
    fig, ejes = plt.subplots(1, paneles, figsize=(7 * paneles, 7))
    ejes[0].imshow(vista_rgb)
    ejes[0].set_title("Imagen original")
    ejes[1].imshow(mapa_rgb, interpolation="nearest")
    ejes[1].set_title(titulo)
    if res.confianza is not None:
        im = ejes[2].imshow(np.where(res.mapa == SIN_DATO, np.nan, res.confianza), cmap="viridis", vmin=0, vmax=1,
                            interpolation="nearest")
        ejes[2].set_title("Confianza del modelo")
        fig.colorbar(im, ax=ejes[2], fraction=0.046)
    for e in ejes:
        e.axis("off")
    fig.legend(handles=[Patch(color=c, label=nombre) for c, nombre in zip(colores(n, res.nombres), res.nombres)],
               loc="lower center", ncol=min(n, 6), frameon=False, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def guardar_geotiff(res: Resultado, transform, crs, ruta: str) -> str:
    """Guarda el mapa georreferenciado, listo para abrir en QGIS o ArcGIS."""
    import rasterio

    datos = np.where(res.mapa == SIN_DATO, 255, res.mapa).astype("uint8")
    with rasterio.open(ruta, "w", driver="GTiff", height=datos.shape[0], width=datos.shape[1], count=1,
                       dtype="uint8", crs=crs, transform=transform, nodata=255, compress="deflate") as dst:
        dst.write(datos, 1)
        dst.write_colormap(1, {i: _rgb(c) + (255,) for i, c in enumerate(colores(len(res.nombres), res.nombres))})
        dst.update_tags(1, **{f"clase_{i}": nombre for i, nombre in enumerate(res.nombres)})
    with open(os.path.splitext(ruta)[0] + "_leyenda.csv", "w", encoding="utf-8") as f:
        f.write("valor,cobertura,color\n")
        for i, (nombre, c) in enumerate(zip(res.nombres, colores(len(res.nombres), res.nombres))):
            f.write(f"{i},{nombre},{c}\n")
    return ruta

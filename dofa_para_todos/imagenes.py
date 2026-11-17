"""Lectura y preparación de imágenes satelitales para DOFA."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine

TAM_VENTANA = 224     # tamaño de entrada de DOFA
TAM_PARCHE = 16       # DOFA divide cada ventana en parches de 16 × 16 píxeles
PARCHES = TAM_VENTANA // TAM_PARCHE   # 14 parches por lado


@dataclass
class Imagen:
    datos: np.ndarray          # (bandas, alto, ancho), float32
    valida: np.ndarray         # (alto, ancho), True donde hay datos
    transform: Affine
    crs: object
    ruta: str

    @property
    def bandas(self) -> int:
        return self.datos.shape[0]

    @property
    def alto(self) -> int:
        return self.datos.shape[1]

    @property
    def ancho(self) -> int:
        return self.datos.shape[2]


def leer_imagen(ruta: str) -> Imagen:
    with rasterio.open(ruta) as src:
        datos = src.read().astype("float32")
        valida = src.dataset_mask() > 0
        if src.nodata is not None:
            valida &= ~np.all(datos == src.nodata, axis=0)
        valida &= np.all(np.isfinite(datos), axis=0)
        return Imagen(np.nan_to_num(datos), valida, src.transform, src.crs, ruta)


def describir(img: Imagen) -> str:
    res_x, res_y = abs(img.transform.a), abs(img.transform.e)
    unidad = "°" if img.crs is not None and img.crs.is_geographic else " m"
    crs = img.crs.to_string() if img.crs is not None else "sin sistema de coordenadas"
    return (f"**{img.bandas} bandas** · {img.ancho} × {img.alto} píxeles · "
            f"píxel de {res_x:g}{unidad} × {res_y:g}{unidad} · {crs} · "
            f"{img.valida.mean() * 100:.0f} % de píxeles con datos")


def indices_color(ondas: list[float] | None, bandas: int) -> tuple[int, int, int]:
    """Elige las bandas más cercanas al rojo, verde y azul según sus longitudes de onda.

    Si las longitudes de onda están en otro orden que el archivo, la vista previa saldrá con colores raros:
    así el usuario nota el error.
    """
    if ondas and len(ondas) == bandas and min(ondas) < 0.7:
        return tuple(int(np.argmin([abs(o - objetivo) for o in ondas])) for objetivo in (0.665, 0.56, 0.49))
    return (0, 1, 2) if bandas >= 3 else (0, 0, 0)


def vista_color(img: Imagen, indices: tuple[int, int, int] | None = None,
                ondas: list[float] | None = None, limites=None) -> np.ndarray:
    """Imagen RGB (alto, ancho, 3) entre 0 y 1 para mostrar en pantalla.

    `limites` (bajo, alto por canal) permite mostrar dos fechas con la misma escala de brillo.
    """
    if indices is None:
        indices = indices_color(ondas, img.bandas)
    rgb = np.stack([img.datos[i] for i in indices], axis=-1).astype("float32")
    for c in range(3):
        canal = rgb[..., c]
        if limites is not None:
            bajo, alto = limites[c]
        else:
            vals = canal[img.valida]
            if not vals.size:
                continue
            bajo, alto = np.percentile(vals, [2, 98])
        rgb[..., c] = np.clip((canal - bajo) / (alto - bajo + 1e-9), 0, 1)
    rgb[~img.valida] = 0
    return rgb


def vistas_comparables(img1: Imagen, img2: Imagen, ondas1, ondas2) -> tuple[np.ndarray, np.ndarray]:
    """Vistas de dos fechas con la misma escala de brillo, para compararlas a simple vista."""
    i1, i2 = indices_color(ondas1, img1.bandas), indices_color(ondas2, img2.bandas)
    limites = []
    for c in range(3):
        vals = np.concatenate([img1.datos[i1[c]][img1.valida], img2.datos[i2[c]][img2.valida]])
        limites.append(tuple(np.percentile(vals, [2, 98])) if vals.size else (0.0, 1.0))
    return (vista_color(img1, i1, limites=limites), vista_color(img2, i2, limites=limites))


def alinear(img: Imagen, ref: Imagen) -> Imagen:
    """Reproyecta una imagen a la misma grilla (sistema de coordenadas, tamaño y píxel) que la de referencia."""
    from rasterio.warp import Resampling, reproject

    misma = (img.crs == ref.crs and img.transform == ref.transform and img.datos.shape[1:] == ref.datos.shape[1:])
    if misma:
        return img
    datos = np.zeros((img.bandas, ref.alto, ref.ancho), dtype="float32")
    for b in range(img.bandas):
        reproject(img.datos[b], datos[b], src_transform=img.transform, src_crs=img.crs,
                  dst_transform=ref.transform, dst_crs=ref.crs, resampling=Resampling.bilinear)
    valida = np.zeros((ref.alto, ref.ancho), dtype="uint8")
    reproject(img.valida.astype("uint8"), valida, src_transform=img.transform, src_crs=img.crs,
              dst_transform=ref.transform, dst_crs=ref.crs, resampling=Resampling.nearest, dst_nodata=0)
    return Imagen(datos, valida.astype(bool), ref.transform, ref.crs, img.ruta)


def _valores(img: Imagen, radar: bool) -> np.ndarray:
    datos = img.datos
    if radar and np.nanmedian(datos[:, img.valida]) > 0:
        datos = 10 * np.log10(np.clip(datos, 1e-6, None))     # radar lineal → decibeles
    return datos


def estadisticas(imgs: list[Imagen], radar: bool = False) -> list[tuple[float, float, float, float]]:
    """Por banda: (mínimo, máximo, media, desviación) tras recortar extremos, con los píxeles de todas las imágenes."""
    salida = []
    datos = [_valores(i, radar) for i in imgs]
    for b in range(imgs[0].bandas):
        vals = np.concatenate([d[b][i.valida] for d, i in zip(datos, imgs)])
        if vals.size == 0:
            salida.append((0.0, 1.0, 0.0, 1.0))
            continue
        bajo, alto = np.percentile(vals, [1, 99])
        recortados = np.clip(vals, bajo, alto)
        salida.append((float(bajo), float(alto), float(recortados.mean()), float(recortados.std()) + 1e-6))
    return salida


def preparar(img: Imagen, radar: bool = False, stats=None) -> np.ndarray:
    """Pone cada banda en una escala común (media 0, desviación 1), con valores extremos recortados.

    Para comparar dos fechas hay que pasar las mismas `stats` a las dos imágenes (ver `estadisticas`):
    si cada una se normaliza por su cuenta, un cambio en una parte altera la escala de toda la imagen.
    """
    stats = stats or estadisticas([img], radar)
    datos = _valores(img, radar).copy()
    for b, (bajo, alto, media, desv) in enumerate(stats):
        datos[b] = (np.clip(datos[b], bajo, alto) - media) / desv
    datos[:, ~img.valida] = 0
    return datos.astype("float32")


def lado_ventana(detalle: int) -> int:
    """Píxeles originales que cubre cada ventana. Con más detalle la ventana se agranda antes de entrar a DOFA."""
    if detalle not in (1, 2, 4):
        raise ValueError("El nivel de detalle debe ser 1, 2 o 4")
    return TAM_VENTANA // detalle


def cortar_ventanas(datos: np.ndarray, detalle: int = 1) -> tuple[np.ndarray, int, int]:
    """Corta la imagen en ventanas cuadradas (rellenando los bordes). Devuelve ventanas, filas y columnas."""
    lado = lado_ventana(detalle)
    _, alto, ancho = datos.shape
    nf, nc = -(-alto // lado), -(-ancho // lado)
    relleno = ((0, 0), (0, nf * lado - alto), (0, nc * lado - ancho))
    modo = "reflect" if alto > 1 and ancho > 1 and nf * lado - alto < alto and nc * lado - ancho < ancho else "edge"
    datos = np.pad(datos, relleno, mode=modo)
    ventanas = (datos.reshape(datos.shape[0], nf, lado, nc, lado)
                .transpose(1, 3, 0, 2, 4)
                .reshape(nf * nc, datos.shape[0], lado, lado))
    return ventanas, nf, nc


def tamano_celda(detalle: int) -> int:
    """Píxeles originales que cubre cada celda del mapa de resultados."""
    return TAM_PARCHE // detalle


def malla_resultados(img: Imagen, detalle: int) -> tuple[int, int, Affine]:
    """Filas, columnas y georreferencia de la malla de celdas del mapa de resultados."""
    celda = tamano_celda(detalle)
    filas, columnas = -(-img.alto // celda), -(-img.ancho // celda)
    escala = Affine.scale(celda)
    t = img.transform @ escala if hasattr(Affine, "__matmul__") else img.transform * escala
    return filas, columnas, t


def celdas_validas(img: Imagen, detalle: int) -> np.ndarray:
    """Una celda es válida si al menos la mitad de sus píxeles dentro de la imagen tiene datos.

    Las celdas del borde, que quedan en parte fuera de la imagen, se evalúan solo con su parte de adentro.
    """
    celda = tamano_celda(detalle)
    filas, columnas, _ = malla_resultados(img, detalle)
    validos = np.zeros((filas * celda, columnas * celda), dtype="float32")
    dentro = np.zeros_like(validos)
    validos[:img.alto, :img.ancho] = img.valida
    dentro[:img.alto, :img.ancho] = 1
    suma_validos = validos.reshape(filas, celda, columnas, celda).sum(axis=(1, 3))
    suma_dentro = dentro.reshape(filas, celda, columnas, celda).sum(axis=(1, 3))
    return suma_validos >= 0.5 * np.maximum(suma_dentro, 1)


def area_celda_ha(img: Imagen, detalle: int) -> float:
    """Área aproximada de una celda del mapa, en hectáreas."""
    _, _, t = malla_resultados(img, detalle)
    ancho, alto = abs(t.a), abs(t.e)
    if img.crs is not None and img.crs.is_geographic:
        lat = img.transform.f + img.transform.e * img.alto / 2
        ancho *= 111320 * np.cos(np.radians(lat))
        alto *= 110540
    return ancho * alto / 10000

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


def vista_color(img: Imagen, indices: tuple[int, int, int] | None = None) -> np.ndarray:
    """Imagen RGB (alto, ancho, 3) entre 0 y 1 para mostrar en pantalla."""
    if indices is None:
        indices = (0, 1, 2) if img.bandas >= 3 else (0, 0, 0)
    rgb = np.stack([img.datos[i] for i in indices], axis=-1).astype("float32")
    for c in range(3):
        canal = rgb[..., c]
        vals = canal[img.valida]
        if vals.size:
            bajo, alto = np.percentile(vals, [2, 98])
            rgb[..., c] = np.clip((canal - bajo) / (alto - bajo + 1e-9), 0, 1)
    rgb[~img.valida] = 0
    return rgb


def preparar(img: Imagen, radar: bool = False) -> np.ndarray:
    """Pone cada banda en una escala común (media 0, desviación 1), con valores extremos recortados."""
    datos = img.datos.copy()
    if radar and np.nanmedian(datos[:, img.valida]) > 0:
        datos = 10 * np.log10(np.clip(datos, 1e-6, None))     # radar lineal → decibeles
    for b in range(datos.shape[0]):
        vals = datos[b][img.valida]
        if vals.size == 0:
            continue
        bajo, alto = np.percentile(vals, [1, 99])
        banda = np.clip(datos[b], bajo, alto)
        recortados = banda[img.valida]
        datos[b] = (banda - recortados.mean()) / (recortados.std() + 1e-6)
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

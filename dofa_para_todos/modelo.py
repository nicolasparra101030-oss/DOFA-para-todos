"""Carga de DOFA y obtención de los resúmenes (embeddings) de cada parche."""
from __future__ import annotations

import numpy as np

from .imagenes import PARCHES, TAM_VENTANA, lado_ventana

TAMANOS = {"Base (recomendado)": "base", "Grande (más preciso, más lento)": "large"}
_cache: dict = {}


def cargar_modelo(tamano: str = "base"):
    """Carga DOFA con sus pesos preentrenados (se descargan solos la primera vez)."""
    if tamano in _cache:
        return _cache[tamano]
    import torch
    from torchgeo.models import (DOFABase16_Weights, DOFALarge16_Weights,
                                 dofa_base_patch16_224, dofa_large_patch16_224)

    if tamano == "large":
        modelo = dofa_large_patch16_224(weights=DOFALarge16_Weights.DOFA_MAE)
    else:
        modelo = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = modelo.to(dispositivo).eval()
    _cache[tamano] = modelo
    return modelo


def resumenes_por_parche(modelo, ventanas, ondas: list[float]):
    """Resumen de cada parche de 16 × 16 píxeles, en vez del promedio de la ventana.

    Junta dos capas de los bloques Transformer compartidos:
    - una capa intermedia, que describe sobre todo lo que hay en el propio parche (información local);
    - la última capa, que agrega el contexto de toda la ventana.
    Usar solo la última capa hace que lo más llamativo de la ventana (por ejemplo, una nube) "contagie" a todos
    sus parches.

    Entrada: (n, bandas, 224, 224). Salida: (n, 196, 2 × dimensión).
    """
    import torch
    import torch.nn.functional as F

    x, _ = modelo.patch_embed(ventanas, torch.tensor(ondas, device=ventanas.device).float())
    x = x + modelo.pos_embed[:, 1:, :]
    cls = (modelo.cls_token + modelo.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    mitad = len(modelo.blocks) // 2
    intermedia = None
    for i, bloque in enumerate(modelo.blocks):
        x = bloque(x)
        if i == mitad - 1:
            intermedia = x[:, 1:, :]
    normalizar = modelo.fc_norm if getattr(modelo, "global_pool", True) else modelo.norm
    final = normalizar(x[:, 1:, :])
    local = F.layer_norm(intermedia, intermedia.shape[-1:])
    return torch.cat((local, final), dim=-1)


def _pesos_parches() -> np.ndarray:
    """Peso de cada parche dentro de su ventana: más alto en el centro, más bajo en los bordes."""
    v = np.hanning(PARCHES + 2)[1:-1]
    return np.outer(v, v).astype("float32")


def posiciones(tamano: int, lado: int, paso: int) -> list[int]:
    """Inicios de las ventanas a lo largo de un eje, con superposición y cubriendo todo el eje."""
    if tamano <= lado:
        return [0]
    inicios = list(range(0, tamano - lado + 1, paso))
    if inicios[-1] + lado < tamano:
        inicios.append(inicios[-1] + paso)
    return inicios


def resumir_imagen(modelo, datos: np.ndarray, ondas: list[float], detalle: int = 1, lote: int = 16,
                   progreso=None, extractor=resumenes_por_parche, superponer: bool = True) -> np.ndarray:
    """Pasa toda la imagen por DOFA y devuelve una malla (filas, columnas, dimensión) de resúmenes.

    Con `superponer`, las ventanas se solapan a la mitad y cada celda promedia los resúmenes de las ventanas que la
    cubren (dando más peso al centro de cada ventana). Así no se notan los bordes de las ventanas en el mapa.
    """
    import torch
    import torch.nn.functional as F

    if len(ondas) != datos.shape[0]:
        raise ValueError(f"La imagen tiene {datos.shape[0]} bandas pero hay {len(ondas)} longitudes de onda.")
    lado = lado_ventana(detalle)
    celda = lado // PARCHES
    paso = lado // 2 if superponer else lado
    _, alto, ancho = datos.shape
    filas_v, cols_v = posiciones(alto, lado, paso), posiciones(ancho, lado, paso)
    alto_p, ancho_p = filas_v[-1] + lado, cols_v[-1] + lado
    modo = "reflect" if alto_p - alto < alto and ancho_p - ancho < ancho else "edge"
    datos = np.pad(datos, ((0, 0), (0, alto_p - alto), (0, ancho_p - ancho)), mode=modo)

    inicios = [(f, c) for f in filas_v for c in cols_v]
    pesos = _pesos_parches() if superponer else np.ones((PARCHES, PARCHES), dtype="float32")
    suma, cuenta = None, np.zeros((alto_p // celda, ancho_p // celda), dtype="float32")
    dispositivo = next(modelo.parameters()).device
    with torch.no_grad():
        for i in range(0, len(inicios), lote):
            grupo = inicios[i:i + lote]
            v = torch.from_numpy(np.stack([datos[:, f:f + lado, c:c + lado] for f, c in grupo])).float().to(dispositivo)
            if v.shape[-1] != TAM_VENTANA:
                v = F.interpolate(v, size=TAM_VENTANA, mode="bilinear", align_corners=False)
            r = extractor(modelo, v, ondas).float().cpu().numpy()          # (n, 196, dim)
            r = r.reshape(len(grupo), PARCHES, PARCHES, -1)
            if suma is None:
                suma = np.zeros((alto_p // celda, ancho_p // celda, r.shape[-1]), dtype="float32")
            for (f, c), resumen in zip(grupo, r):
                fc, cc = f // celda, c // celda
                suma[fc:fc + PARCHES, cc:cc + PARCHES] += resumen * pesos[..., None]
                cuenta[fc:fc + PARCHES, cc:cc + PARCHES] += pesos
            if progreso is not None:
                progreso(min(i + lote, len(inicios)) / len(inicios))
    return (suma / np.maximum(cuenta, 1e-6)[..., None]).astype("float16")

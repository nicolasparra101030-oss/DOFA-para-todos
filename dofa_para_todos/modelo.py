"""Carga de DOFA y obtención de los resúmenes (embeddings) de cada parche."""
from __future__ import annotations

import numpy as np

from .imagenes import PARCHES, TAM_VENTANA, cortar_ventanas

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
    """Igual que DOFA.forward_features, pero devuelve el resumen de cada parche en vez del promedio.

    Entrada: (n, bandas, 224, 224). Salida: (n, 196, dimensión).
    """
    import torch

    x, _ = modelo.patch_embed(ventanas, torch.tensor(ondas, device=ventanas.device).float())
    x = x + modelo.pos_embed[:, 1:, :]
    cls = (modelo.cls_token + modelo.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    for bloque in modelo.blocks:
        x = bloque(x)
    x = x[:, 1:, :]
    normalizar = modelo.fc_norm if getattr(modelo, "global_pool", True) else modelo.norm
    return normalizar(x)


def resumir_imagen(modelo, datos: np.ndarray, ondas: list[float], detalle: int = 1,
                   lote: int = 16, progreso=None, extractor=resumenes_por_parche) -> np.ndarray:
    """Pasa toda la imagen por DOFA. Devuelve una malla (filas, columnas, dimensión) de resúmenes."""
    import torch
    import torch.nn.functional as F

    if len(ondas) != datos.shape[0]:
        raise ValueError(f"La imagen tiene {datos.shape[0]} bandas pero hay {len(ondas)} longitudes de onda.")
    ventanas, nf, nc = cortar_ventanas(datos, detalle)
    dispositivo = next(modelo.parameters()).device
    partes = []
    with torch.no_grad():
        for i in range(0, len(ventanas), lote):
            v = torch.from_numpy(ventanas[i:i + lote]).float().to(dispositivo)
            if v.shape[-1] != TAM_VENTANA:
                v = F.interpolate(v, size=TAM_VENTANA, mode="bilinear", align_corners=False)
            partes.append(extractor(modelo, v, ondas).float().cpu().numpy().astype("float16"))
            if progreso is not None:
                progreso(min(i + lote, len(ventanas)) / len(ventanas))
    r = np.concatenate(partes)                                   # (n, 196, dim)
    dim = r.shape[-1]
    r = r.reshape(nf, nc, PARCHES, PARCHES, dim).transpose(0, 2, 1, 3, 4)
    return r.reshape(nf * PARCHES, nc * PARCHES, dim)

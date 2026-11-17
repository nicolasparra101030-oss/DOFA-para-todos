"""Tareas disponibles: agrupar zonas parecidas (A) y clasificar coberturas con ejemplos (B)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIN_DATO = -1


@dataclass
class Resultado:
    mapa: np.ndarray                      # (filas, columnas) con el número de clase o grupo; -1 = sin dato
    nombres: list[str]                    # nombre de cada clase o grupo
    informe: str = ""                     # texto para mostrar al usuario
    confianza: np.ndarray | None = None   # (filas, columnas) entre 0 y 1, solo en la tarea B
    extras: dict = field(default_factory=dict)


def suavizar(mapa: np.ndarray, n_clases: int, tamano: int = 3, pasadas: int = 1) -> np.ndarray:
    """Filtro de mayoría: cada celda toma la clase más común a su alrededor. Quita el efecto 'sal y pimienta'."""
    from scipy.ndimage import uniform_filter

    if n_clases < 2:
        return mapa
    resultado = mapa.copy()
    validas = mapa != SIN_DATO
    for _ in range(pasadas):
        votos = np.stack([uniform_filter((resultado == c).astype("float32"), size=tamano, mode="nearest")
                          for c in range(n_clases)])
        resultado = np.where(validas, votos.argmax(axis=0), SIN_DATO).astype(mapa.dtype)
    return resultado


def _reducir(x: np.ndarray, dims: int = 50, semilla: int = 0):
    from sklearn.decomposition import PCA

    dims = min(dims, x.shape[1], max(1, x.shape[0] - 1))
    muestra = x if len(x) <= 50000 else x[np.random.default_rng(semilla).choice(len(x), 50000, replace=False)]
    return PCA(n_components=dims, random_state=semilla).fit(muestra)


def agrupar(resumenes: np.ndarray, validas: np.ndarray, grupos: int = 5, semilla: int = 0) -> Resultado:
    """Tarea A: agrupa las celdas cuyos resúmenes se parecen. No necesita etiquetas."""
    from sklearn.cluster import MiniBatchKMeans

    x = resumenes[validas].astype("float32")
    if len(x) < grupos:
        raise ValueError("La imagen es demasiado pequeña para esa cantidad de grupos.")
    pca = _reducir(x, semilla=semilla)
    z = pca.transform(x)
    km = MiniBatchKMeans(n_clusters=grupos, random_state=semilla, n_init=5, batch_size=4096).fit(z)
    mapa = np.full(validas.shape, SIN_DATO, dtype="int16")
    mapa[validas] = km.labels_
    nombres = [f"Grupo {i + 1}" for i in range(grupos)]
    informe = (f"DOFA encontró **{grupos} grupos** de zonas parecidas. El modelo no sabe qué es cada grupo: "
               "compáralos con la imagen y decide tú qué cobertura representa cada color.")
    return Resultado(mapa, nombres, informe)


def leer_ejemplos(ruta: str):
    """Lee los ejemplos marcados (GeoJSON, GeoPackage, KML o shapefile en .zip)."""
    import geopandas as gpd

    if ruta.lower().endswith(".zip"):
        ruta = f"zip://{ruta}"
    gdf = gpd.read_file(ruta)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise ValueError("El archivo de ejemplos no tiene geometrías.")
    return gdf


def columnas_de_clase(gdf) -> list[str]:
    """Columnas que pueden contener el nombre de la cobertura, las más probables primero."""
    columnas = [c for c in gdf.columns if c != gdf.geometry.name]
    preferidas = [c for c in columnas if c.lower() in ("clase", "class", "cobertura", "nombre", "tipo", "label")]
    return preferidas + [c for c in columnas if c not in preferidas]


def _rasterizar(gdf, columna, filas, columnas, transform, crs):
    from rasterio.features import rasterize

    if gdf.crs is None:
        raise ValueError("El archivo de ejemplos no tiene sistema de coordenadas. Guárdalo de nuevo desde QGIS.")
    if crs is not None:
        gdf = gdf.to_crs(crs)
    gdf = gdf.reset_index(drop=True)
    nombres = sorted(gdf[columna].astype(str).unique())
    indice = {n: i + 1 for i, n in enumerate(nombres)}
    forma = (filas, columnas)
    clases = rasterize(((g, indice[str(c)]) for g, c in zip(gdf.geometry, gdf[columna])),
                       out_shape=forma, transform=transform, fill=0, dtype="int32", all_touched=False)
    poligonos = rasterize(((g, i + 1) for i, g in enumerate(gdf.geometry)),
                          out_shape=forma, transform=transform, fill=0, dtype="int32", all_touched=False)
    pequenos = (clases == 0) & rasterize(((g, 1) for g in gdf.geometry), out_shape=forma, transform=transform,
                                          fill=0, dtype="uint8", all_touched=True).astype(bool)
    # Los polígonos o puntos más pequeños que una celda también cuentan
    if pequenos.any():
        extra = rasterize(((g, indice[str(c)]) for g, c in zip(gdf.geometry, gdf[columna])),
                          out_shape=forma, transform=transform, fill=0, dtype="int32", all_touched=True)
        extra_p = rasterize(((g, i + 1) for i, g in enumerate(gdf.geometry)),
                            out_shape=forma, transform=transform, fill=0, dtype="int32", all_touched=True)
        clases = np.where(pequenos, extra, clases)
        poligonos = np.where(pequenos, extra_p, poligonos)
    return clases, poligonos, nombres


def clasificar(resumenes, validas, gdf, columna, transform, crs, semilla: int = 0) -> Resultado:
    """Tarea B: aprende de los ejemplos marcados y clasifica toda la imagen."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    filas, cols = validas.shape
    clases, poligonos, nombres = _rasterizar(gdf, columna, filas, cols, transform, crs)
    marcadas = (clases > 0) & validas
    if marcadas.sum() == 0:
        raise ValueError("Ningún ejemplo cae dentro de la imagen. Revisa que los polígonos estén sobre la zona "
                         "de la imagen y que tengan sistema de coordenadas.")
    y = clases[marcadas] - 1
    grupos = poligonos[marcadas]
    presentes = np.unique(y)
    if len(presentes) < 2:
        raise ValueError("Se necesitan ejemplos de al menos 2 coberturas distintas dentro de la imagen.")

    x_todo = resumenes[validas].astype("float32")
    pca = _reducir(x_todo, dims=100, semilla=semilla)
    x = pca.transform(resumenes[marcadas].astype("float32"))
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))

    # Validación honesta: se evalúa en polígonos que el modelo no vio al entrenar
    informe_validacion = ""
    poligonos_por_clase = [len(np.unique(grupos[y == c])) for c in presentes]
    try:
        pred = None
        if min(poligonos_por_clase) >= 2:
            n = min(5, min(poligonos_por_clase))
            try:
                pred = cross_val_predict(modelo, x, y, groups=grupos,
                                         cv=StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=semilla))
                forma = f"validación cruzada por polígonos ({n} grupos)"
            except ValueError:
                pred = None
        if pred is None:
            n = min(5, int(np.bincount(y)[presentes].min()))
            if n < 2:
                raise ValueError
            pred = cross_val_predict(modelo, x, y, cv=StratifiedKFold(n_splits=n, shuffle=True, random_state=semilla))
            forma = "validación cruzada por celdas (dibuja más polígonos por cobertura para una medida más honesta)"
        exactitud = accuracy_score(y, pred)
        reporte = classification_report(y, pred, labels=list(presentes),
                                        target_names=[nombres[c] for c in presentes], output_dict=True, zero_division=0)
        filas_f1 = "\n".join(f"| {nombres[c]} | {reporte[nombres[c]]['f1-score']:.2f} | {int((y == c).sum())} |"
                             for c in presentes)
        informe_validacion = (f"**Exactitud estimada: {exactitud * 100:.0f} %** ({forma}).\n\n"
                              f"| Cobertura | F1 | Celdas de ejemplo |\n|---|---|---|\n{filas_f1}")
    except ValueError:
        informe_validacion = "No hay suficientes ejemplos para medir la exactitud; dibuja más polígonos por cobertura."

    modelo.fit(x, y)
    z = pca.transform(x_todo)
    proba = modelo.predict_proba(z)
    mapa = np.full(validas.shape, SIN_DATO, dtype="int16")
    mapa[validas] = modelo.classes_[proba.argmax(axis=1)]
    confianza = np.zeros(validas.shape, dtype="float32")
    confianza[validas] = proba.max(axis=1)
    faltan = [nombres[i] for i in range(len(nombres)) if i not in presentes]
    aviso = f"\n\nAviso: estas coberturas no tienen ejemplos dentro de la imagen: {', '.join(faltan)}." if faltan else ""
    informe = (f"DOFA aprendió **{len(presentes)} coberturas** a partir de {len(np.unique(grupos))} ejemplos "
               f"({int(marcadas.sum())} celdas).\n\n{informe_validacion}{aviso}")
    return Resultado(mapa, nombres, informe, confianza)


SENSIBILIDADES = {"Baja (solo cambios fuertes)": 1.25, "Media (recomendado)": 1.0, "Alta (detecta cambios sutiles)": 0.8}


def umbral_otsu(valores: np.ndarray, bins: int = 256) -> float:
    """Umbral automático que mejor separa dos grupos de valores (método de Otsu)."""
    hist, bordes = np.histogram(valores, bins=bins)
    centros = (bordes[:-1] + bordes[1:]) / 2
    peso1 = np.cumsum(hist)
    peso2 = peso1[-1] - peso1
    media1 = np.cumsum(hist * centros) / np.maximum(peso1, 1)
    media2 = (np.sum(hist * centros) - np.cumsum(hist * centros)) / np.maximum(peso2, 1)
    varianza = peso1 * peso2 * (media1 - media2) ** 2
    # Si varios cortes separan igual de bien (un hueco sin valores), se toma el punto medio del hueco
    mejores = np.flatnonzero(varianza >= varianza.max() * (1 - 1e-9))
    return float(centros[mejores].mean())


def detectar_cambios(antes: np.ndarray, despues: np.ndarray, validas: np.ndarray,
                     factor: float = 1.0) -> Resultado:
    """Tarea C: compara los resúmenes de DOFA de dos fechas y marca las celdas que cambiaron.

    La medida de cambio es la distancia coseno entre los resúmenes de cada celda: 0 si son iguales,
    más alta cuanto más distinto se ve el lugar. El umbral se elige solo (método de Otsu) y la
    sensibilidad lo mueve hacia arriba o hacia abajo.
    """
    a = antes[validas].astype("float32")
    d = despues[validas].astype("float32")
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-6
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
    distancia = 1 - (a * d).sum(axis=1)
    if distancia.size < 2:
        raise ValueError("Las dos imágenes casi no se superponen. Revisa que sean de la misma zona.")
    umbral = umbral_otsu(distancia) * factor
    mapa = np.full(validas.shape, SIN_DATO, dtype="int16")
    mapa[validas] = (distancia > umbral).astype("int16")
    intensidad = np.zeros(validas.shape, dtype="float32")
    escala = np.percentile(distancia, 99) + 1e-6
    intensidad[validas] = np.clip(distancia / escala, 0, 1)
    porcentaje = 100 * (distancia > umbral).mean()
    informe = (f"DOFA comparó las dos fechas y marcó como **cambio el {porcentaje:.1f} %** del área.\n\n"
               "El mapa de intensidad muestra qué tan distinto se ve cada lugar entre una fecha y otra. "
               "**Ojo:** las nubes, sus sombras y los cambios de estación (por ejemplo, pasto verde en lluvias y seco "
               "en verano) también cuentan como cambio. Compara siempre con las imágenes.")
    return Resultado(mapa, ["Sin cambio", "Cambio"], informe, confianza=intensidad,
                     extras={"umbral": umbral, "distancia": distancia})

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

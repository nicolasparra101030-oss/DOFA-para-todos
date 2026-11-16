"""Pruebas con la arquitectura real de DOFA (sin pesos preentrenados) e imágenes sintéticas."""
import json
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dofa_para_todos import flujo, imagenes, modelo as mod, sensores  # noqa: E402


@pytest.fixture(scope="module")
def modelo_pequeno():
    import torch
    from torchgeo.models import dofa_small_patch16_224

    torch.manual_seed(0)
    return dofa_small_patch16_224().eval()


@pytest.fixture()
def imagen_sintetica(tmp_path):
    """Imagen de 4 bandas, 300 × 260 píxeles, con una zona de 'bosque' y otra de 'agua'."""
    rng = np.random.default_rng(0)
    alto, ancho = 300, 260
    datos = rng.normal(1000, 50, (4, alto, ancho)).astype("float32")
    datos[:, :, :130] += np.array([200, 400, 300, 3000])[:, None, None]    # bosque: NIR alto
    datos[:, :, 130:] += np.array([100, 150, 50, -500])[:, None, None]     # agua: NIR bajo
    datos[:, 280:, :] = 0                                                  # franja sin datos
    ruta = tmp_path / "imagen.tif"
    with rasterio.open(ruta, "w", driver="GTiff", height=alto, width=ancho, count=4, dtype="float32",
                       crs="EPSG:32618", transform=from_origin(500000, 100000, 10, 10), nodata=0) as dst:
        dst.write(datos)
    return str(ruta)


@pytest.fixture()
def ejemplos(tmp_path):
    def caja(x0, y0, x1, y1):
        return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]
    feats = []
    for i, (x0, clase) in enumerate([(500200, "bosque"), (500600, "bosque"), (501600, "agua"), (502000, "agua")]):
        feats.append({"type": "Feature", "properties": {"clase": clase, "id": i},
                      "geometry": {"type": "Polygon", "coordinates": caja(x0, 98000 + i * 300, x0 + 400, 98400 + i * 300)}})
    gj = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:32618"}}, "features": feats}
    ruta = tmp_path / "ejemplos.geojson"
    ruta.write_text(json.dumps(gj))
    return str(ruta)


def test_catalogo_y_orden_de_bandas():
    assert "Sentinel-2 (L2A)" in sensores.nombres_sensores()
    todas, defecto = sensores.opciones_bandas("Sentinel-2 (L2A)")
    assert len(defecto) == 9
    ondas = sensores.ondas_de_etiquetas("Sentinel-2 (L2A)", [defecto[2], defecto[0]])
    assert ondas == [0.665, 0.49]                     # respeta el orden elegido
    assert sensores.es_radar("Sentinel-1 (radar)")


def test_longitudes_manuales_en_nm_y_um():
    assert sensores.ondas_manuales("665, 560\n490") == [0.665, 0.56, 0.49]
    assert sensores.ondas_manuales("B4: 0,665; B8: 0,842") == [0.665, 0.842]
    assert sensores.ondas_manuales("0.665,0.56, 0.49") == [0.665, 0.56, 0.49]
    with pytest.raises(ValueError):
        sensores.ondas_manuales("rojo")


@pytest.mark.parametrize("detalle", [1, 2, 4])
def test_ventanas_y_malla(imagen_sintetica, detalle):
    img = imagenes.leer_imagen(imagen_sintetica)
    ventanas, nf, nc = imagenes.cortar_ventanas(imagenes.preparar(img), detalle)
    lado = imagenes.lado_ventana(detalle)
    assert ventanas.shape == (nf * nc, 4, lado, lado)
    filas, columnas, _ = imagenes.malla_resultados(img, detalle)
    assert filas <= nf * 14 and columnas <= nc * 14


def test_resumenes_por_parche(modelo_pequeno, imagen_sintetica):
    img = imagenes.leer_imagen(imagen_sintetica)
    malla = mod.resumir_imagen(modelo_pequeno, imagenes.preparar(img), [0.49, 0.56, 0.665, 0.842], detalle=1)
    assert malla.shape == (28, 28, 384)                # 2 × 2 ventanas de 14 × 14 parches
    assert np.isfinite(malla.astype("float32")).all()


def test_flujo_agrupar(modelo_pequeno, imagen_sintetica, tmp_path):
    salida = flujo.ejecutar(imagen_sintetica, [0.49, 0.56, 0.665, 0.842], tarea=flujo.AGRUPAR, grupos=3,
                            carpeta=str(tmp_path / "res"), modelo=modelo_pequeno)
    for ruta in salida["archivos"].values():
        assert os.path.exists(ruta)
    with rasterio.open(salida["archivos"]["geotiff"]) as src:
        assert src.crs.to_epsg() == 32618
        assert src.transform.a == 160                    # celdas de 16 × 16 píxeles de 10 m
        mapa = src.read(1)
    assert set(np.unique(mapa)) <= {0, 1, 2, 255}
    assert (mapa[-1] == 255).all()                       # la franja sin datos queda vacía
    assert abs(salida["tabla"]["Porcentaje"].sum() - 100) < 0.5


def test_flujo_clasificar(modelo_pequeno, imagen_sintetica, ejemplos, tmp_path):
    salida = flujo.ejecutar(imagen_sintetica, [0.49, 0.56, 0.665, 0.842], tarea=flujo.CLASIFICAR,
                            ruta_ejemplos=ejemplos, columna="clase", detalle=2,
                            carpeta=str(tmp_path / "res"), modelo=modelo_pequeno)
    res = salida["resultado"]
    assert res.nombres == ["agua", "bosque"]
    mapa = res.mapa
    # Sin pesos entrenados el modelo es débil; aun así, cada lado debe salir mayormente con su cobertura
    validas = mapa[:30] != -1
    izquierda, derecha = mapa[:30, :10], mapa[:30, -10:]
    assert (izquierda[validas[:, :10]] == 1).mean() > 0.6
    assert (derecha[validas[:, -10:]] == 0).mean() > 0.6
    assert "Exactitud" in salida["informe"]


def test_bandas_que_no_coinciden(modelo_pequeno, imagen_sintetica, tmp_path):
    with pytest.raises(ValueError, match="4 bandas"):
        flujo.ejecutar(imagen_sintetica, [0.49, 0.56, 0.665], carpeta=str(tmp_path), modelo=modelo_pequeno)


def test_la_app_se_construye():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import app

    assert app.crear_app() is not None

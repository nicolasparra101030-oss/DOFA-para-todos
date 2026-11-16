"""Catálogo de sensores: bandas y longitudes de onda."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

MANUAL = "Otro sensor (escribir longitudes de onda a mano)"
DESDE_IMAGEN = "Hiperespectral (leer longitudes de onda de la imagen)"


@lru_cache(maxsize=1)
def catalogo() -> dict:
    ruta = Path(__file__).with_name("sensores.yaml")
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def nombres_sensores() -> list[str]:
    return list(catalogo().keys()) + [MANUAL, DESDE_IMAGEN]


def es_radar(sensor: str) -> bool:
    return bool(catalogo().get(sensor, {}).get("radar", False))


def bandas(sensor: str) -> list[tuple[str, str, float]]:
    """Lista de (nombre, descripción, longitud de onda en µm) del sensor."""
    return [tuple(b) for b in catalogo().get(sensor, {}).get("bandas", [])]


def etiqueta_banda(banda: tuple[str, str, float]) -> str:
    nombre, descripcion, onda = banda
    return f"{nombre} · {descripcion} ({onda} µm)"


def opciones_bandas(sensor: str) -> tuple[list[str], list[str]]:
    """Etiquetas de todas las bandas y las que vienen marcadas por defecto."""
    todas = bandas(sensor)
    defecto = catalogo().get(sensor, {}).get("por_defecto", [b[0] for b in todas])
    por_nombre = {b[0]: b for b in todas}
    return [etiqueta_banda(b) for b in todas], [etiqueta_banda(por_nombre[n]) for n in defecto if n in por_nombre]


def ondas_de_etiquetas(sensor: str, etiquetas: list[str]) -> list[float]:
    """Longitudes de onda, en el mismo orden en que el usuario eligió las bandas."""
    por_etiqueta = {etiqueta_banda(b): b[2] for b in bandas(sensor)}
    return [float(por_etiqueta[e]) for e in etiquetas]


def a_micrometros(valor: float) -> float:
    """Acepta micrómetros o nanómetros (valores mayores a 100 se toman como nm), salvo el radar."""
    return valor / 1000 if valor > 100 else valor


def ondas_manuales(texto: str) -> list[float]:
    """Lee longitudes de onda escritas a mano: '0.665, 0.56, 0.49', una por línea, o 'B4: 665'."""
    texto = texto or ""
    # Si hay puntos decimales, la coma separa bandas; si no, la coma pegada a números es decimal ("0,665")
    separador = r"[,;\n]+" if "." in texto else r"[;\n]+|,\s+"
    valores = []
    for parte in re.split(separador, texto):
        parte = parte.strip()
        if not parte:
            continue
        numero = re.findall(r"[-+]?\d+(?:[.,]\d+)?", parte.split(":")[-1])
        if not numero:
            raise ValueError(f"No entendí '{parte}'. Escribe un número por banda, por ejemplo 0.665")
        valores.append(a_micrometros(float(numero[-1].replace(",", "."))))
    return valores


def ondas_desde_metadatos(ruta: str) -> list[float] | None:
    """Intenta leer las longitudes de onda de los metadatos de la imagen (ENVI o etiquetas por banda)."""
    import rasterio

    with rasterio.open(ruta) as src:
        envi = src.tags(ns="ENVI")
        texto = envi.get("wavelength") or src.tags().get("wavelength")
        if texto:
            numeros = re.findall(r"[-+]?\d+(?:\.\d+)?", texto)
            if len(numeros) == src.count:
                return [a_micrometros(float(n)) for n in numeros]
        valores = []
        for i in range(1, src.count + 1):
            etiquetas = src.tags(i)
            dato = etiquetas.get("wavelength") or etiquetas.get("WAVELENGTH") or etiquetas.get("CENTRAL_WAVELENGTH")
            if dato is None:
                return None
            valores.append(a_micrometros(float(re.findall(r"[-+]?\d+(?:\.\d+)?", str(dato))[0])))
        return valores

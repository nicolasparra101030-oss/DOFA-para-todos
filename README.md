# 🛰️ DOFA para todos

**Usa el modelo de inteligencia artificial DOFA con tus imágenes satelitales, sin escribir código.**

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolasparra101030-oss/DOFA-para-todos/blob/main/DOFA_para_todos.ipynb)

DOFA (*Dynamic One-For-All*) es un modelo fundacional multimodal para la observación de la Tierra, creado en la
Universidad Técnica de Múnich. Su gran ventaja es que **un solo modelo sirve para cualquier sensor**: solo necesita
saber la longitud de onda de cada banda de la imagen. *DOFA para todos* lo pone en una aplicación tipo asistente que
se abre desde Google Colab, con GPU gratuita.

## Cómo usarlo

1. Pulsa el botón **Abrir en Colab**.
2. En Colab: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU → Guardar**.
3. Ejecuta la única celda del notebook. En 2 a 3 minutos aparece la aplicación.
4. Sigue los pasos de la aplicación:
   1. **¿Qué quieres hacer?** Agrupar zonas parecidas o clasificar coberturas con tus ejemplos.
   2. **¿Qué sensor usas y qué bandas tiene tu imagen?** Las longitudes de onda se cargan solas.
   3. **Sube tu imagen** (GeoTIFF).
   4. **Opciones de la tarea** y pulsa **Ejecutar DOFA**.
5. Descarga el mapa en GeoTIFF (se abre directo en QGIS o ArcGIS), su leyenda y la tabla de hectáreas.

## Qué puedes hacer

| Tarea | Qué necesitas | Qué obtienes |
|---|---|---|
| **A · Agrupar zonas parecidas** (sin etiquetas) | Solo la imagen | Un mapa con N grupos de zonas parecidas. Tú decides qué cobertura es cada grupo. |
| **B · Clasificar coberturas** (con tus ejemplos) | La imagen + polígonos de ejemplo | Un mapa de coberturas con tus nombres, la exactitud estimada y un mapa de confianza. |

## Qué preparar

### La imagen (tareas A y B)

- Un **GeoTIFF con varias bandas** en un solo archivo, por ejemplo un recorte de Sentinel-2 descargado del
  [Copernicus Browser](https://browser.dataspace.copernicus.eu) o exportado desde QGIS o Google Earth Engine.
- **Anota el orden de las bandas** del archivo: en la aplicación debes elegirlas en ese mismo orden.
- Tamaño recomendado para empezar: entre 500 × 500 y 3000 × 3000 píxeles (con Sentinel-2, de 5 a 30 km de lado).

### Los ejemplos (solo tarea B)

1. En QGIS, abre tu imagen y crea una capa nueva de **polígonos** (GeoPackage o shapefile).
2. Agrega un campo de texto llamado **`clase`**.
3. Dibuja polígonos sobre zonas que conozcas bien y escribe su cobertura en `clase`: `bosque`, `pasto`, `agua`, `cultivo`, `urbano`...
   - Al menos **3 a 5 polígonos por cobertura**, repartidos por la imagen (no todos juntos).
   - Que cada polígono sea **grande respecto a una celda del mapa**: con el detalle normal, una celda mide
     16 × 16 píxeles (160 m con Sentinel-2). Polígonos de 300 × 300 m o más funcionan bien.
   - De 3 a 6 coberturas para empezar.
4. Guarda la capa como **GeoJSON** o **GeoPackage** (o el shapefile comprimido en `.zip`). El sistema de coordenadas
   puede ser cualquiera: la aplicación lo ajusta al de la imagen.

## Sensores incluidos

Sentinel-2 · Landsat 8/9 · Landsat 5/7 · Sentinel-1 (radar) · PlanetScope (4 y 8 bandas) · IKONOS · foto RGB
(dron o aérea) · dron multiespectral. Además: **escribir las longitudes de onda a mano** para cualquier otro sensor,
o **leerlas de los metadatos** de una imagen hiperespectral.

Para agregar un sensor, edita [`dofa_para_todos/sensores.yaml`](dofa_para_todos/sensores.yaml): son unas pocas
líneas con el nombre, la descripción y la longitud de onda de cada banda.

## Cómo funciona por dentro

1. Cada banda se normaliza (media 0, desviación 1).
2. La imagen se corta en ventanas de 224 × 224 píxeles y cada ventana pasa por DOFA con las longitudes de onda de sus bandas.
3. En lugar de un solo resumen por ventana, se toma el **resumen de cada parche de 16 × 16 píxeles**
   (14 × 14 parches por ventana). Así el mapa es mucho más detallado. Cada resumen junta dos capas de los bloques
   Transformer compartidos: una **intermedia**, que describe lo que hay en el propio parche, y la **última**, que
   aporta el contexto de la ventana. Las ventanas se **superponen a la mitad** y se promedian, para que no se noten
   sus bordes en el mapa.
4. **Tarea A:** los resúmenes se agrupan con *k-means*. **Tarea B:** una regresión logística (la *cabeza de tarea*)
   aprende de tus polígonos; DOFA no se reentrena. La exactitud se estima con validación cruzada por polígonos.

5. Un **filtro de mayoría** suaviza el mapa: cada celda toma la clase más común a su alrededor, lo que quita las celdas
   sueltas (efecto "sal y pimienta"). Se puede desactivar en las opciones avanzadas.

Con **más detalle** (opciones avanzadas), cada celda del mapa mide 8 × 8 o 4 × 4 píxeles, a cambio de más tiempo.

**Consejos:** en ciudades o zonas muy variadas, usa de 4 a 6 grupos. Las nubes y sus sombras suelen formar sus
propios grupos: es normal, y sirve para identificarlas.

## Limitaciones conocidas

- El mapa se arma por celdas, no píxel a píxel.
- Imágenes muy grandes pueden agotar la memoria de Colab: recorta la zona de estudio o baja el nivel de detalle.
- **Sentinel-1:** el repositorio oficial de DOFA usa `5.405` como valor para VV y VH, mientras que la documentación
  de TorchGeo 0.10 indica `3.75` para los pesos v1. Este proyecto usa `5.405`; si tus resultados con radar no son
  buenos, prueba `3.75` con la opción manual.
- Los resultados dependen de la calidad de la imagen (nubes, sombras) y de los ejemplos.

## Para desarrolladores

```bash
pip install -r requirements.txt pytest
pytest tests          # pruebas con la arquitectura real de DOFA, sin pesos, e imágenes sintéticas
python app.py         # abre la aplicación
```

También se puede usar sin interfaz:

```python
from dofa_para_todos import flujo
salida = flujo.ejecutar("mi_imagen.tif", [0.49, 0.56, 0.665, 0.842], tarea=flujo.AGRUPAR, grupos=6)
print(salida["informe"]); print(salida["tabla"])
```

## Créditos

- **DOFA:** Xiong, Z., Wang, Y., Zhang, F., Stewart, A. J., Hanna, J., Borth, D., Papoutsis, I., Le Saux, B.,
  Camps-Valls, G. y Zhu, X. X. (2024). *Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation*.
  [arXiv:2403.15356](https://arxiv.org/abs/2403.15356) · [Repositorio oficial](https://github.com/zhu-xlab/DOFA) (licencia MIT).
- **TorchGeo**, que distribuye el modelo y sus pesos.
- *DOFA para todos* fue creado por Juan Nicolás Parra Melo, Universidad Distrital Francisco José de Caldas. Licencia MIT.

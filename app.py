"""DOFA para todos: interfaz tipo asistente (se abre desde Google Colab)."""
from __future__ import annotations

import gradio as gr
import numpy as np

from dofa_para_todos import flujo, imagenes, sensores, tareas
from dofa_para_todos.modelo import TAMANOS

INTRO = """
# 🛰️ DOFA para todos
Usa **DOFA**, un modelo de inteligencia artificial para imágenes satelitales, sin escribir código.
Sigue los pasos en orden. DOFA funciona con cualquier sensor: solo necesita saber **la longitud de onda de cada banda**.
"""

AYUDA_EJEMPLOS = """
**¿Qué subir?** Un archivo con polígonos (o puntos) dibujados sobre tu imagen en QGIS, con una columna que diga
la cobertura de cada uno (por ejemplo `clase`: bosque, pasto, agua). Formatos: GeoJSON, GeoPackage, KML
o shapefile comprimido en `.zip`. Dibuja **varios polígonos por cobertura**, repartidos por la imagen.
"""

DETALLES = {"Normal (más rápido)": 1, "Alto (4 veces más lento)": 2, "Muy alto (16 veces más lento)": 4}
SUAVIZADOS = {"Sin suavizar": 0, "Suave (recomendado)": 1, "Fuerte": 3}


def _tabla_bandas(sensor, elegidas, manual, ruta_imagen):
    try:
        ondas = _ondas(sensor, elegidas, manual, ruta_imagen)
    except Exception as e:
        return f"⚠️ {e}"
    if not ondas:
        return "Todavía no hay bandas elegidas."
    nombres = elegidas if sensor in sensores.catalogo() else [f"Banda {i + 1}" for i in range(len(ondas))]
    filas = "\n".join(f"| {i + 1} | {n.split(' · ')[0]} | {o:g} µm |" for i, (n, o) in enumerate(zip(nombres, ondas)))
    return f"**Orden que recibirá DOFA** ({len(ondas)} bandas):\n\n| # | Banda | Longitud de onda |\n|---|---|---|\n{filas}"


def _ondas(sensor, elegidas, manual, ruta_imagen):
    if sensor == sensores.MANUAL:
        return sensores.ondas_manuales(manual)
    if sensor == sensores.DESDE_IMAGEN:
        if not ruta_imagen:
            raise ValueError("Sube la imagen (paso 3) para leer sus longitudes de onda.")
        ondas = sensores.ondas_desde_metadatos(ruta_imagen)
        if not ondas:
            raise ValueError("La imagen no trae longitudes de onda en sus metadatos. Elige la opción manual.")
        return ondas
    return sensores.ondas_de_etiquetas(sensor, elegidas or [])


def _ruta(archivo):
    if archivo is None:
        return None
    return archivo if isinstance(archivo, str) else getattr(archivo, "name", None)


def al_cambiar_sensor(sensor):
    en_catalogo = sensor in sensores.catalogo()
    todas, defecto = sensores.opciones_bandas(sensor) if en_catalogo else ([], [])
    descripcion = sensores.catalogo().get(sensor, {}).get("descripcion", "")
    if sensor == sensores.MANUAL:
        descripcion = "Escribe una longitud de onda por banda, en el orden del archivo (µm o nm)."
    elif sensor == sensores.DESDE_IMAGEN:
        descripcion = "Se leerán de los metadatos de la imagen (formato ENVI o etiquetas por banda)."
    return (gr.update(choices=todas, value=defecto, visible=en_catalogo),
            gr.update(visible=sensor == sensores.MANUAL),
            descripcion)


def _revisar_bandas(ruta, img, sensor, elegidas, manual):
    try:
        n = len(_ondas(sensor, elegidas, manual, ruta))
    except Exception:
        return ""
    if n and n != img.bandas:
        return f"\n\n⚠️ La imagen tiene **{img.bandas} bandas** pero elegiste **{n}**. Ajusta el paso 2."
    return "\n\n✅ El número de bandas coincide." if n else ""


def al_cambiar_bandas(archivo, sensor, elegidas, manual):
    """Recalcula el aviso de bandas y la tabla cuando el usuario cambia el sensor o las bandas."""
    ruta = _ruta(archivo)
    tabla = _tabla_bandas(sensor, elegidas, manual, ruta)
    if not ruta:
        return gr.update(), tabla
    try:
        img = imagenes.leer_imagen(ruta)
    except Exception:
        return gr.update(), tabla
    return imagenes.describir(img) + _revisar_bandas(ruta, img, sensor, elegidas, manual), tabla


def al_subir_imagen(archivo, sensor, elegidas, manual):
    ruta = _ruta(archivo)
    if not ruta:
        return "Sube una imagen GeoTIFF.", None, _tabla_bandas(sensor, elegidas, manual, None)
    try:
        img = imagenes.leer_imagen(ruta)
    except Exception as e:
        return f"⚠️ No pude leer la imagen: {e}", None, ""
    texto = imagenes.describir(img) + _revisar_bandas(ruta, img, sensor, elegidas, manual)
    try:
        ondas = _ondas(sensor, elegidas, manual, ruta)
    except Exception:
        ondas = None
    vista = (imagenes.vista_color(img, ondas=ondas) * 255).astype(np.uint8)
    return texto, vista, _tabla_bandas(sensor, elegidas, manual, ruta)


def al_subir_imagen2(archivo, sensor, elegidas, manual):
    """Imagen 'después' de la tarea de cambios: misma revisión que la primera."""
    texto, vista, _ = al_subir_imagen(archivo, sensor, elegidas, manual)
    return texto, vista


def al_cambiar_tarea(tarea):
    cambios = tarea == flujo.CAMBIOS
    return (gr.update(visible=tarea == flujo.AGRUPAR), gr.update(visible=tarea == flujo.CLASIFICAR),
            gr.update(visible=cambios), gr.update(visible=cambios),
            gr.update(label="Imagen de antes" if cambios else "Imagen"),
            "### 3 · Sube tus dos imágenes (GeoTIFF): antes y después" if cambios else "### 3 · Sube tu imagen (GeoTIFF)")


def al_subir_ejemplos(archivo):
    ruta = _ruta(archivo)
    if not ruta:
        return gr.update(choices=[], value=None), ""
    try:
        gdf = tareas.leer_ejemplos(ruta)
    except Exception as e:
        return gr.update(choices=[], value=None), f"⚠️ No pude leer los ejemplos: {e}"
    columnas = tareas.columnas_de_clase(gdf)
    valor = columnas[0] if columnas else None
    resumen = _resumen_ejemplos(gdf, valor)
    return gr.update(choices=columnas, value=valor), resumen


def _resumen_ejemplos(gdf, columna):
    if columna is None:
        return "⚠️ El archivo no tiene columnas con atributos. Agrega una columna con el nombre de la cobertura."
    conteo = gdf[columna].astype(str).value_counts()
    filas = "\n".join(f"| {k} | {v} |" for k, v in conteo.items())
    return f"{len(gdf)} ejemplos encontrados:\n\n| Cobertura | Polígonos |\n|---|---|\n{filas}"


def al_cambiar_columna(archivo, columna):
    ruta = _ruta(archivo)
    if not ruta or not columna:
        return ""
    return _resumen_ejemplos(tareas.leer_ejemplos(ruta), columna)


def ejecutar(tarea, sensor, elegidas, manual, archivo, grupos, ejemplos, columna, detalle, tamano,
             suavizado="Suave (recomendado)", archivo2=None, sensibilidad="Media (recomendado)",
             progress=gr.Progress()):
    ruta = _ruta(archivo)
    if not ruta:
        raise gr.Error("Falta la imagen (paso 3).")
    try:
        ondas = _ondas(sensor, elegidas, manual, ruta)
        if not ondas:
            raise ValueError("Elige las bandas de tu imagen (paso 2).")
        salida = flujo.ejecutar(
            ruta, ondas, tarea=tarea, radar=sensores.es_radar(sensor), detalle=DETALLES[detalle],
            tamano=TAMANOS[tamano], grupos=int(grupos), ruta_ejemplos=_ruta(ejemplos), columna=columna,
            suavizado=SUAVIZADOS[suavizado], ruta_imagen2=_ruta(archivo2),
            sensibilidad=tareas.SENSIBILIDADES[sensibilidad],
            progreso=lambda f, texto="": progress(f, desc=texto))
    except ValueError as e:
        raise gr.Error(str(e))
    a = salida["archivos"]
    return a["figura"], salida["informe"], salida["tabla"], [a["geotiff"], a["leyenda"], a["areas"], a["figura"]]


def crear_app() -> gr.Blocks:
    sensor_inicial = sensores.nombres_sensores()[0]
    todas, defecto = sensores.opciones_bandas(sensor_inicial)

    with gr.Blocks(title="DOFA para todos") as app:
        gr.Markdown(INTRO)

        with gr.Group():
            gr.Markdown("### 1 · ¿Qué quieres hacer?")
            tarea = gr.Radio(flujo.TAREAS, value=flujo.AGRUPAR, show_label=False)

        with gr.Group():
            gr.Markdown("### 2 · ¿De qué sensor es tu imagen y qué bandas tiene?")
            with gr.Row():
                sensor = gr.Dropdown(sensores.nombres_sensores(), value=sensor_inicial, label="Sensor")
                descripcion = gr.Markdown(sensores.catalogo()[sensor_inicial]["descripcion"])
            bandas = gr.Dropdown(todas, value=defecto, multiselect=True,
                                 label="Bandas de tu archivo, en el mismo orden en que están guardadas",
                                 info="Quita las que tu archivo no tiene. Si el orden es distinto, bórralas y "
                                      "vuelve a elegirlas en el orden correcto.")
            manual = gr.Textbox(label="Longitudes de onda (una por banda, en orden)", visible=False,
                                placeholder="0.665, 0.56, 0.49")
            with gr.Accordion("Ver el orden y las longitudes de onda que recibirá DOFA", open=False):
                tabla_bandas = gr.Markdown(_tabla_bandas(sensor_inicial, defecto, "", None))

        with gr.Group():
            titulo_paso3 = gr.Markdown("### 3 · Sube tu imagen (GeoTIFF)")
            with gr.Row():
                imagen = gr.File(label="Imagen", file_types=[".tif", ".tiff"])
                vista = gr.Image(label="Vista previa", interactive=False, height=280)
            info_imagen = gr.Markdown()
            with gr.Column(visible=False) as fila_imagen2:
                with gr.Row():
                    imagen2 = gr.File(label="Imagen de después", file_types=[".tif", ".tiff"])
                    vista2 = gr.Image(label="Vista previa (después)", interactive=False, height=280)
                info_imagen2 = gr.Markdown("Debe ser del mismo sensor, con las mismas bandas y en el mismo orden. "
                                           "Si no coincide exactamente con la primera, se ajusta sola a su grilla.")

        with gr.Group():
            gr.Markdown("### 4 · Opciones de la tarea")
            with gr.Column(visible=True) as caja_agrupar:
                grupos = gr.Slider(2, 12, value=5, step=1, label="¿En cuántos grupos quieres dividir la imagen?",
                                   info="En ciudades o zonas muy variadas, de 4 a 6 grupos suele dar mapas más limpios.")
            with gr.Column(visible=False) as caja_clasificar:
                gr.Markdown(AYUDA_EJEMPLOS)
                ejemplos = gr.File(label="Tus ejemplos marcados",
                                   file_types=[".geojson", ".json", ".gpkg", ".kml", ".zip"])
                columna = gr.Dropdown([], label="Columna con el nombre de la cobertura")
                info_ejemplos = gr.Markdown()
            with gr.Column(visible=False) as caja_cambios:
                sensibilidad = gr.Radio(list(tareas.SENSIBILIDADES), value="Media (recomendado)",
                                        label="Sensibilidad",
                                        info="Con más sensibilidad se marcan cambios más sutiles, pero también más "
                                             "falsas alarmas (nubes, sombras, cambios de estación).")
            with gr.Accordion("Opciones avanzadas", open=False):
                detalle = gr.Radio(list(DETALLES), value="Normal (más rápido)", label="Nivel de detalle del mapa",
                                   info="Con 'Normal', cada celda del mapa mide 16 × 16 píxeles de la imagen.")
                suavizado = gr.Radio(list(SUAVIZADOS), value="Suave (recomendado)", label="Suavizado del mapa",
                                     info="Quita las celdas sueltas (efecto 'sal y pimienta').")
                tamano = gr.Radio(list(TAMANOS), value=list(TAMANOS)[0], label="Tamaño del modelo DOFA")

        boton = gr.Button("🚀 Ejecutar DOFA", variant="primary", size="lg")

        gr.Markdown("### Resultados")
        figura = gr.Image(label="Mapa", interactive=False)
        informe = gr.Markdown()
        tabla = gr.Dataframe(label="Área por cobertura o grupo", interactive=False)
        descargas = gr.File(label="Descargar (GeoTIFF para QGIS, leyenda, tabla y figura)", file_count="multiple")

        gr.Markdown("Basado en DOFA (Xiong et al., 2024) y TorchGeo · Código abierto bajo licencia MIT.")

        # Eventos
        sensor.change(al_cambiar_sensor, sensor, [bandas, manual, descripcion])
        for comp in (sensor, bandas, manual):
            comp.change(al_cambiar_bandas, [imagen, sensor, bandas, manual], [info_imagen, tabla_bandas])
        imagen.change(al_subir_imagen, [imagen, sensor, bandas, manual], [info_imagen, vista, tabla_bandas])
        tarea.change(al_cambiar_tarea, tarea, [caja_agrupar, caja_clasificar, caja_cambios, fila_imagen2, imagen,
                                               titulo_paso3])
        imagen2.change(al_subir_imagen2, [imagen2, sensor, bandas, manual], [info_imagen2, vista2])
        ejemplos.change(al_subir_ejemplos, ejemplos, [columna, info_ejemplos])
        columna.change(al_cambiar_columna, [ejemplos, columna], info_ejemplos)
        boton.click(ejecutar, [tarea, sensor, bandas, manual, imagen, grupos, ejemplos, columna, detalle, tamano,
                               suavizado, imagen2, sensibilidad],
                    [figura, informe, tabla, descargas])
    return app


if __name__ == "__main__":
    crear_app().launch(share=True, debug=True)

"""Genera DOFA_para_todos.ipynb (el notebook que se abre desde Colab)."""
import json

REPO = "https://github.com/nicolasparra101030-oss/DOFA-para-todos.git"

celdas = [
    ("markdown", """# 🛰️ DOFA para todos

Usa el modelo **DOFA** con tus imágenes satelitales **sin escribir código**.

**Solo dos pasos:**
1. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU → Guardar**.
2. Ejecuta la celda de abajo (botón ▶). En 2 a 3 minutos aparece la aplicación aquí mismo y un enlace para abrirla en otra pestaña.

La primera vez que ejecutes DOFA, descarga sus pesos (unos 450 MB)."""),
    ("code", f"""# Instala DOFA para todos y abre la aplicación
import os, sys
if not os.path.exists("/content/DOFA-para-todos"):
    !git clone -q {REPO} /content/DOFA-para-todos
else:
    !git -C /content/DOFA-para-todos pull -q
%cd /content/DOFA-para-todos
!pip install -q -r requirements.txt
sys.path.insert(0, "/content/DOFA-para-todos")

from app import crear_app
crear_app().launch(share=True, debug=True)"""),
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
    "cells": [],
}
for i, (tipo, texto) in enumerate(celdas):
    lineas = texto.split("\n")
    celda = {"cell_type": tipo, "id": f"celda{i}", "metadata": {},
             "source": [l + "\n" for l in lineas[:-1]] + [lineas[-1]]}
    if tipo == "code":
        celda.update(execution_count=None, outputs=[])
    nb["cells"].append(celda)

with open("DOFA_para_todos.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("ok")

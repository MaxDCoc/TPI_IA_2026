"""
Widget de anywidget: la interfaz del laboratorio dentro de un notebook (Colab o Jupyter).

GUÍA DE LECTURA
---------------
anywidget deja escribir la parte visual en JavaScript y CSS propios (static/app.js y static/app.css) y la parte de
cálculo en Python. Se comunican con mensajes: la interfaz manda {id, method, params} y este widget le contesta
{id, ok, result} o {id, ok: False, error}. Quien realmente responde es LabService.handle (service.py), el mismo
que usa el servidor de desarrollo, así que la interfaz funciona igual en los dos lados.

  - show(): muestra el laboratorio. En Colab hay que activar antes el gestor de widgets personalizados;
    show() lo hace solo (enable_colab()).
  - Los cálculos largos (comparar, barrer parámetros) bloquean el kernel mientras corren y la interfaz muestra
    "calculando"; cada pedido se contesta cuando termina.
"""
from __future__ import annotations

from pathlib import Path

import anywidget

from .service import LabService

STATIC = Path(__file__).resolve().parent / "static"


def enable_colab() -> bool:
    """Activa los widgets personalizados en Colab. Devuelve False si no hace falta o no se puede.

    Hace falta estar dentro del kernel de un notebook de Colab, no basta con tener instalado google.colab: en Colab,
    un subproceso (por ejemplo el pytest de la última celda) también puede importarlo, pero no tiene shell de
    IPython y el gestor de widgets no tiene dónde engancharse.
    """
    try:
        from google.colab import output
        from IPython import get_ipython
    except ImportError:
        return False
    if get_ipython() is None:
        return False
    output.enable_custom_widget_manager()
    return True


class TreeLab(anywidget.AnyWidget):
    _esm = STATIC / "app.js"
    _css = STATIC / "app.css"

    def __init__(self, service: LabService | None = None, **kwargs):
        super().__init__(**kwargs)
        self.service = service if service is not None else LabService()
        self.on_msg(self._on_message)

    def _on_message(self, _widget, content, _buffers) -> None:
        self.send(self.service.handle(content))


def show(service: LabService | None = None) -> TreeLab:
    """Crea el laboratorio y lo muestra en la celda."""
    from IPython.display import display
    enable_colab()
    widget = TreeLab(service)
    display(widget)
    return widget

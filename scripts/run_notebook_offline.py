# -*- coding: utf-8 -*-
"""
Ejecuta las celdas de código del notebook en una carpeta vacía, como si fuera un Colab recién abierto.

GUÍA DE LECTURA
---------------
Sirve para comprobar que el notebook funciona SIN el repositorio: se corre desde una carpeta vacía (que no tiene el
código del proyecto) y, si todo está bien, las propias celdas escriben los archivos, importan el programa, abren la
interfaz y entrenan el árbol. Usa el mismo motor que Jupyter y Colab (IPython), así que entiende `%%writefile`.

    python scripts/run_notebook_offline.py notebooks/laboratorio_arboles_c45.ipynb  <carpeta_vacía>

Se salta la última celda (las pruebas automáticas), porque son las del propio proyecto y tardan. Al terminar imprime
una línea "RESULT {...}" con un resumen para que otro programa (tests/test_notebook.py) lo compruebe.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def run(notebook: Path, workdir: Path) -> dict:
    from IPython.core.interactiveshell import InteractiveShell

    os.chdir(workdir)
    shell = InteractiveShell.instance()
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    executed = 0
    for number, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if '"pytest"' in source and "subprocess.run" in source:        # la celda de las pruebas del proyecto
            continue
        result = shell.run_cell(source, store_history=False)
        if not result.success:
            error = result.error_in_exec or result.error_before_exec
            raise SystemExit(f"Falló la celda {number}: {error!r}\n--- código ---\n{source[:400]}")
        executed += 1
    ns = shell.user_ns
    return {"cells_run": executed, "dataset": ns["ds"].name, "rows": int(len(ns["ds"].X)), "nodes": int(ns["tree"].n_nodes_),
            "accuracy": float(ns["tree"].score(ns["X_test"], ns["y_test"])), "widget": type(ns["lab"]).__name__,
            "root": str(Path(ns["ROOT"]).resolve())}


if __name__ == "__main__":
    summary = run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
    print("RESULT " + json.dumps(summary, ensure_ascii=False))

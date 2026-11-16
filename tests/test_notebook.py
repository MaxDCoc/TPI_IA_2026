# -*- coding: utf-8 -*-
"""Tests del notebook autocontenido: está al día, no baja código de ningún lado y funciona desde una carpeta vacía."""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "laboratorio_arboles_c45.ipynb"
BUILDER = REPO / "scripts" / "build_colab_notebook.py"
RUNNER = REPO / "scripts" / "run_notebook_offline.py"
REGENERATE = "Regeneralo con: python scripts/build_colab_notebook.py"


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_colab_notebook", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shipped():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def written_files(notebook) -> dict:
    """ruta -> contenido de cada celda %%writefile del notebook."""
    files = {}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("%%writefile "):
            header, _, body = source.partition("\n")
            files[header.removeprefix("%%writefile ").strip()] = body
    return files


def normalized(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip("\n") + "\n"


# --------------------------------------------------------------------------------------------- está al día
def test_the_shipped_notebook_is_the_one_the_builder_produces(builder, shipped):
    assert shipped == builder.build_notebook(), f"El notebook quedó desactualizado. {REGENERATE}"


def test_every_project_file_travels_inside_the_notebook(builder, shipped):
    files = written_files(shipped)
    listed, tests = builder.project_files()
    expected = set(listed) | set(tests) | {"pytest.ini"}
    assert set(files) == expected
    for path in expected:
        assert normalized(files[path]) == normalized((REPO / path).read_text(encoding="utf-8")), f"{path} cambió. {REGENERATE}"


def test_the_notebook_has_no_stored_outputs(shipped):
    # Las salidas guardadas se desactualizan (rutas del disco de quien lo ejecutó, cantidad de tests) y no son necesarias.
    assert all(not cell.get("outputs") for cell in shipped["cells"] if cell["cell_type"] == "code")


# --------------------------------------------------------------------------------------- sin dependencias externas
def test_the_notebook_never_downloads_code(shipped):
    # Se revisan las celdas que prepara el entorno y usan el programa; los archivos embebidos son el propio programa.
    own_cells = [("".join(c["source"])) for c in shipped["cells"]
                 if c["cell_type"] == "code" and not "".join(c["source"]).startswith("%%writefile ")]
    forbidden = re.compile(r"""git\b|clone|github|githubusercontent|urlopen|requests\.|wget|curl""", re.IGNORECASE)
    for source in own_cells:
        found = forbidden.findall(source)
        assert not found, f"Una celda del notebook consulta un repositorio externo: {found}\n{source[:200]}"


def test_the_only_package_the_notebook_installs_is_anywidget_and_pytest(shipped):
    installs = set()
    for cell in shipped["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and not source.startswith("%%writefile "):
            installs.update(re.findall(r'"pip", "install", "-q", "([^"]+)"', source))
    assert installs == {"anywidget", "pytest"}


# ---------------------------------------------------------------------------------- funciona desde una carpeta vacía
def test_the_notebook_runs_from_an_empty_folder(tmp_path):
    pytest.importorskip("IPython")
    pytest.importorskip("anywidget")
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}        # que no vea src/ del repositorio
    done = subprocess.run([sys.executable, str(RUNNER), str(NOTEBOOK), str(tmp_path)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
    summary = json.loads(next(line for line in done.stdout.splitlines() if line.startswith("RESULT "))[len("RESULT "):])
    assert summary["dataset"] == "Enfermedad cardíaca" and summary["rows"] == 303
    assert summary["widget"] == "TreeLab" and summary["nodes"] > 1 and 0.6 < summary["accuracy"] < 1
    assert Path(summary["root"]).is_relative_to(tmp_path.resolve())                       # trabajó en la carpeta vacía
    assert (tmp_path / "tp_ia_2026" / "src" / "tp_ia" / "c45" / "tree.py").exists()
    assert (tmp_path / "tp_ia_2026" / "data" / "uci_heart" / "processed.cleveland.data").exists()

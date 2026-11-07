"""Tests del transporte de la web: protocolo de mensajes, widget de anywidget y servidor de desarrollo."""
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tp_ia.web import LabService
from tp_ia.web.devserver import make_handler

STATIC = Path(__file__).resolve().parents[1] / "src" / "tp_ia" / "web" / "static"


# --------------------------------------------------------------------- protocolo {id, method, params}
def test_handle_answers_with_the_same_id_and_the_result():
    lab = LabService()
    reply = lab.handle({"id": 7, "method": "datasets", "params": {}})
    assert reply["id"] == 7 and reply["ok"] is True
    assert [d["key"] for d in reply["result"]["datasets"]][:2] == ["play_tennis", "iris"]
    json.dumps(reply, allow_nan=False)


def test_handle_turns_every_failure_into_an_error_reply_instead_of_raising():
    lab = LabService()
    for message, expected in [({"id": 1, "method": "no_existe"}, "ValueError"),
                              ({"id": 2, "method": "train"}, "RuntimeError"),                       # sin dataset elegido
                              ({"id": 3, "method": "select", "params": {"key": "nada"}}, "ValueError"),
                              ({"id": 4, "method": "select", "params": {"clave": "iris"}}, "TypeError"),
                              ({"id": 5}, "ValueError")]:
        reply = lab.handle(message)
        assert reply["id"] == message["id"] and reply["ok"] is False and reply["error"].startswith(expected)
    bad = lab.handle("no soy un diccionario")
    assert bad["id"] is None and bad["ok"] is False and "objeto" in bad["error"]


def test_a_failed_request_does_not_break_the_session():
    lab = LabService()
    assert lab.handle({"id": 1, "method": "select", "params": {"key": "iris"}})["ok"]
    assert not lab.handle({"id": 2, "method": "sweep", "params": {"kind": "dt", "param": "criterion", "values": ["x"]}})["ok"]
    assert lab.handle({"id": 3, "method": "train"})["ok"]


# --------------------------------------------------------------------- widget de anywidget
def test_the_widget_answers_messages_through_the_service():
    pytest.importorskip("anywidget")
    from tp_ia.web.widget import TreeLab

    sent = []
    widget = TreeLab()
    widget.send = lambda content, buffers=None: sent.append(content)
    widget._on_message(widget, {"id": 11, "method": "select", "params": {"key": "iris"}}, [])
    widget._on_message(widget, {"id": 12, "method": "train", "params": {"params": {"prune": True}}}, [])
    widget._on_message(widget, {"id": 13, "method": "inexistente"}, [])
    assert [m["id"] for m in sent] == [11, 12, 13] and [m["ok"] for m in sent] == [True, True, False]
    assert sent[0]["result"]["n_rows"] == 150 and sent[1]["result"]["tree"]["n_pruned"] >= 0
    assert widget.service.dataset is not None


def test_the_widget_carries_the_javascript_and_the_styles():
    pytest.importorskip("anywidget")
    from tp_ia.web.widget import TreeLab

    widget = TreeLab()
    assert "export default" in str(widget._esm) and "render" in str(widget._esm)
    assert ".tl " in str(widget._css) and "tl-tree" in str(widget._css)


def test_enable_colab_is_a_noop_outside_colab():
    pytest.importorskip("anywidget")
    from tp_ia.web.widget import enable_colab

    assert enable_colab() is False


# --------------------------------------------------------------------- servidor de desarrollo
@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(LabService()))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def test_the_dev_server_serves_the_page_the_script_and_the_styles(server):
    for path, kind, marker in [("/", "text/html", "app.js"), ("/app.js", "text/javascript", "export default"),
                               ("/app.css", "text/css", ".tl ")]:
        with urllib.request.urlopen(server + path, timeout=10) as response:
            assert response.status == 200 and kind in response.headers["Content-Type"]
            assert marker in response.read().decode("utf-8")
            assert response.headers["Cache-Control"] == "no-store"
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(server + "/secreto.py", timeout=10)
    assert missing.value.code == 404


def test_the_dev_server_speaks_the_same_protocol_as_the_widget(server):
    assert post(server + "/rpc", {"id": 1, "method": "select", "params": {"key": "iris"}})["result"]["n_rows"] == 150
    reply = post(server + "/rpc", {"id": 2, "method": "train", "params": {"params": {"max_depth": 2}}})
    assert reply["ok"] and reply["id"] == 2 and reply["result"]["tree"]["depth"] <= 2
    assert post(server + "/rpc", {"id": 3, "method": "no_existe"})["ok"] is False
    with pytest.raises(urllib.error.HTTPError) as wrong:
        post(server + "/otra_ruta", {})
    assert wrong.value.code == 404


def test_the_dev_server_survives_a_body_that_is_not_json(server):
    request = urllib.request.Request(server + "/rpc", data=b"esto no es json")
    with urllib.request.urlopen(request, timeout=10) as response:
        assert json.loads(response.read())["ok"] is False


# --------------------------------------------------------------------- archivos estáticos
@pytest.mark.skipif(shutil.which("node") is None, reason="Node no está instalado")
def test_the_javascript_has_valid_module_syntax():
    result = subprocess.run(["node", "--experimental-default-type=module", "--check", str(STATIC / "app.js")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_the_stylesheet_is_balanced_and_scoped_to_the_widget():
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert css.count("{") == css.count("}")
    body = css.split(";", 1)[1]                                                      # sin el @import de la tipografía
    for line in body.splitlines():
        line = line.strip()
        if line and line[0] in ".@" and "{" in line and not line.startswith(("@media", "@keyframes")):
            assert line.startswith(".tl"), f"selector fuera de .tl: {line[:60]}"

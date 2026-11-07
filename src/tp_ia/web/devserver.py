"""
Servidor de desarrollo: muestra la interfaz en un navegador común, sin Jupyter ni Colab.

GUÍA DE LECTURA
---------------
La interfaz (static/app.js) habla con Python por mensajes {id, method, params}. En Colab los transporta el widget de
anywidget; acá los transporta un POST a /rpc. En los dos casos contesta LabService.handle, así que lo que se ve en
este servidor es lo que se ve en el notebook. Sirve para desarrollar la interfaz y para probarla con un navegador.

  python src/tp_ia/web/devserver.py --port 8765       (abrir http://127.0.0.1:8765)

Los archivos estáticos se leen en cada pedido: alcanza con recargar la página para ver un cambio.
Un solo usuario a la vez (un candado serializa los pedidos, igual que el kernel de un notebook).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__ in (None, ""):                         # ejecutado como script, no como módulo
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tp_ia.web.service import LabService
else:
    from .service import LabService

STATIC = Path(__file__).resolve().parent / "static"
FILES = {"/": ("dev.html", "text/html; charset=utf-8"),
         "/app.js": ("app.js", "text/javascript; charset=utf-8"),
         "/app.css": ("app.css", "text/css; charset=utf-8")}


def make_handler(service: LabService):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path not in FILES:
                self._send(404, b"No encontrado", "text/plain; charset=utf-8")
                return
            name, content_type = FILES[path]
            self._send(200, (STATIC / name).read_bytes(), content_type)

        def do_POST(self):
            if self.path != "/rpc":
                self._send(404, b"No encontrado", "text/plain; charset=utf-8")
                return
            size = int(self.headers.get("Content-Length") or 0)
            try:
                message = json.loads(self.rfile.read(size) or b"{}")
            except json.JSONDecodeError:
                message = None
            with lock:
                reply = service.handle(message)
            self._send(200, json.dumps(reply, allow_nan=False).encode("utf-8"), "application/json")

        def log_message(self, format, *args):          # sin ruido en la consola
            pass

    return Handler


def serve(port: int = 8765, host: str = "127.0.0.1", service: LabService | None = None) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(service or LabService()))
    print(f"Laboratorio en http://{host}:{port}  (Ctrl+C para cortar)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    serve(args.port, args.host)

"""Start the local GeoTwin viewer and open it in the default browser."""
from __future__ import annotations

import socket
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE = "/web/geotwin_mvp.html"


class LocalOnlyServer(ThreadingHTTPServer):
    daemon_threads = True


handler = partial(SimpleHTTPRequestHandler, directory=str(PROJECT_ROOT))
for port in (8000, 8001, 8002, 0):
    try:
        server = LocalOnlyServer(("127.0.0.1", port), handler)
        break
    except OSError:
        continue
else:  # pragma: no cover - the ephemeral-port fallback should always bind
    raise SystemExit("Could not start the local GeoTwin web server.")

url = f"http://127.0.0.1:{server.server_address[1]}{PAGE}"
print("GeoTwin is running locally at:", url)
print("Keep this window open while using the viewer. Press Ctrl+C to stop.")
webbrowser.open(url)
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Stopping the local GeoTwin viewer.")
finally:
    server.server_close()

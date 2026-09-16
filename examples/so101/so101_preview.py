"""Browser camera preview for the SO-101 client.

LeRobot pins `opencv-python-headless`, so `cv2.imshow` has no GUI backend in
the client environment. Instead, serve an MJPEG stream: every camera side by
side plus one status line per arm, viewable at http://127.0.0.1:8102/.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import cv2
import numpy as np

_PAGE = b"""<!doctype html><title>SO-101 cameras</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui}
img{display:block;max-width:100%;margin:auto}
#msg{padding:6px 10px;color:#f5a}</style>
<div id="msg" hidden>stream lost - reconnecting...</div>
<img id="v" src="/stream" alt="camera stream">
<script>
// The stream dies whenever the policy client restarts; keep retrying so the
// tab comes back on its own instead of showing a broken image.
const img = document.getElementById('v'), msg = document.getElementById('msg');
img.onerror = () => { msg.hidden = false; setTimeout(retry, 1500); };
img.onload = () => { msg.hidden = true; };
function retry() { img.src = '/stream?t=' + Date.now(); }
setInterval(() => { if (!img.complete) return; }, 5000);
</script>""" 


class PreviewServer:
    def __init__(self, cameras: dict, port: int = 8102, host: str = "127.0.0.1",
                 status_fn: Callable[[], list[str]] | None = None, fps: float = 10.0):
        self.cameras = cameras
        self.status_fn = status_fn or (lambda: [])
        self.period = 1.0 / fps
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):  # keep the robot log readable
                pass

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(_PAGE)
                    return
                if self.path != "/stream":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        jpg = preview._render()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                        time.sleep(preview.period)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self.url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"
        threading.Thread(target=self._httpd.serve_forever, daemon=True, name="preview").start()
        print(f"[preview] camera stream at {self.url}")

    def _render(self) -> bytes:
        tiles = []
        for name, cam in self.cameras.items():
            img = cam.read()
            img = np.zeros((360, 480, 3), np.uint8) if img is None else img
            img = cv2.resize(img, (int(img.shape[1] * 360 / img.shape[0]), 360))
            cv2.putText(img, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(img)
        frame = cv2.hconcat(tiles)
        lines = self.status_fn()
        if lines:
            bar = np.zeros((24 * len(lines) + 8, frame.shape[1], 3), np.uint8)
            for i, line in enumerate(lines):
                cv2.putText(bar, line, (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            frame = cv2.vconcat([frame, bar])
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

"""
Reasoning Monitor — HTTP + SSE server for live training visualization.

Serves a single-page dashboard and streams metrics via Server-Sent Events.
No external dependencies — uses only Python stdlib.
"""

import json
import os
import threading
import time
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler


class ReasoningMonitor:
    """Thread-safe monitor that records metrics and streams them to a browser.

    Parameters
    ----------
    port : int
        HTTP port for the dashboard (default 8080).
    max_history : int
        Maximum number of data points to retain (default 2000).
    """

    def __init__(self, port=8080, max_history=2000):
        self.port = port
        self.max_history = max_history

        # Time-series data (thread-safe via lock)
        self._lock = threading.Lock()
        self._surprise_history = deque(maxlen=max_history)
        self._stress_history = deque(maxlen=max_history)
        self._timestamps = deque(maxlen=max_history)
        self._step = 0
        self._start_time = time.time()

        # Latest snapshot for dashboard
        self._current = {
            "surprise": 0.0,
            "stress": 0.0,
            "step": 0,
            "uptime_s": 0.0,
            "cache_stats": {},
            "training_stats": {},
            "device_name": "",
            "profile_name": "",
        }

        # SSE subscribers
        self._sse_lock = threading.Lock()
        self._sse_events = []  # list of threading.Event for wake-up

        self._server = None
        self._thread = None

    def record(self, surprise=0.0, stress=0.0,
               cache_stats=None, training_stats=None,
               device_name=None, profile_name=None):
        """Record a single training step's metrics.

        Call this once per forward/backward pass. The dashboard updates live.
        """
        with self._lock:
            self._step += 1
            self._surprise_history.append(surprise)
            self._stress_history.append(stress)
            self._timestamps.append(self._step)

            self._current["surprise"] = surprise
            self._current["stress"] = stress
            self._current["step"] = self._step
            self._current["uptime_s"] = time.time() - self._start_time

            if cache_stats is not None:
                self._current["cache_stats"] = dict(cache_stats)
            if training_stats is not None:
                self._current["training_stats"] = dict(training_stats)
            if device_name is not None:
                self._current["device_name"] = device_name
            if profile_name is not None:
                self._current["profile_name"] = profile_name

        # Wake all SSE subscribers
        with self._sse_lock:
            for evt in self._sse_events:
                evt.set()

    def get_snapshot(self):
        """Get current state as a JSON-serializable dict."""
        with self._lock:
            return {
                **self._current,
                "surprise_history": list(self._surprise_history),
                "stress_history": list(self._stress_history),
                "timestamps": list(self._timestamps),
            }

    def start(self):
        """Start the dashboard server in a background thread."""
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Silence request logging

            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self._serve_dashboard()
                elif self.path == "/api/snapshot":
                    self._serve_json(monitor.get_snapshot())
                elif self.path == "/api/stream":
                    self._serve_sse()
                else:
                    self.send_error(404)

            def _serve_dashboard(self):
                html_path = os.path.join(
                    os.path.dirname(__file__), "dashboard.html")
                with open(html_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content.encode())))
                self.end_headers()
                self.wfile.write(content.encode())

            def _serve_json(self, data):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _serve_sse(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                evt = threading.Event()
                with monitor._sse_lock:
                    monitor._sse_events.append(evt)

                try:
                    # Send initial snapshot
                    data = json.dumps(monitor.get_snapshot())
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()

                    while True:
                        evt.wait(timeout=2.0)
                        evt.clear()
                        data = json.dumps(monitor.get_snapshot())
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with monitor._sse_lock:
                        if evt in monitor._sse_events:
                            monitor._sse_events.remove(evt)

        self._server = HTTPServer(("0.0.0.0", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        print(f"Reasoning Monitor running at http://localhost:{self.port}")

    def stop(self):
        """Shut down the dashboard server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None

"""Forward OpenAI-style requests to Ollama with the model's thinking turned off (reasoning_effort=none).

The runner talks to this proxy (port 8080); Ollama listens on 11434. Started by services.py.
"""
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://127.0.0.1:11434"
PORT = 8080


class Handler(BaseHTTPRequestHandler):
    def _forward(self, body: bytes | None = None) -> None:
        if body is not None and self.path.endswith("/chat/completions"):
            body = json.dumps({**json.loads(body), "reasoning_effort": "none"}).encode()
        request = urllib.request.Request(UPSTREAM + self.path, data=body, method=self.command,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=900) as reply:
                status, payload, ctype = reply.status, reply.read(), reply.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as error:
            status, payload, ctype = error.code, error.read(), "application/json"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        self._forward(self.rfile.read(int(self.headers.get("Content-Length", 0))))

    def do_GET(self):
        self._forward()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

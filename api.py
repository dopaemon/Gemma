#!/usr/bin/env python3
"""OpenAI-compatible API server for the trained RE adapter.
Thin wrapper around mlx_lm's built-in server, with Bearer API-key auth added.

Usage:
    API_KEY=secret .venv/bin/python3 api.py
"""
import os
import sys

import mlx_lm.server as srv

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    sys.exit("Set the API_KEY environment variable, e.g. API_KEY=secret python3 api.py")


class AuthAPIHandler(srv.APIHandler):
    def _authorized(self):
        got = self.headers.get("Authorization", "")
        if got == f"Bearer {API_KEY}":
            return True
        self._set_completion_headers(401)
        self.end_headers()
        self.wfile.write(b'{"error": "invalid api key"}')
        return False

    def do_POST(self):
        if self._authorized():
            super().do_POST()

    def do_GET(self):
        if self.path == "/health" or self._authorized():
            super().do_GET()


_original_run_http_server = srv._run_http_server


def _run_http_server_with_auth(host, port, response_generator, **_):
    _original_run_http_server(host, port, response_generator, handler_class=AuthAPIHandler)


srv._run_http_server = _run_http_server_with_auth

# This server only ever hosts one model. Clients (Codex/9router) send whatever
# model name they were configured with, which won't match our path and would
# otherwise make mlx_lm try to load it as a literal (wrong) directory. Force
# every request onto the one model+adapter we started with.
_original_load = srv.ModelProvider.load


def _load_default_only(self, model_path, adapter_path=None, draft_model_path=None):
    return _original_load(self, "default_model", None, "default_model")


srv.ModelProvider.load = _load_default_only

sys.argv = [
    "mlx_lm.server",
    "--model", "./gemma-mlx-4bit",
    "--host", "127.0.0.1",
    "--port", "8080",
    # mlx_lm defaults to 512 when a request omits max_tokens. Codex's requests
    # (translated through 9router) never set it, so every turn was getting cut
    # off at 512 tokens regardless of the tool-calling fix above.
    "--max-tokens", "32768",
    # mlx_lm's sampling defaults (temp 0.0 = pure greedy, top-p 1.0, top-k off)
    # ignore what the model itself asks for. These three mirror the model's own
    # generation_config.json, which is Gemma's tuned configuration; greedy
    # decoding in particular makes it loop on long answers.
    "--temp", "1.0",
    "--top-p", "0.95",
    "--top-k", "64",
]
# The RE adapter is opt-in because it specializes the model for one task
# (naming decompiled functions), not because it breaks general use - measured
# at 3157 vs 3406 completion tokens on the same long-form chat prompt, both
# ending naturally. Attach it via ADAPTER_PATH=./adapters.
adapter_path = os.environ.get("ADAPTER_PATH")
if adapter_path:
    sys.argv += ["--adapter-path", adapter_path]

if __name__ == "__main__":
    srv.main()

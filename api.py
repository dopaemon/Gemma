#!/usr/bin/env python3
"""OpenAI-compatible API server for the trained RE adapter.
Thin wrapper around mlx_lm's built-in server, with Bearer API-key auth added.

Usage:
    API_KEY=secret .venv/bin/python3 api.py
"""
import io
import json
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

    def _strip_tools(self):
        # Gemma-4 has native tool calling: given a tools schema, it will
        # happily pause mid-answer to emit a <tool_call>. mlx_lm's server
        # treats that as a state-machine transition, but callers like
        # Codex/9router - built for a real tool-execution loop - just end
        # the turn there, truncating what would've been a long answer to a
        # couple hundred tokens. This server doesn't execute tools, so strip
        # the schema before it ever reaches the chat template.
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self.rfile = io.BytesIO(raw)
            return
        if "tools" in body:
            body.pop("tools", None)
            body.pop("tool_choice", None)
            raw = json.dumps(body).encode()
            self.headers.replace_header("Content-Length", str(len(raw)))
        self.rfile = io.BytesIO(raw)

    def do_POST(self):
        if self._authorized():
            if self.path in ("/v1/completions", "/v1/chat/completions", "/chat/completions"):
                self._strip_tools()
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
    model, tokenizer = _original_load(self, "default_model", None, "default_model")
    # Gemma's chat template makes tokenizer.has_tool_calling True unconditionally
    # (it's a model-level property, not derived from the request). mlx_lm uses it
    # to insert <tool_call> as a stop sequence for EVERY generation, so even after
    # _strip_tools removes the tools schema, the model still reads Codex's system
    # prompt describing tools in plain text and emits a <tool_call>, which the
    # server treats as end-of-turn. Disable it so <tool_call> is just text.
    tokenizer._tool_call_start = None
    tokenizer._tool_call_end = None
    return model, tokenizer


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
]
# The RE QLoRA adapter was trained on completions that are always a single
# function name. Loaded for every request, it biases the model to emit EOS
# after a couple hundred tokens even on unrelated general-purpose chat/coding
# tasks (e.g. via Codex) - cutting long explanations short. Only attach it
# when explicitly asked for, via ADAPTER_PATH=./adapters.
adapter_path = os.environ.get("ADAPTER_PATH")
if adapter_path:
    sys.argv += ["--adapter-path", adapter_path]

if __name__ == "__main__":
    srv.main()

"""Minimal OpenAI-compatible server for a locally hosted open model, for `llm.provider=local`.

    python tools/local_llm_server.py --model Qwen/Qwen2.5-1.5B-Instruct [--port 8000] [--device cuda]

Needs PyTorch and transformers (NOT part of this project's pinned environments: run it with any Python that has
them, e.g. a separate venv). Serves POST /v1/chat/completions (greedy decoding) and GET /v1/models using only the
standard library for HTTP. Any other local server (Ollama, LM Studio, llama.cpp, vLLM) works the same way: point
`LLM_BASE_URL` at it and set `LLM_PROVIDER=local`, `LLM_MODEL=<its model name>`.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    dtype = getattr(torch, args.dtype) if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    lock = threading.Lock()                                   # one generation at a time
    print(f"serving {args.model} on {device} ({dtype}) at http://127.0.0.1:{args.port}/v1", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"object": "list", "data": [{"id": args.model, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send(404, {"error": "not found"})
                return
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            prompt = tok.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with lock, torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=int(req.get("max_tokens", 512)), do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            new = out[0][inputs["input_ids"].shape[1]:]
            self._send(200, {
                "id": "local", "object": "chat.completion", "created": int(time.time()), "model": args.model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": tok.decode(new, skip_special_tokens=True)}}],
                "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(new.shape[0])},
            })

        def log_message(self, *a):        # quiet
            pass

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

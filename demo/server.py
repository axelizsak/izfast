import json
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hw.ref import HF_MODELS, chat_text, chip_image, qwen_config

MODEL = "qwen3.5-4b"
MAX_SEQ = 1024
LANES = 1024
STOP = [248044, 248046]
QUESTIONS = ["What is the capital of France?", "Explain how a CPU works in two sentences.", "Write a haiku about the sea.",
             "A child runs onto the road 20 m ahead of a car at 50 km/h. What should the car do? One sentence.",
             "Why is the sky blue?", "Translate 'good morning' into French, Spanish and German.",
             "Explique-moi ce qu'est un GPU en deux phrases.", "Tell me a short joke about computers."]
MARKET = [
    {"name": "Taalas HC1", "where": "datacenter card, 815 mm² at 6 nm", "model": "Llama 3.1 8B, hard-wired", "tps": 14357,
     "how": "independent test (Karl Freund); vendor claims 17,000", "url": "https://heise.de/-11185112"},
    {"name": "Snapdragon 8 Elite Gen 5", "where": "flagship phone", "model": "3B-class model", "tps": 220, "how": "Qualcomm claim",
     "url": "https://futurumgroup.com/insights/is-snapdragon-8-elite-gen-5-the-benchmark-for-next-gen-flagship-phones/"},
    {"name": "Apple M5 Max", "where": "MacBook Pro", "model": "Qwen3.5-4B, 4-bit", "tps": 148, "how": "third-party benchmark aggregate",
     "url": "https://bestllmfor.com/guide/mac-m5-llm-local-mlx-vs-llamacpp/"},
    {"name": "NVIDIA Jetson AGX Thor", "where": "robotics module", "model": "Qwen3-4B, FP4", "tps": 73.7, "how": "NVIDIA benchmark, batch 1",
     "url": "https://nvidia.github.io/TensorRT-Edge-LLM/0.9.1/user_guide/performance/performance-benchmarks.html"},
    {"name": "Hailo-10H", "where": "2.5 W accelerator", "model": "2B models", "tps": 10, "how": "Hailo claim", "url": "https://hailo.ai/"},
]
LOCK = threading.Lock()
JOB = {"phase": "idle"}


class Text:
    def __init__(self):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(HF_MODELS[MODEL] / "tokenizer.json"))

    def encode(self, prompt):
        return self.tok.encode(chat_text(MODEL, prompt), add_special_tokens=False).ids

    def pieces(self, ids):
        out, prev = [], ""
        for k in range(1, len(ids) + 1):
            s = self.tok.decode([t for t in ids[:k] if t not in STOP])
            if s.endswith("�"):
                out.append("")
                continue
            out.append(s[len(prev):])
            prev = s
        return out


class Emu:
    def __init__(self):
        self.p = None

    def run(self, ids, n, progress):
        if self.p is None or self.p.poll() is not None:
            subprocess.run([sys.executable, str(ROOT / "hw" / "run.py"), "emu", MODEL, "--wbits", "4", "--hbits", "8"], check=True, capture_output=True)
            self.p = subprocess.Popen([str(ROOT / "hw" / "build" / "emu" / "emu"), str(chip_image(MODEL))], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
            self.p.stdout.readline()
        self.p.stdin.write(f"g {n} {len(ids)} {len(STOP)} {' '.join(map(str, ids + STOP))}\n")
        self.p.stdin.flush()
        out = []
        while (line := self.p.stdout.readline().split()) and line[0] == "t":
            out.append(int(line[1]))
            progress(len(out))
        return out

    def close(self):
        if self.p:
            self.p.kill()


TEXT, EMU = None, Emu()


def measurements():
    f = ROOT / "results" / "measurements.json"
    return json.loads(f.read_text()) if f.exists() else {}


def info():
    c, m = qwen_config(MODEL), measurements()
    s = m.get("synthesis", {})
    units = {"lanes": "mvu_asap7_w4h8_l8", "vector": "vpu_asap7_vk1", "deltanet": "dnu_asap7_vk1", "attention": "attn_asap7_vk1_bitnet"}
    clocks = {k: s[v]["fmax_mhz"] for k, v in units.items() if s.get(v, {}).get("fmax_mhz")}
    bench = m.get("cycles", {}).get(f"{MODEL}/l{LANES}", {})
    q = m.get("quality", {}).get("eval_qwen_w4h8", {})
    D, L, F, V = c["d_model"], c["n_layers"], c["d_ff"], c["vocab_size"]
    n_att = c["layer_types"].count("full_attention")
    lin = (2 * c["lk_heads"] * c["lk_dim"] + 2 * c["lv_heads"] * c["lv_dim"] + 2 * c["lv_heads"]) * D + c["lv_heads"] * c["lv_dim"] * D
    att = (2 * c["n_heads"] + 2 * c["n_kv_heads"]) * c["head_dim"] * D + c["n_heads"] * c["head_dim"] * D
    layers = (L - n_att) * lin + n_att * att + L * 3 * F * D
    return {"model": {"name": "Qwen3.5-4B", "layers": L, "attention_layers": n_att, "linear_layers": L - n_att, "n_heads": c["n_heads"],
                      "n_kv_heads": c["n_kv_heads"], "weights": layers + V * D},
            "lanes": LANES, "vk": bench.get("vk", 4), "at": bench.get("att_tok", 4), "max_seq": MAX_SEQ,
            "cycles": {int(k): v for k, v in bench.get("per_ctx", {}).items()}, "clocks": clocks,
            "weight_bytes": (layers * 4.5 + V * D * 8.5) / 8, "quality": q, "market": MARKET, "examples": QUESTIONS}


def job(req):
    ids = TEXT.encode((req.get("prompt", "") or "Hello")[:1500])
    n = max(1, min(int(req.get("tokens", 256)), MAX_SEQ - len(ids)))
    JOB.update(phase="chip", log="C emulator computing the chip's answer, bit for bit", prompt_len=len(ids), chip_n=0)
    t0 = time.time()
    chip = EMU.run(ids, n, lambda k: JOB.update(chip_n=k))
    JOB.update(phase="done", log="", chip=TEXT.pieces(chip), emu_seconds=round(time.time() - t0, 2))


def run_job(req):
    with LOCK:
        try:
            job(req)
        except Exception as e:
            JOB.update(phase="error", log=str(e))


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.reply(200, (ROOT / "demo" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self.reply(200, info())
        elif self.path == "/api/job":
            self.reply(200, dict(JOB))
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/stop":
            self.reply(200, {"ok": True})
            return threading.Thread(target=self.server.shutdown, daemon=True).start()
        if self.path != "/api/ask":
            return self.reply(404, {"error": "not found"})
        if LOCK.locked():
            return self.reply(409, {"error": "busy"})
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        JOB.clear()
        JOB.update(phase="chip", log="Starting")
        threading.Thread(target=run_job, args=(req,), daemon=True).start()
        self.reply(202, {"ok": True})

    def log_message(self, *args):
        pass


def record(path):
    runs = []
    for q in QUESTIONS:
        job({"prompt": q, "tokens": 256})
        runs.append({"prompt": q, "prompt_len": JOB["prompt_len"], "chip": JOB["chip"], "emu_seconds": JOB["emu_seconds"]})
        print(q, len(JOB["chip"]), "tokens,", JOB["emu_seconds"], "s in the emulator")
    path.write_text(json.dumps({"info": info(), "runs": runs}, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    TEXT = Text()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if "--record" in sys.argv:
        try:
            record(ROOT / "demo" / "runs.json")
        finally:
            EMU.close()
        sys.exit(0)
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"izfast demo on http://127.0.0.1:{port}")
    if "--no-browser" not in sys.argv:
        webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    finally:
        EMU.close()

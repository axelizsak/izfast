import json
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "hw" / "build"
sys.path.insert(0, str(ROOT))

from hw.ref import HF_MODELS, hf_config

MODEL = "llama-3.2-1b"
MAX_SEQ = 1024
AT = 32
CHIPS = {256: 4, 1024: 4, 4096: 16}
GGUF = ROOT / "data" / "llama-3.2-1b-instruct-gguf" / "Llama-3.2-1B-Instruct-{q}.gguf"
STOP = [128001, 128008, 128009]
QUESTIONS = ["What is the capital of France?", "Explain how a CPU works in two sentences.", "Write a haiku about the sea.",
             "Give me three tips to sleep better.", "Why is the sky blue?", "Translate 'good morning' into French, Spanish and German.",
             "Explique-moi ce qu'est un GPU en deux phrases.", "Tell me a short joke about computers."]
LOCK = threading.Lock()
JOB = {"phase": "idle"}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Text:
    def __init__(self):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(HF_MODELS[MODEL] / "tokenizer.json"))

    def encode(self, prompt):
        text = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + prompt.strip()
                + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        return self.tok.encode(text, add_special_tokens=False).ids

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
        self.procs = {}

    def run(self, wbits, ids, n, log, progress):
        p = self.procs.get(wbits)
        if p is None or p.poll() is not None:
            img = BUILD / "emu" / f"{MODEL}_w{wbits}_s{MAX_SEQ}.bin"
            if not img.exists():
                log(f"Preparing the {wbits}-bit chip image, once")
            subprocess.run([sys.executable, str(ROOT / "hw" / "run.py"), "emu", MODEL, "--wbits", str(wbits),
                            "--max-seq", str(MAX_SEQ)], check=True, capture_output=True)
            p = subprocess.Popen([str(BUILD / "emu" / "emu"), str(img)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            p.stdout.readline()
            self.procs[wbits] = p
        p.stdin.write(f"g {n} {len(ids)} {len(STOP)} {' '.join(map(str, ids + STOP))}\n")
        p.stdin.flush()
        out = []
        while (line := p.stdout.readline().split()) and line[0] == "t":
            out.append(int(line[1]))
            progress(len(out))
        return out

    def close(self):
        for p in self.procs.values():
            p.kill()


class Mac:
    def __init__(self):
        self.q, self.proc, self.port, self.lock = None, None, None, threading.Lock()

    def ensure(self, q, log):
        with self.lock:
            self.start(q, log)

    def start(self, q, log):
        if q == self.q and self.proc and self.proc.poll() is None:
            return
        self.close()
        log("Starting llama.cpp on your Mac")
        self.port = free_port()
        self.proc = subprocess.Popen(["llama-server", "-m", str(GGUF).format(q=q), "--port", str(self.port), "-ngl", "99",
                                      "-c", str(MAX_SEQ), "-np", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:
                time.sleep(0.1)
        self.q = q
        for _ in self.stream([128000, 9906, 1174], 32):
            pass

    def stream(self, ids, n):
        body = json.dumps({"prompt": ids, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False,
                           "stream": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                if raw.startswith(b"data: "):
                    yield json.loads(raw[6:])

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.q, self.proc = None, None


TEXT, EMU, MAC = None, Emu(), Mac()


def cycle_tables():
    out = {}
    for lanes, vk in CHIPS.items():
        f = BUILD / f"bench_{MODEL}_l{lanes}_v{vk}_a{AT}_s{MAX_SEQ}" / "bench.json"
        if not f.exists():
            continue
        blocks = {}
        for r in json.loads(f.read_text()):
            blocks.setdefault(r["ctx"] // AT, set()).add(r["cycles"])
        if sorted(blocks) == list(range(MAX_SEQ // AT)) and all(len(v) == 1 for v in blocks.values()):
            out[lanes] = {"vk": vk, "cycles": [blocks[b].pop() for b in range(MAX_SEQ // AT)]}
    return out


def clocks():
    out = {}
    for sta in sorted((BUILD / "synth").glob("*/sta.txt")):
        top, lib, *tag = sta.parent.name.split("_")
        unit = "_".join([top, *tag])
        m = re.search(r"FMAX_MHZ ([\d.]+)", sta.read_text())
        if m and re.fullmatch(r"mvu|(vpu|attn)(_vk\d+)?", unit):
            out.setdefault(lib, {})[unit] = float(m.group(1))
    return out


def info():
    c = hf_config(MODEL, MAX_SEQ)
    D, H, KV, hd, F = c["d_model"], c["n_heads"], c["n_kv_heads"], c["head_dim"], c["d_ff"]
    n = 2 * c["vocab_size"] * D + c["n_layers"] * ((H + 2 * KV) * hd * D + D * H * hd + 3 * F * D)
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return {"model": c, "cpu": cpu, "at": AT, "max_seq": MAX_SEQ, "cycles": cycle_tables(), "clocks": clocks(), "examples": QUESTIONS,
            "weight_bytes": {b: n * b // 8 + n // 32 * 2 for b in (4, 8)},
            "kv_bytes": c["n_layers"] * MAX_SEQ * c["n_kv_heads"] * c["head_dim"] * 2}


def job(req):
    wbits = 8 if int(req.get("wbits", 4)) == 8 else 4
    ids = TEXT.encode((req.get("prompt", "") or "Hello")[:1500])
    n = max(1, min(int(req.get("tokens", 256)), MAX_SEQ - len(ids)))
    log = lambda s: JOB.update(log=s)
    JOB.update(phase="chip", log="C emulator computing the chip's answer, bit for bit", prompt_len=len(ids), wbits=wbits, chip_n=0)
    t0 = time.time()
    chip = EMU.run(wbits, ids, n, log, lambda k: JOB.update(chip_n=k))
    JOB.update(chip=TEXT.pieces(chip), emu_seconds=round(time.time() - t0, 2))
    MAC.ensure("Q8_0" if wbits == 8 else "Q4_0", log)
    for _ in MAC.stream(ids, 1):
        pass
    pieces, times, timings = [], [], {}
    JOB.update(phase="mac", log="Your Mac is answering", mac=pieces, mac_ms=times)
    t0 = time.perf_counter()
    for ev in MAC.stream(ids, n):
        if ev.get("content"):
            times.append(round((time.perf_counter() - t0) * 1000, 2))
            pieces.append(ev["content"])
        if ev.get("stop"):
            timings = ev.get("timings", {})
    JOB.update(phase="done", log="", mac_tps=timings.get("predicted_per_second"), mac_n=timings.get("predicted_n"),
               mac_engine=f"llama.cpp, Metal GPU, {'Q8_0' if wbits == 8 else 'Q4_0'}")


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
    runs = [{"prompt": q} for q in QUESTIONS]
    for wbits in (4, 8):
        for r in runs:
            job({"prompt": r["prompt"], "wbits": wbits, "tokens": 256})
            r["prompt_len"] = JOB["prompt_len"]
            r[wbits] = {k: JOB[k] for k in ("chip", "mac", "mac_ms", "mac_tps", "mac_engine")}
            print(wbits, r["prompt"], len(JOB["chip"]), "chip tokens,", round(JOB["mac_tps"]), "tokens/second on the Mac")
    path.write_text(json.dumps({"info": info(), "runs": runs}, separators=(",", ":")))


if __name__ == "__main__":
    TEXT = Text()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if "--record" in sys.argv:
        try:
            record(ROOT / "demo" / "runs.json")
        finally:
            EMU.close()
            MAC.close()
        sys.exit(0)
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=lambda: MAC.ensure("Q4_0", lambda s: None), daemon=True).start()
    print(f"izfast demo on http://127.0.0.1:{port}")
    if "--no-browser" not in sys.argv:
        webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    finally:
        EMU.close()
        MAC.close()

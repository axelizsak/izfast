import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cocotb_tools.runner import get_runner

HW = Path(__file__).resolve().parent
ROOT = HW.parent
sys.path.insert(0, str(ROOT))

from hw.compiler import QShape, Shape, compile_model, compile_shape, pow2, qwen_in_rows, vmem_map, write_image

MODELS = {
    "tc1": Shape(8, 256, 4, 2, 64, 768, 4096, True, True, True),
    "qwen3.5-4b": QShape(32, 2560, 16, 4, 256, 64, 9216, 248320, 16, 32, 128, 128, 4, ("lin", "lin", "lin", "att") * 8),
}
SOURCES = [HW / "rtl" / f for f in ("reduce.sv", "mvu.sv", "vpu.sv", "attn.sv", "dnu.sv", "izcore.sv")]
BUILD_ARGS = ["-Wno-fatal", "-Wno-WIDTH", "-Wno-UNUSED", "-O3"]
ENV = {"PYTHONPATH": str(ROOT)}


def bits_tag(wbits, hbits, lora=0):
    return f"w{wbits}" + (f"h{hbits}" if hbits and hbits != wbits else "") + (f"r{lora}" if lora else "")


def gain_params(gain_len, vk):
    return {"GAINS": int(gain_len > 0), "GDEPTH": pow2(max(1, gain_len // (32 * vk)))}


def core_params(m, lanes, vk, att_tok, sh=1):
    from hw.ref import FixedQwen
    img = compile_model(m, lanes, banks=False)
    if isinstance(m, FixedQwen):
        n_att = m.kinds.count("att")
        return {"LANES": lanes, "VK": vk, "ATT_TOK": att_tok, "WBITS": m.wbits, "HBITS": m.hbits, "HD": m.hd, "RD": m.rd, "NKV": m.KV,
                "REP": m.H // m.KV, "NLAYERS": max(1, n_att), "MAXSEQ": m.c["max_seq"], "VDEPTH": pow2(img.vmem["end"]),
                "IDEPTH": pow2(len(img.program)), "EDEPTH": pow2(m.emb.rows * (m.d // 32)), "WDEPTH": pow2(img.words_per_lane),
                "HDEPTH": pow2(max(1, img.wide_words)), "XDEPTH": pow2(max(m.F, m.d, m.Hv * m.Dv, m.H * m.hd) // 32),
                "YDEPTH": pow2(max(m.head.rows, 2 * m.F, max(qm.rows for lay in m.layers for qm in (lay["in"],)))), "SDEPTH": pow2(img.scale_depth),
                "GAINS": 1, "GDEPTH": pow2(max(1, img.gain_len // (32 * vk))), "EPS": m.eps, "LIN": 1, "HK": m.Hk, "HV": m.Hv, "DK": m.Dk,
                "DV": m.Dv, "CK": m.K, "NLIN": max(1, m.L - n_att), "SH": sh, "LS": m.ls}
    return {"LANES": lanes, "VK": vk, "ATT_TOK": att_tok, "WBITS": m.wbits, "HBITS": m.hbits, "HD": m.hd, "NKV": m.KV, "NLAYERS": m.L,
            "MAXSEQ": m.c["max_seq"], "REP": m.H // m.KV, "VDEPTH": pow2(img.vmem["end"]), "IDEPTH": pow2(len(img.program)),
            "EDEPTH": pow2(m.emb.rows * (m.d // 32)), "WDEPTH": pow2(img.words_per_lane), "HDEPTH": pow2(max(1, img.wide_words)), "XDEPTH": pow2(max(m.F, m.d) // 32),
            "YDEPTH": pow2(max(m.head.rows, 2 * m.F, (m.H + 2 * m.KV) * m.hd)), "ACT": int(m.relu2), "SUBNORM": int(m.subnorm),
            "SDEPTH": pow2(img.scale_depth)} | gain_params(img.gain_len, vk)


def test(a):
    from hw.ref import load
    build = HW / "build" / f"test_{a.model}_{bits_tag(a.wbits, a.hbits, a.lora)}_l{a.lanes}_v{a.vk}{f'_h{a.sh}' if a.sh > 1 else ''}{'_waves' if a.waves else ''}"
    args = list(BUILD_ARGS)
    if a.waves:
        args += ["--trace-fst"]
        if Path("/opt/homebrew/include").exists():
            args += ["-CFLAGS", "-I/opt/homebrew/include", "-LDFLAGS", "-L/opt/homebrew/lib -llz4"]
    runner = get_runner("verilator")
    runner.build(sources=SOURCES, hdl_toplevel="izcore", build_dir=build, build_args=args, waves=a.waves,
                 parameters=core_params(load(a.wbits, a.model, hbits=a.hbits, lora=a.lora)[0], a.lanes, a.vk, 8, a.sh))
    runner.test(hdl_toplevel="izcore", test_module="test_core", test_dir=HW / "tests", build_dir=build, waves=a.waves,
                test_filter="test_tokens" if a.waves else None,
                extra_env=ENV | {"WBITS": str(a.wbits), "HBITS": str(a.hbits or a.wbits), "LORA": str(a.lora), "LANES": str(a.lanes), "MODEL": a.model})
    if a.waves:
        (HW / "tests" / "dump.fst").replace(build / "izcore.fst")
        print(f"waveform: {build / 'izcore.fst'}")


def serve_config(model, lanes, vk, att_tok, wbits, hbits=None, sh=1):
    from hw.ref import load
    m, meta = load(wbits, model, hbits=hbits)
    name = f"serve_{model}_{bits_tag(wbits, hbits)}_l{lanes}_v{vk}_a{att_tok}{f'_h{sh}' if sh > 1 else ''}"
    out = HW / "build" / name
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "image.json").exists():
        write_image(m, lanes, out)
    return out, core_params(m, lanes, vk, att_tok, sh), meta, m


def serve(a):
    out, params, _, _ = serve_config(a.model, a.lanes, a.vk, a.att_tok, a.wbits, a.hbits, a.sh)
    req = Path(a.req).resolve()
    req.mkdir(parents=True, exist_ok=True)
    runner = get_runner("verilator")
    runner.build(sources=SOURCES, hdl_toplevel="izcore", build_dir=out / "sim", parameters=params,
                 build_args=BUILD_ARGS + ["--threads", str(a.threads)] if a.threads > 1 else BUILD_ARGS)
    runner.test(hdl_toplevel="izcore", test_module="serve_core", test_dir=HW / "tests", build_dir=out / "sim",
                plusargs=[f"+wmem={out / 'w'}", f"+hmem={out / 'h'}", f"+emb={out / 'e.hex'}", f"+smem={out / 's'}"],
                extra_env=ENV | {"IMAGE": str(out), "REQ": str(req)})


def bench(a):
    s = MODELS[a.model]
    blocks = range(1024 // a.att_tok)
    ctx = [int(c) for c in a.ctx.split(",")] if a.ctx else [p for b in blocks for p in (b * a.att_tok, (b + 1) * a.att_tok - 1)]
    img = compile_shape(s, a.lanes)
    if isinstance(s, QShape):
        from hw.ref import head_scale
        n_att = s.kinds.count("att")
        params = {"LANES": a.lanes, "VK": a.vk, "ATT_TOK": a.att_tok, "HD": s.head_dim, "RD": s.rope_dim, "NKV": s.n_kv_heads,
                  "REP": s.n_heads // s.n_kv_heads, "NLAYERS": max(1, n_att), "MAXSEQ": pow2(max(ctx) + 1), "VDEPTH": pow2(img.vmem["end"]),
                  "IDEPTH": pow2(len(img.program)), "EDEPTH": 64, "WDEPTH": 64, "HDEPTH": 64,
                  "XDEPTH": pow2(max(s.d_ff, s.d_model, s.lv_heads * s.lv_dim, s.n_heads * s.head_dim) // 32),
                  "YDEPTH": pow2(max(s.vocab, 2 * s.d_ff, max(qwen_in_rows(s, k) for k in set(s.kinds)))), "WBITS": 4, "HBITS": 8,
                  "SDEPTH": pow2(img.scale_depth), "GAINS": 1, "GDEPTH": pow2(max(1, img.gain_len // (32 * a.vk))), "EPS": 4295, "LIN": 1,
                  "HK": s.lk_heads, "HV": s.lv_heads, "DK": s.lk_dim, "DV": s.lv_dim, "CK": s.conv, "NLIN": s.n_layers - n_att, "SH": a.sh,
                  "LS": head_scale(s.lk_dim)}
    else:
        params = {"LANES": a.lanes, "VK": a.vk, "ATT_TOK": a.att_tok, "HD": s.head_dim, "NKV": s.n_kv_heads,
              "NLAYERS": s.n_layers, "MAXSEQ": pow2(max(ctx) + 1), "REP": s.n_heads // s.n_kv_heads,
              "VDEPTH": pow2(vmem_map(s)["end"]), "IDEPTH": pow2(len(img.program)), "EDEPTH": 64, "WDEPTH": 64, "HDEPTH": 64,
              "XDEPTH": pow2(max(s.d_ff, s.d_model) // 32), "YDEPTH": pow2(max(s.vocab, 2 * s.d_ff, (s.n_heads + 2 * s.n_kv_heads) * s.head_dim)),
              "ACT": int(s.subnorm), "SUBNORM": int(s.subnorm), "WBITS": 2 if s.wide else 4, "HBITS": 8 if s.wide else 4, "SDEPTH": pow2(img.scale_depth)} | gain_params(img.gain_len, a.vk)
    build = HW / "build" / f"bench_{a.model}_l{a.lanes}_v{a.vk}_a{a.att_tok}_s{params['MAXSEQ']}{f'_h{a.sh}' if isinstance(s, QShape) else ''}"
    out = build / "bench.json"
    runner = get_runner("verilator")
    runner.build(sources=SOURCES, hdl_toplevel="izcore", build_dir=build, build_args=BUILD_ARGS, parameters=params)
    runner.test(hdl_toplevel="izcore", test_module="bench_core", test_dir=HW / "tests", build_dir=build,
                extra_env=ENV | {"SHAPE": json.dumps(s.__dict__ | {"_q": isinstance(s, QShape)}), "LANES": str(a.lanes), "CTX": ",".join(map(str, ctx)), "OUT": str(out)})
    rows = json.loads(out.read_text())
    print(f"{'model':14s} {'lanes':>5s} {'vk':>3s} {'att':>4s} {'ctx':>5s} {'cycles':>8s} {'tokens/second at 1 GHz':>23s}  per-op cycles")
    for r in rows:
        ops = " ".join(f"{k}={v}" for k, v in r["per_op"].items() if v)
        print(f"{a.model:14s} {a.lanes:5d} {a.vk:3d} {a.att_tok:4d} {r['ctx']:5d} {r['cycles']:8d} "
              f"{1e9 / r['cycles']:23.0f}  {ops} FETCH={r['cycles'] - sum(r['per_op'].values())}")


def emu(a):
    from hw.compiler import write_emu
    from hw.ref import load
    out = HW / "build" / "emu"
    out.mkdir(parents=True, exist_ok=True)
    exe, src = out / "emu", HW / "emu.c"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["clang", "-O3", "-mcpu=apple-m1", "-o", str(exe), str(src)], check=True)
    from hw.ref import chip_image
    real = a.model.startswith("qwen")
    img = chip_image(a.model, a.wbits, a.hbits or 8, a.max_seq) if real else out / f"{a.model}_{bits_tag(a.wbits, a.hbits)}_s{a.max_seq}.bin"
    if not img.exists() or (not real and img.stat().st_mtime < max((HW / f).stat().st_mtime for f in ("compiler.py", "ref.py"))):
        m, _ = load(a.wbits, a.model, a.max_seq, a.hbits)
        write_emu(m, img.with_suffix(".part"))
        img.with_suffix(".part").replace(img)
    print(img)
    return exe, img


def check(a):
    from tokenizers import Tokenizer
    from hw.ref import HF_MODELS, chat_text
    exe, img = emu(argparse.Namespace(model=a.model, wbits=a.wbits, hbits=a.hbits, max_seq=1024))
    tok = Tokenizer.from_file(str(HF_MODELS[a.model] / "tokenizer.json"))
    ids = tok.encode(chat_text(a.model, a.prompt), add_special_tokens=False).ids
    p = subprocess.run([str(exe), str(img)], input=f"x {a.tokens} {len(ids)} 0 {' '.join(map(str, ids))}\n",
                       capture_output=True, text=True, check=True).stdout.split("\n")
    ex = [list(map(int, f.split()[2:])) for f in p if f.startswith("x ")]
    et = [int(f.split()[1]) for f in p if f.startswith("t ")]
    req = HW / "build" / f"req_check_{a.model}"
    shutil.rmtree(req, ignore_errors=True)
    req.mkdir(parents=True)
    sim = subprocess.Popen([sys.executable, __file__, "serve", a.model, "--wbits", str(a.wbits), "--hbits", str(a.hbits or a.wbits), "--lanes", str(a.lanes),
                            "--vk", str(a.vk), "--att-tok", str(a.att_tok), "--threads", str(a.threads), "--sh", str(a.sh), "--req", str(req)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while not (req / "ready").exists():
        if sim.poll() is not None:
            raise SystemExit("RTL simulation stopped")
        time.sleep(1)
    (req / "req.tmp").write_text(json.dumps({"tokens": ids, "n": a.tokens, "stop": [], "max_seq": 512, "dump": len(ex[0])}))
    (req / "req.tmp").replace(req / "req.json")
    while not (req / "resp.json").exists():
        time.sleep(1)
    r = json.loads((req / "resp.json").read_text())
    (req / "req.json").write_text(json.dumps({"quit": True}))
    sim.wait()
    same = sum(s["x"] == x for s, x in zip(r["steps"], ex))
    print(f"residual stream bit-exact at {same} of {len(ex)} positions, generated tokens "
          f"{'equal' if r['tokens'][len(ids):] == et else 'different'}: {tok.decode(et)!r}")
    out = HW / "build" / "exp" / f"check_{a.model}_{bits_tag(a.wbits, a.hbits)}_l{a.lanes}.json"
    out.write_text(json.dumps({"answer": tok.decode(et), "positions": len(ex), "positions_bit_exact": same, "tokens_equal": r["tokens"][len(ids):] == et,
                               "cycles_per_token": r["steps"][0]["cycles"], "lanes": a.lanes, "vk": a.vk, "att_tok": a.att_tok,
                               "date": time.strftime("%Y-%m-%d")}, indent=1))


if __name__ == "__main__":
    os.environ["PATH"] = f"{ROOT / '.venv' / 'bin'}:{os.environ.get('PATH', '')}"
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test")
    t.add_argument("--wbits", type=int, choices=[2, 4, 8], default=4)
    t.add_argument("--hbits", type=int, choices=[2, 4, 8])
    t.add_argument("--lora", type=int, default=0)
    t.add_argument("--lanes", type=int, default=8)
    t.add_argument("--vk", type=int, default=1)
    t.add_argument("--waves", action="store_true")
    t.add_argument("--model", choices=["tiny", "tiny-bitnet", "tc1", "tiny-qwen", "tiny-qwen-w"], default="tiny")
    t.add_argument("--sh", type=int, default=1)
    q = sub.add_parser("serve")
    q.add_argument("model", choices=["tiny", "tiny-bitnet", "tc1", "tiny-qwen", "tiny-qwen-w", "qwen3.5-4b"])
    q.add_argument("--lanes", type=int, default=8)
    q.add_argument("--vk", type=int, default=4)
    q.add_argument("--att-tok", type=int, default=4)
    q.add_argument("--wbits", type=int, choices=[2, 4, 8], default=4)
    q.add_argument("--hbits", type=int, choices=[2, 4, 8])
    q.add_argument("--threads", type=int, default=1)
    q.add_argument("--sh", type=int, default=1)
    q.add_argument("--req", default=str(HW / "build" / "req"))
    b = sub.add_parser("bench")
    b.add_argument("model", choices=MODELS)
    b.add_argument("--lanes", type=int, default=1024)
    b.add_argument("--vk", type=int, default=4)
    b.add_argument("--att-tok", type=int, default=4)
    b.add_argument("--sh", type=int, default=8)
    b.add_argument("--ctx")
    e = sub.add_parser("emu")
    e.add_argument("model", choices=["tiny", "tiny-bitnet", "tc1", "tiny-qwen", "tiny-qwen-w", "qwen3.5-4b"])
    e.add_argument("--wbits", type=int, choices=[2, 4, 8], default=4)
    e.add_argument("--hbits", type=int, choices=[2, 4, 8])
    e.add_argument("--max-seq", type=int, default=1024)
    c = sub.add_parser("check")
    c.add_argument("--model", choices=["tc1", "qwen3.5-4b"], default="qwen3.5-4b")
    c.add_argument("--prompt", default="What is the capital of France?")
    c.add_argument("--tokens", type=int, default=6)
    c.add_argument("--wbits", type=int, choices=[2, 4, 8], default=4)
    c.add_argument("--hbits", type=int, choices=[4, 8])
    c.add_argument("--lanes", type=int, default=128)
    c.add_argument("--vk", type=int, default=4)
    c.add_argument("--att-tok", type=int, default=4)
    c.add_argument("--threads", type=int, default=6)
    c.add_argument("--sh", type=int, default=8)
    a = p.parse_args()
    {"test": test, "serve": serve, "bench": bench, "emu": emu, "check": check}[a.cmd](a)

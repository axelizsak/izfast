import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HW = Path(__file__).resolve().parent
ROOT = HW.parent
sys.path.insert(0, str(ROOT))

from hw.compiler import QShape, compile_shape, pow2, qwen_in_rows
from hw.ref import head_scale
from hw.synth import cut_memories

TC2 = QShape(8, 256, 4, 2, 64, 16, 768, 4096, 2, 4, 64, 64, 4, ("lin", "lin", "lin", "att") * 2)
BOARDS = {"AMD Kria KV260 (K26)": (117_120, 1_248, 144 * 36 + 64 * 288), "AMD ZCU104 (ZU7EV)": (230_400, 1_728, 312 * 36 + 96 * 288),
          "AMD VCU118 (VU9P)": (1_182_240, 6_840, 2_160 * 36 + 960 * 288), "AWS EC2 F2 (VU47P)": (1_304_000, 9_024, 2_016 * 36 + 960 * 288)}


def params(s, lanes, vk, att_tok, sh, max_seq):
    img = compile_shape(s, lanes)
    n_att = s.kinds.count("att")
    return {"LANES": lanes, "VK": vk, "ATT_TOK": att_tok, "HD": s.head_dim, "RD": s.rope_dim, "NKV": s.n_kv_heads, "REP": s.n_heads // s.n_kv_heads,
            "NLAYERS": max(1, n_att), "MAXSEQ": max_seq, "VDEPTH": pow2(img.vmem["end"]), "IDEPTH": pow2(len(img.program)),
            "EDEPTH": pow2(s.vocab * s.d_model // 32), "WDEPTH": pow2(img.words_per_lane), "HDEPTH": pow2(max(1, img.wide_words)),
            "XDEPTH": pow2(max(s.d_ff, s.d_model, s.lv_heads * s.lv_dim, s.n_heads * s.head_dim) // 32),
            "YDEPTH": pow2(max(s.vocab, 2 * s.d_ff, max(qwen_in_rows(s, k) for k in set(s.kinds)))), "WBITS": 4, "HBITS": 8,
            "SDEPTH": pow2(img.scale_depth), "GAINS": 1, "GDEPTH": pow2(max(1, img.gain_len // (32 * vk))), "EPS": 4295, "LIN": 1,
            "HK": s.lk_heads, "HV": s.lv_heads, "DK": s.lk_dim, "DV": s.lv_dim, "CK": s.conv, "NLIN": s.n_layers - n_att, "SH": sh,
            "LS": head_scale(s.lk_dim)}


def memory_mbit(s, max_seq):
    C = 2 * s.lk_heads * s.lk_dim + s.lv_heads * s.lv_dim
    lin = (C + s.lv_heads * s.lv_dim + 2 * s.lv_heads) * s.d_model + s.d_model * s.lv_heads * s.lv_dim + 3 * s.d_ff * s.d_model
    att = (2 * s.n_heads + 2 * s.n_kv_heads) * s.head_dim * s.d_model + s.d_model * s.n_heads * s.head_dim + 3 * s.d_ff * s.d_model
    n_att = s.kinds.count("att")
    weights = ((s.n_layers - n_att) * lin + n_att * att) * 4.5 + s.vocab * s.d_model * 8.5
    state = (s.n_layers - n_att) * (s.lv_heads * s.lk_dim * s.lv_dim * 32 + (s.conv - 1) * C * 48)
    kv = n_att * max_seq * s.n_kv_heads * s.head_dim * 2 * 8
    return {"weights": weights / 1e6, "embedding": s.vocab * s.d_model * 8.5 / 1e6, "state": state / 1e6, "kv": kv / 1e6}


def run(lanes, vk, att_tok, sh, max_seq):
    s = TC2
    out = HW / "build" / "fpga" / f"tc2_l{lanes}_v{vk}_a{att_tok}_h{sh}"
    out.mkdir(parents=True, exist_ok=True)
    rtl = sorted(str(p) for p in (HW / "rtl").glob("*.sv"))
    subprocess.run(["sv2v", "-DSYNTHESIS", "--top=izcore", f"--write={out / 'design.v'}", *rtl], check=True)
    chp = " ".join(f"-chparam {k} {v}" for k, v in params(s, lanes, vk, att_tok, sh, max_seq).items())
    (out / "pre.ys").write_text(f"read_verilog -sv {out / 'design.v'}\nhierarchy -top izcore {chp}\nproc\nflatten\nopt_clean\n"
                                f"memory -nomap\nopt_clean\nwrite_json {out / 'pre.json'}\n")
    subprocess.run(["yosys", "-q", "-s", str(out / "pre.ys")], check=True)
    cut_memories(out / "pre.json", out / "cut.json", "izcore")
    (out / "syn.ys").write_text(f"read_json {out / 'cut.json'}\nsynth_xilinx -family xcup -top izcore -noiopad\ntee -o {out / 'stat.txt'} stat\n")
    subprocess.run(["yosys", "-q", "-s", str(out / "syn.ys")], check=True)
    stat = (out / "stat.txt").read_text()
    count = lambda pat: sum(int(n) for n, c in re.findall(r"^\s+(\d+)\s+(\S+)$", stat, re.M) if re.fullmatch(pat, c))
    mem = memory_mbit(s, max_seq)
    res = {"luts": count(r"LUT\d"), "ffs": count(r"FD\w+"), "dsps": count(r"DSP48E2"), "mbit": sum(mem.values()), "memory": mem,
           "config": {"lanes": lanes, "vk": vk, "att_tok": att_tok, "heads_parallel": sh, "max_seq": max_seq, "model": "TC2 shape"}}
    res["fit"] = {b: {"luts": res["luts"] / bl, "dsps": res["dsps"] / bd, "memory": res["mbit"] * 1e3 / bk} for b, (bl, bd, bk) in BOARDS.items()}
    (HW / "build" / "exp").mkdir(parents=True, exist_ok=True)
    (HW / "build" / "exp" / f"fpga_{out.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lanes", type=int, default=16)
    p.add_argument("--vk", type=int, default=1)
    p.add_argument("--att-tok", type=int, default=4)
    p.add_argument("--sh", type=int, default=1)
    p.add_argument("--max-seq", type=int, default=256)
    a = p.parse_args()
    run(a.lanes, a.vk, a.att_tok, a.sh, a.max_seq)

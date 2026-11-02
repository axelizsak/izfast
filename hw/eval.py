import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parent
sys.path.insert(0, str(HW.parent))

from hw.ref import HF_MODELS, chip_image, qwen_weights


def text_sample():
    from pydoc_data.topics import topics
    keys = ["execmodel", "assignment", "customization", "attribute-access", "debugger"]
    return "\n\n".join(topics[k] for k in keys)


def qwen_hidden(w, c, ids, trace=None):
    eps, T = c["norm_eps"], len(ids)
    rms = lambda x, g: x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g
    silu = lambda x: x / (1 + np.exp(-x))
    H, KV, hd, rd = c["n_heads"], c["n_kv_heads"], c["head_dim"], c["rope_dim"]
    Hk, Hv, Dk, Dv, K = c["lk_heads"], c["lv_heads"], c["lk_dim"], c["lv_dim"], c["conv"]
    ang = np.arange(T)[:, None] * (1.0 / c["rope_theta"] ** (np.arange(0, rd, 2) / rd))[None, :]
    cos, sin = np.cos(ang)[:, None], np.sin(ang)[:, None]

    def rope(v):
        a, b = v[..., 0:rd:2], v[..., 1:rd:2]
        out = v.copy()
        out[..., 0:rd:2], out[..., 1:rd:2] = a * cos - b * sin, a * sin + b * cos
        return out

    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
    x = w["tok_emb"][ids]
    for l, kind in enumerate(c["layer_types"]):
        p = f"l{l}."
        h = rms(x, 1 + w[p + "attn_norm"])
        if kind == "linear_attention":
            qkv = h @ w[p + "in_qkv"].T
            cw = w[p + "conv"]
            pad = np.concatenate([np.zeros((K - 1, qkv.shape[1]), np.float32), qkv])
            mixed = silu(sum(pad[j:j + T] * cw[:, j] for j in range(K)))
            qq, kk, vv = np.split(mixed, [Hk * Dk, 2 * Hk * Dk], -1)
            qq, kk = (np.repeat(v.reshape(T, Hk, Dk), Hv // Hk, 1) for v in (qq, kk))
            qq = qq / np.sqrt((qq * qq).sum(-1, keepdims=True) + 1e-6) / np.sqrt(Dk)
            kk = kk / np.sqrt((kk * kk).sum(-1, keepdims=True) + 1e-6)
            vv = vv.reshape(T, Hv, Dv)
            z = (h @ w[p + "in_z"].T).reshape(T, Hv, Dv)
            beta = 1 / (1 + np.exp(-(h @ w[p + "in_b"].T)))
            g = -np.exp(w[p + "A_log"]) * np.logaddexp(0, h @ w[p + "in_a"].T + w[p + "dt_bias"])
            S, o = np.zeros((Hv, Dk, Dv), np.float32), np.empty((T, Hv, Dv), np.float32)
            for i in range(T):
                S *= np.exp(g[i])[:, None, None]
                delta = (vv[i] - np.einsum("hkv,hk->hv", S, kk[i])) * beta[i][:, None]
                S += kk[i][:, :, None] * delta[:, None, :]
                o[i] = np.einsum("hkv,hk->hv", S, qq[i])
            o = rms(o, w[p + "lnorm"]) * silu(z)
            x = x + o.reshape(T, Hv * Dv) @ w[p + "out"].T
        else:
            qq = (h @ w[p + "wq"].T).reshape(T, H, hd)
            gate = (h @ w[p + "wg"].T).reshape(T, H, hd)
            kk = (h @ w[p + "wk"].T).reshape(T, KV, hd)
            vv = (h @ w[p + "wv"].T).reshape(T, KV, hd)
            qq, kk = rope(rms(qq, 1 + w[p + "q_norm"])), rope(rms(kk, 1 + w[p + "k_norm"]))
            kk, vv = np.repeat(kk, H // KV, 1), np.repeat(vv, H // KV, 1)
            sc = np.einsum("thd,shd->hts", qq, kk) / np.sqrt(hd) + mask
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            o = np.einsum("hts,shd->thd", sc / sc.sum(-1, keepdims=True), vv) / (1 + np.exp(-gate))
            x = x + o.reshape(T, H * hd) @ w[p + "wo"].T
        h = rms(x, 1 + w[p + "mlp_norm"])
        x = x + (silu(h @ w[p + "w_gate"].T) * (h @ w[p + "w_up"].T)) @ w[p + "w_down"].T
        if trace is not None:
            trace.append(x.copy())
    return rms(x, 1 + w["final_norm"])


def head_logprobs(hid, emb, ids, chunk=64):
    lp, am = [], []
    for i in range(0, len(ids) - 1, chunk):
        z = hid[i:min(i + chunk, len(ids) - 1)] @ emb.T
        m = z.max(-1, keepdims=True)
        lz = z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))
        lp.append(lz[np.arange(len(z)), ids[i + 1:i + 1 + len(z)]])
        am.append(z.argmax(-1))
    return np.concatenate(lp), np.concatenate(am)


def chip_logprobs(model, spec, chunks):
    w, _, h = spec.partition("h")
    subprocess.run([sys.executable, str(HW / "run.py"), "emu", model, "--wbits", w] + (["--hbits", h] if h else []),
                   check=True, capture_output=True)
    exe, img = HW / "build" / "emu" / "emu", chip_image(model, int(w), int(h or w))
    req = "".join(f"l 1 {len(ids)} 0 {' '.join(map(str, ids))}\n" for ids in chunks)
    out = subprocess.run([str(exe), str(img)], input=req, capture_output=True, text=True, check=True).stdout.split("\n")
    rows = [f.split() for f in out if f.startswith("l ")]
    return np.array([float(r[2]) for r in rows]), np.array([int(r[3]) for r in rows])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--chunks", type=int, default=4)
    a.add_argument("--length", type=int, default=512)
    a.add_argument("--wbits", default="4h8")
    a.add_argument("--model", choices=["qwen3.5-4b"], default="qwen3.5-4b")
    a = a.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF_MODELS[a.model] / "tokenizer.json"))
    ids = tok.encode(text_sample(), add_special_tokens=False).ids
    chunks = [ids[i * a.length:(i + 1) * a.length] for i in range(a.chunks)]
    ref = HW / "build" / "exp" / f"float_{a.model}_{a.chunks}x{a.length}.npz"
    if not ref.exists():
        w, c, _ = qwen_weights(a.model, a.length)
        emb = w["tok_emb"]
        dict.__setitem__(w, "tok_emb", lambda: emb)
        out = [head_logprobs(qwen_hidden(w, c, np.array(ch)), emb, np.array(ch)) for ch in chunks]
        ref.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ref, lp=np.concatenate([o[0] for o in out]), am=np.concatenate([o[1] for o in out]))
    f = np.load(ref)
    res = {"model": a.model, "tokens": int(f["lp"].size), "reference": {"perplexity": float(np.exp(-f["lp"].mean()))}}
    for spec in a.wbits.split(","):
        lp, am = chip_logprobs(a.model, spec, chunks)
        res[f"chip_w{spec}"] = {"perplexity": float(np.exp(-lp.mean())), "same_next_token": float((am == f["am"]).mean())}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

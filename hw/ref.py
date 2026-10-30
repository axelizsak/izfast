import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GROUP = 32
SBITS = 16
FRAC = 16
VBITS = 48
DMAX = 24
L2E = round(math.log2(math.e) * 2 ** 15)
EPS_FIX = 43
TERNARY_CODES = {0: 0b00, 1: 0b01, -1: 0b11}

RSQRT = np.array([round(2 ** 23 / math.sqrt(i * 2 ** 10 + 2 ** 9)) for i in range(64, 256)], dtype=np.int64)
EXP2 = np.array([round(2 ** 16 * 2 ** (-f / 256)) for f in range(256)], dtype=np.int64)
SIGMOID = np.array([round(2 ** 16 / (1 + math.exp(-((i + 0.5) / 64 - 8)))) for i in range(1024)], dtype=np.int64)


def obj(a):
    return np.asarray(a).astype(object)


def wrap(v, bits=VBITS):
    half = 1 << (bits - 1)
    return (obj(v) + half) % (1 << bits) - half


def rsr(v, n):
    v = obj(v)
    n = np.asarray(n, dtype=object)
    right = (v + (obj(1) << np.maximum(n - 1, 0))) >> np.maximum(n, 0)
    left = v << np.maximum(-n, 0)
    return np.where(n > 0, right, left)


def shfind(m):
    m = int(m)
    sh = max(0, m.bit_length() - 7)
    return sh + 1 if ((m + (1 << (sh - 1))) >> sh if sh else m) > 127 else sh


def quant8(x):
    sh = shfind(np.abs(obj(x)).max())
    return np.clip(rsr(x, sh), -127, 127).astype(np.int64), sh


def quant_mx(x):
    xg = obj(x).reshape(-1, GROUP)
    sh = np.array([shfind(m) for m in np.abs(xg).max(-1)], dtype=np.int64)
    ref = int(max(sh.min(), sh.max() - DMAX))
    sh = np.maximum(sh, ref)
    xq = np.clip(rsr(xg, sh[:, None]), -127, 127).astype(np.int64)
    return xq.reshape(-1), sh - ref, ref


def quantize_matrix(w, wbits=4, chunk=8192):
    R, C = w.shape
    top = 1 if wbits == 2 else (1 << (wbits - 1)) - 1
    q = np.empty((R, C // GROUP, GROUP), np.int8)
    scale = np.empty((R, C // GROUP))
    for r in range(0, R, chunk):
        wg = np.asarray(w[r:r + chunk], dtype=np.float64).reshape(-1, C // GROUP, GROUP)
        sc = np.abs(wg).mean(-1, keepdims=True) if wbits == 2 else np.abs(wg).max(-1, keepdims=True) / top
        sc[sc == 0] = 1.0
        q[r:r + chunk] = np.clip(np.round(wg / sc), -top if wbits == 2 else -top - 1, top)
        scale[r:r + chunk] = sc[..., 0]
    E = int(np.floor(np.log2((2 ** SBITS - 1) / scale.max())))
    s = np.clip(np.round(scale * 2.0 ** E), 1, 2 ** SBITS - 1).astype(np.int32)
    return q, s, E


def matvec_mx(q, s, xq, d, chunk=4096):
    R, G, V = q.shape
    x = np.asarray(xq).astype(np.int32).reshape(G, V)
    y = np.empty(R, np.int64)
    with np.errstate(over="ignore"):
        for r in range(0, R, chunk):
            dots = (q[r:r + chunk].astype(np.int32) * x).sum(-1, dtype=np.int32).astype(np.int64)
            y[r:r + chunk] = ((dots * s[r:r + chunk]) << d[None, :]).sum(-1)
    return y


def rms_factor(x, inv_n):
    ss = int((obj(x) ** 2).sum())
    mean = ((ss * inv_n) >> 32) + EPS_FIX
    e = (mean.bit_length() - 1 - 16) >> 1
    m = mean >> (2 * e) if e >= 0 else mean << (-2 * e)
    return int(RSQRT[(m >> 10) - 64]), e


def inv_count(n):
    return round(2 ** 32 / n)


def head_scale(hd):
    return round(2 ** 15 / math.sqrt(hd))


def sigmoid_fix(g):
    g = obj(g)
    idx = np.clip((g + (8 << FRAC)) >> (FRAC - 6), 0, 1023).astype(np.int64)
    sig = obj(SIGMOID[idx])
    sig = np.where(g >= (8 << FRAC), 1 << FRAC, sig)
    return np.where(g < -(8 << FRAC), 0, sig)


def rope_tables(head_dim, max_seq, theta, scaling=None):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))
    if scaling:
        f, lo, hi = scaling["factor"], scaling["low_freq_factor"], scaling["high_freq_factor"]
        smooth = np.clip((scaling["original_max_position_embeddings"] * inv / (2 * math.pi) - lo) / (hi - lo), 0, 1)
        inv = (1 - smooth) * inv / f + smooth * inv
    ang = np.arange(max_seq)[:, None] * inv[None, :]
    return np.round(np.cos(ang) * 2 ** 14).astype(np.int64), np.round(np.sin(ang) * 2 ** 14).astype(np.int64)


def pack_weight_word(q_group, s, wbits):
    word = 0
    for i, v in enumerate(q_group):
        code = TERNARY_CODES[int(v)] if wbits == 2 else int(v) & ((1 << wbits) - 1)
        word |= code << (i * wbits)
    return word | (int(s) << (GROUP * wbits))


def bank_layout(q, s, lanes, wbits):
    R, G, _ = q.shape
    for r in range(R):
        lane, block = r % lanes, r // lanes
        for g in range(G):
            yield lane, block * G + g, pack_weight_word(q[r, g], s[r, g], wbits)


@dataclass
class QMat:
    q: np.ndarray
    s: np.ndarray
    E: int

    @property
    def rows(self):
        return self.q.shape[0]


class FixedLlama:
    def __init__(self, weights, cfg, wbits=4):
        self.c = cfg
        self.L, self.d = cfg["n_layers"], cfg["d_model"]
        self.H, self.KV, self.hd = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
        self.F = cfg["d_ff"]
        self.inv_d = inv_count(self.d)
        self.hs = head_scale(self.hd)
        qz = lambda w: QMat(*quantize_matrix(w, wbits))
        self.emb = qz(weights["tok_emb"])
        self.layers = []
        for l in range(self.L):
            p = f"l{l}."
            wqkv = np.concatenate([weights[p + "wq"], weights[p + "wk"], weights[p + "wv"]]) * weights[p + "attn_norm"][None, :]
            wgu = np.concatenate([weights[p + "w_gate"], weights[p + "w_up"]]) * weights[p + "mlp_norm"][None, :]
            self.layers.append({"qkv": qz(wqkv), "o": qz(weights[p + "wo"]), "gu": qz(wgu), "down": qz(weights[p + "w_down"])})
        self.head = qz(weights["tok_emb"] * weights["final_norm"][None, :])
        self.cos, self.sin = rope_tables(self.hd, cfg["max_seq"], cfg["rope_theta"], cfg.get("rope_scaling"))

    def embed(self, token):
        m = self.emb
        return wrap(rsr((m.q[token].astype(np.int64) * m.s[token][:, None]).reshape(-1), m.E - FRAC))

    @staticmethod
    def mv(m, x, r=None):
        xq, d, ref = quant_mx(x)
        y = obj(matvec_mx(m.q, m.s, xq, d))
        if r is None:
            return wrap(rsr(y, m.E - ref))
        L, e = r
        return wrap(rsr(y * L, m.E - ref + 7 + e))

    def rope(self, v, pos):
        a, b = v[:, 0::2], v[:, 1::2]
        c, s = obj(self.cos[pos])[None, :], obj(self.sin[pos])[None, :]
        out = np.empty_like(v)
        out[:, 0::2] = wrap(rsr(a * c - b * s, 14))
        out[:, 1::2] = wrap(rsr(a * s + b * c, 14))
        return out

    def attention(self, q, cache, l, pos):
        K, Ksh, V, Vsh = cache["k"][l], cache["ksh"][l], cache["v"][l], cache["vsh"][l]
        n = pos + 1
        rep = self.H // self.KV
        out = np.empty((self.H, self.hd), dtype=object)
        for h in range(self.H):
            g = h // rep
            qq, qsh = quant8(q[h])
            ksh = Ksh[:n, g]
            kmin = int(ksh.min())
            a = obj(K[:n, g] @ qq) << obj(ksh - kmin)
            z = a.max() - a
            u = rsr(z * (L2E * self.hs), 32 + 15 + 15 - 8 - qsh - kmin)
            ip = u >> 8
            p = np.array([0 if i >= 17 else int(EXP2[int(f) & 255]) >> int(i) for i, f in zip(ip, u)], dtype=object)
            P = int(p.sum())
            vsh = Vsh[:n, g]
            vmin = int(vsh.min())
            num = (p << obj(vsh - vmin)) @ obj(V[:n, g])
            out[h] = wrap(rsr(num * ((1 << 32) // P), 32 - vmin))
        return out.reshape(-1)

    def new_cache(self):
        S = self.c["max_seq"]
        return {"k": np.zeros((self.L, S, self.KV, self.hd), np.int64), "ksh": np.zeros((self.L, S, self.KV), np.int64),
                "v": np.zeros((self.L, S, self.KV, self.hd), np.int64), "vsh": np.zeros((self.L, S, self.KV), np.int64)}

    def step(self, token, pos, cache, trace=None):
        H, KV, hd, F = self.H, self.KV, self.hd, self.F
        x = self.embed(token)
        for l, m in enumerate(self.layers):
            qkv = self.mv(m["qkv"], x, rms_factor(x, self.inv_d))
            q = self.rope(qkv[:H * hd].reshape(H, hd), pos)
            k = self.rope(qkv[H * hd:(H + KV) * hd].reshape(KV, hd), pos)
            v = qkv[(H + KV) * hd:].reshape(KV, hd)
            for g in range(KV):
                cache["k"][l, pos, g], cache["ksh"][l, pos, g] = quant8(k[g])
                cache["v"][l, pos, g], cache["vsh"][l, pos, g] = quant8(v[g])
            x = wrap(x + self.mv(m["o"], self.attention(q, cache, l, pos)))
            gu = self.mv(m["gu"], x, rms_factor(x, self.inv_d))
            g_, u = gu[:F], gu[F:]
            silu = rsr(g_ * sigmoid_fix(g_), FRAC)
            x = wrap(x + self.mv(m["down"], wrap(rsr(silu * u, FRAC))))
            if trace is not None:
                trace.append(x.copy())
        xq, d, _ = quant_mx(x)
        return matvec_mx(self.head.q, self.head.s, xq, d)


HF_MODELS = {"llama-3.2-1b": ROOT / "data" / "llama-3.2-1b-instruct"}


class Lazy(dict):
    def __getitem__(self, k):
        return super().__getitem__(k)()


def read_safetensors(path):
    raw = np.memmap(path, np.uint8, "r")
    n = int(raw[:8].view(np.uint64)[0])
    out = {}
    for name, t in json.loads(bytes(raw[8:8 + n])).items():
        if name != "__metadata__":
            assert t["dtype"] == "BF16", t["dtype"]
            a, b = t["data_offsets"]
            out[name] = raw[8 + n + a:8 + n + b].view(np.uint16).reshape(t["shape"])
    return out


def bf16(a):
    return (np.asarray(a).astype(np.uint32) << 16).view(np.float32)


def hf_config(name, max_seq=512):
    hc = json.loads((HF_MODELS[name] / "config.json").read_text())
    H, D = hc["num_attention_heads"], hc["hidden_size"]
    return {"vocab_size": hc["vocab_size"], "d_model": D, "n_layers": hc["num_hidden_layers"], "n_heads": H,
            "n_kv_heads": hc["num_key_value_heads"], "head_dim": hc.get("head_dim", D // H), "d_ff": hc["intermediate_size"],
            "max_seq": max_seq, "rope_theta": float(hc["rope_theta"]), "rope_scaling": hc.get("rope_scaling"),
            "norm_eps": hc["rms_norm_eps"]}


def tiny_model(seed=0):
    cfg = {"vocab_size": 512, "d_model": 256, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2, "head_dim": 64, "d_ff": 768,
           "max_seq": 256, "rope_theta": 10000.0, "rope_scaling": None, "norm_eps": 1e-5}
    r = np.random.default_rng(seed)
    g = lambda *shape: (r.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)
    norm = lambda: (1 + 0.1 * r.standard_normal(cfg["d_model"])).astype(np.float32)
    D, H, KV, hd, F = cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"], cfg["d_ff"]
    w = {"tok_emb": g(cfg["vocab_size"], D), "final_norm": norm()}
    for l in range(cfg["n_layers"]):
        w.update({f"l{l}.wq": g(H * hd, D), f"l{l}.wk": g(KV * hd, D), f"l{l}.wv": g(KV * hd, D), f"l{l}.wo": g(D, H * hd),
                  f"l{l}.attn_norm": norm(), f"l{l}.mlp_norm": norm(), f"l{l}.w_gate": g(F, D), f"l{l}.w_up": g(F, D),
                  f"l{l}.w_down": g(D, F)})
    return w, cfg, {"config": cfg}


def load_weights(name="tiny", max_seq=512):
    if name == "tiny":
        return tiny_model()
    d = HF_MODELS[name]
    cfg = hf_config(name, max_seq)
    H, KV, hd = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    t = read_safetensors(d / "model.safetensors")
    pair = lambda w, h: w.reshape(h, 2, hd // 2, -1).transpose(0, 2, 1, 3).reshape(h * hd, -1)
    hf = lambda k, h=0: (lambda: pair(bf16(t[k]), h)) if h else (lambda: bf16(t[k]))
    w = Lazy({"tok_emb": hf("model.embed_tokens.weight"), "final_norm": hf("model.norm.weight")})
    for l in range(cfg["n_layers"]):
        p = f"model.layers.{l}."
        w[f"l{l}.wq"] = hf(p + "self_attn.q_proj.weight", H)
        w[f"l{l}.wk"] = hf(p + "self_attn.k_proj.weight", KV)
        w[f"l{l}.wv"] = hf(p + "self_attn.v_proj.weight")
        w[f"l{l}.wo"] = hf(p + "self_attn.o_proj.weight")
        w[f"l{l}.attn_norm"] = hf(p + "input_layernorm.weight")
        w[f"l{l}.mlp_norm"] = hf(p + "post_attention_layernorm.weight")
        w[f"l{l}.w_gate"] = hf(p + "mlp.gate_proj.weight")
        w[f"l{l}.w_up"] = hf(p + "mlp.up_proj.weight")
        w[f"l{l}.w_down"] = hf(p + "mlp.down_proj.weight")
    return w, cfg, {"config": cfg, "tokenizer": str(d / "tokenizer.json")}


def load(wbits=4, name="tiny", max_seq=512):
    w, cfg, meta = load_weights(name, max_seq)
    return FixedLlama(w, cfg, wbits), meta

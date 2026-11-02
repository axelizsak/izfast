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
GAIN_FRAC = 15
DMAX = 24
L2E = round(math.log2(math.e) * 2 ** 15)
EPS_FIX = 43
TERNARY_CODES = {0: 0b00, 1: 0b01, -1: 0b11}

RSQRT = np.array([round(2 ** 23 / math.sqrt(i * 2 ** 10 + 2 ** 9)) for i in range(64, 256)], dtype=np.int64)
EXP2 = np.array([round(2 ** 16 * 2 ** (-f / 256)) for f in range(256)], dtype=np.int64)
SIGMOID = np.array([round(2 ** 16 / (1 + math.exp(-((i + 0.5) / 64 - 8)))) for i in range(1024)], dtype=np.int64)
SOFTPLUS = np.array([round(2 ** 16 * math.log1p(math.exp((i + 0.5) / 256 - 8))) for i in range(4096)], dtype=np.int64)
EXP2_HI = np.array([round(2 ** 16 * 2 ** (-i / 256)) for i in range(256)], dtype=np.int64)
EXP2_LO = np.array([round(2 ** 16 * 2 ** (-i / 65536)) for i in range(256)], dtype=np.int64)
SBITS_STATE = 32


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
        aw = np.abs(wg)
        if wbits == 2:
            am = aw.max((1, 2), keepdims=True)
            exact = ((aw == 0) | (aw == am)).all((1, 2), keepdims=True)
            sc = np.broadcast_to(np.where(exact, am, aw.mean((1, 2), keepdims=True)), aw[..., :1].shape).copy()
        else:
            sc = aw.max(-1, keepdims=True) / top
        sc[sc == 0] = 1.0
        if wbits == 4:
            best, err = sc, np.full(sc.shape, np.inf)
            for k in np.linspace(1.0, 0.6, 9):
                e = ((np.clip(np.round(wg / (sc * k)), -top - 1, top) * sc * k - wg) ** 2).sum(-1, keepdims=True)
                best, err = np.where(e < err, sc * k, best), np.minimum(e, err)
            sc = best
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


def rms_factor(x, inv_n, eps=EPS_FIX):
    ss = int((obj(x) ** 2).sum())
    mean = ((ss * inv_n) >> 32) + eps
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


def softplus_fix(x):
    x = obj(x)
    idx = np.clip((x + (8 << FRAC)) >> (FRAC - 8), 0, 4095).astype(np.int64)
    sp = obj(SOFTPLUS[idx])
    sp = np.where(x >= (8 << FRAC), x, sp)
    return np.where(x < -(8 << FRAC), 0, sp)


def silu_fix(x):
    return rsr(obj(x) * sigmoid_fix(x), FRAC)


def exp2_neg(u):
    u = np.asarray(u, np.int64)
    v = (EXP2_HI[(u >> 8) & 255] * EXP2_LO[u & 255] + (1 << 15)) >> 16
    return np.where(u >> 16 >= 17, 0, v >> np.minimum(u >> 16, 63))


def norm_apply(x, inv_n, eps):
    L, e = rms_factor(x, inv_n, eps)
    return wrap(rsr(obj(x) * L, 7 + e))


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


def pack_trits(q_group):
    t = np.concatenate([np.asarray(q_group, np.int64) + 1, np.ones(3, np.int64)]).reshape(7, 5)
    return int(sum(int((t[j] * 3 ** np.arange(5)).sum()) << (8 * j) for j in range(7)))


def bank_layout(q, s, lanes, wbits):
    R, G, _ = q.shape
    for r in range(R):
        lane, block = r % lanes, r // lanes
        for g in range(G):
            yield lane, block * G + g, pack_trits(q[r, g]) if wbits == 2 else pack_weight_word(q[r, g], s[r, g], wbits)


def row_scales(s, lanes):
    for r in range(s.shape[0]):
        yield r % lanes, r // lanes, int(s[r, 0])


@dataclass
class QMat:
    q: np.ndarray
    s: np.ndarray
    E: int

    @property
    def rows(self):
        return self.q.shape[0]


def gain_vector(g):
    g = np.asarray(g, np.float64)
    gf = int(np.floor(np.log2(32767 / max(np.abs(g).max(), 1e-30))))
    return np.clip(np.round(g * 2.0 ** gf), -32767, 32767).astype(np.int64), gf


class FixedLlama:
    def __init__(self, weights, cfg, wbits=4, hbits=None, lora=0):
        self.c = cfg
        self.rank = lora
        self.wbits, self.hbits = wbits, hbits or wbits
        self.gains = wbits == 2
        self.L, self.d = cfg["n_layers"], cfg["d_model"]
        self.H, self.KV, self.hd = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
        self.F = cfg["d_ff"]
        self.inv_d, self.inv_a, self.inv_f = inv_count(self.d), inv_count(self.H * self.hd), inv_count(self.F)
        self.relu2, self.subnorm = cfg.get("act") == "relu2", bool(cfg.get("subnorm"))
        self.hs = head_scale(self.hd)
        qz = lambda w, b=wbits: QMat(*quantize_matrix(w, b))
        self.emb = qz(weights["tok_emb"], self.hbits)
        self.layers, self.gain = [], []
        norms = {"qkv": "attn_norm", "gu": "mlp_norm"} | ({"o": "attn_sub_norm", "down": "ffn_sub_norm"} if self.subnorm else {})
        for l in range(self.L):
            p = f"l{l}."
            w = {"qkv": np.concatenate([weights[p + "wq"], weights[p + "wk"], weights[p + "wv"]]),
                 "gu": np.concatenate([weights[p + "w_gate"], weights[p + "w_up"]]), "o": weights[p + "wo"], "down": weights[p + "w_down"]}
            g = {}
            for k, n in norms.items():
                if self.gains:
                    g[k] = gain_vector(weights[p + n])
                else:
                    w[k] = w[k] * weights[p + n][None, :]
            layer = {k: qz(v) for k, v in w.items()}
            for k, (_, gf) in g.items():
                layer[k].E += gf - GAIN_FRAC
            self.layers.append(layer)
            self.gain.append({k: v for k, (v, _) in g.items()})
        self.head = qz(weights["tok_emb"] * weights["final_norm"][None, :], self.hbits)
        self.cos, self.sin = rope_tables(self.hd, cfg["max_seq"], cfg["rope_theta"], cfg.get("rope_scaling"))
        self.lora = [self.adapters(l) for l in range(self.L)] if lora else []

    def adapters(self, l):
        r = np.random.default_rng(1000 + l)
        out = {}
        for k, n_in, n_out in (("o", self.H * self.hd, self.d), ("down", self.F, self.d)):
            a = (r.standard_normal((self.rank, n_in)) / np.sqrt(n_in)).astype(np.float32)
            b = (r.standard_normal((n_out, self.rank)) * 0.5 / np.sqrt(self.rank)).astype(np.float32)
            out[k] = (QMat(*quantize_matrix(a, self.hbits)), QMat(*quantize_matrix(b, self.hbits)))
        return out

    def residual(self, x, l, k, m, inp, r):
        y = self.mv(m[k], inp, r)
        if not self.lora:
            return wrap(x + y)
        a, b = self.lora[l][k]
        t = self.mv(a, inp, r)
        return wrap(wrap(x + y) + self.mv(b, t))

    def embed(self, token):
        m = self.emb
        return wrap(rsr((m.q[token].astype(np.int64) * m.s[token][:, None]).reshape(-1), m.E - FRAC))

    @staticmethod
    def scale(x, g):
        return x if g is None else wrap(rsr(obj(x) * obj(g), GAIN_FRAC))

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
            gn = self.gain[l]
            qkv = self.mv(m["qkv"], self.scale(x, gn.get("qkv")), rms_factor(x, self.inv_d))
            q = self.rope(qkv[:H * hd].reshape(H, hd), pos)
            k = self.rope(qkv[H * hd:(H + KV) * hd].reshape(KV, hd), pos)
            v = qkv[(H + KV) * hd:].reshape(KV, hd)
            for g in range(KV):
                cache["k"][l, pos, g], cache["ksh"][l, pos, g] = quant8(k[g])
                cache["v"][l, pos, g], cache["vsh"][l, pos, g] = quant8(v[g])
            a = self.attention(q, cache, l, pos)
            x = self.residual(x, l, "o", m, self.scale(a, gn.get("o")), rms_factor(a, self.inv_a) if self.subnorm else None)
            gu = self.mv(m["gu"], self.scale(x, gn.get("gu")), rms_factor(x, self.inv_d))
            g_, u = gu[:F], gu[F:]
            act = rsr(np.maximum(g_, 0) ** 2, FRAC) if self.relu2 else rsr(g_ * sigmoid_fix(g_), FRAC)
            h = wrap(rsr(act * u, FRAC))
            x = self.residual(x, l, "down", m, self.scale(h, gn.get("down")), rms_factor(h, self.inv_f) if self.subnorm else None)
            if trace is not None:
                trace.append(x.copy())
        xq, d, _ = quant_mx(x)
        return matvec_mx(self.head.q, self.head.s, xq, d)


class FixedQwen(FixedLlama):
    def __init__(self, weights, cfg, wbits=4, hbits=8):
        self.c = cfg
        self.rank, self.lora, self.gains, self.relu2, self.subnorm = 0, [], False, False, False
        self.wbits, self.hbits = wbits, hbits
        self.L, self.d, self.F = cfg["n_layers"], cfg["d_model"], cfg["d_ff"]
        self.H, self.KV, self.hd, self.rd = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"], cfg["rope_dim"]
        self.Hk, self.Hv, self.Dk, self.Dv, self.K = cfg["lk_heads"], cfg["lv_heads"], cfg["lk_dim"], cfg["lv_dim"], cfg["conv"]
        self.kinds = ["lin" if t == "linear_attention" else "att" for t in cfg["layer_types"]]
        self.eps = max(1, round(cfg["norm_eps"] * 2 ** 32))
        self.inv_d, self.inv_f = inv_count(self.d), inv_count(self.F)
        self.hs, self.ls = head_scale(self.hd), head_scale(self.Dk)
        qz = lambda w, b=wbits: QMat(*quantize_matrix(w, b))
        self.emb = qz(weights["tok_emb"], hbits)
        self.layers = []
        for l, kind in enumerate(self.kinds):
            p = f"l{l}."
            n_in, n_mlp = 1 + weights[p + "attn_norm"], 1 + weights[p + "mlp_norm"]
            m = {"gu": qz(np.concatenate([weights[p + "w_gate"], weights[p + "w_up"]]) * n_mlp[None, :]), "down": qz(weights[p + "w_down"])}
            if kind == "lin":
                m["in"] = qz(np.concatenate([weights[p + "in_qkv"], weights[p + "in_z"], weights[p + "in_b"], weights[p + "in_a"]]) * n_in[None, :])
                m["o"] = qz(weights[p + "out"])
                m["conv"] = gain_vector(weights[p + "conv"])
                m["dt"] = np.round(np.asarray(weights[p + "dt_bias"], np.float64) * 2 ** FRAC).astype(np.int64)
                m["A"] = gain_vector(np.exp(np.asarray(weights[p + "A_log"], np.float64)))
                m["norm"] = gain_vector(weights[p + "lnorm"])
            else:
                m["in"] = qz(np.concatenate([weights[p + "wq"], weights[p + "wg"], weights[p + "wk"], weights[p + "wv"]]) * n_in[None, :])
                m["o"] = qz(weights[p + "wo"])
                m["qn"], m["kn"] = gain_vector(1 + weights[p + "q_norm"]), gain_vector(1 + weights[p + "k_norm"])
            self.layers.append(m)
        self.head = qz(weights["tok_emb"] * (1 + weights["final_norm"])[None, :], hbits)
        self.cos, self.sin = rope_tables(self.rd, cfg["max_seq"], cfg["rope_theta"])
        self.n_att = self.kinds.count("att")

    def new_cache(self):
        S, n, n_lin = self.c["max_seq"], self.n_att, self.L - self.n_att
        return {"k": np.zeros((n, S, self.KV, self.hd), np.int64), "ksh": np.zeros((n, S, self.KV), np.int64),
                "v": np.zeros((n, S, self.KV, self.hd), np.int64), "vsh": np.zeros((n, S, self.KV), np.int64),
                "S": np.zeros((n_lin, self.Hv, self.Dk, self.Dv), np.int64),
                "conv": np.zeros((n_lin, self.K - 1, 2 * self.Hk * self.Dk + self.Hv * self.Dv), dtype=object)}

    @staticmethod
    def gain(x, g):
        v, gf = g
        return wrap(rsr(obj(x) * obj(v), gf))

    def rope(self, v, pos):
        out = v.copy()
        out[:, :self.rd] = FixedLlama.rope(self, v[:, :self.rd], pos)
        return out

    def delta(self, m, y, c, i):
        Hk, Hv, Dk, Dv, K = self.Hk, self.Hv, self.Dk, self.Dv, self.K
        C = 2 * Hk * Dk + Hv * Dv
        u, z, b, a = y[:C], y[C:C + Hv * Dv], y[C + Hv * Dv:C + Hv * Dv + Hv], y[C + Hv * Dv + Hv:]
        win = np.concatenate([c["conv"][i], u[None, :]])
        c["conv"][i] = win[1:]
        cw, cf = m["conv"]
        mixed = silu_fix(wrap(rsr((win.T * obj(cw)).sum(1), cf)))
        q, k, v = mixed[:Hk * Dk].reshape(Hk, Dk), mixed[Hk * Dk:2 * Hk * Dk].reshape(Hk, Dk), mixed[2 * Hk * Dk:].reshape(Hv, Dv)
        one = inv_count(1)
        qn = np.array([rsr(norm_apply(q[h], one, self.eps) * self.ls, 15) for h in range(Hk)], dtype=np.int64)
        kn = np.array([norm_apply(k[h], one, self.eps) for h in range(Hk)], dtype=np.int64)
        beta = np.asarray(sigmoid_fix(b), np.int64)
        Av, Af = m["A"]
        sp = softplus_fix(obj(a) + obj(m["dt"]))
        dec = exp2_neg(np.asarray(rsr(sp * obj(Av) * L2E, Af + 15), np.int64))
        S, top = c["S"][i], (1 << (SBITS_STATE - 1)) - 1
        o = np.empty((Hv, Dv), dtype=object)
        rep = Hv // Hk
        r16 = lambda t: (t + (1 << 15)) >> 16
        for h in range(Hv):
            kk, qq = kn[h // rep], qn[h // rep]
            s = np.clip(r16(S[h] * dec[h]), -top, top)
            dl = r16((np.asarray(v[h], np.int64) - r16(kk @ s)) * beta[h])
            s = np.clip(s + r16(np.outer(kk, dl)), -top, top)
            S[h] = s
            oh = norm_apply(r16(qq @ s), inv_count(Dv), self.eps)
            o[h] = wrap(rsr(self.gain(oh, m["norm"]) * silu_fix(z[h * Dv:(h + 1) * Dv]), FRAC))
        return o.reshape(-1)

    def gated_attention(self, m, y, c, i, pos):
        H, KV, hd = self.H, self.KV, self.hd
        q, gate = y[:H * hd].reshape(H, hd), y[H * hd:2 * H * hd]
        k, v = y[2 * H * hd:(2 * H + KV) * hd].reshape(KV, hd), y[(2 * H + KV) * hd:].reshape(KV, hd)
        inv = inv_count(hd)
        q = self.rope(np.array([self.gain(norm_apply(q[h], inv, self.eps), m["qn"]) for h in range(H)], dtype=object), pos)
        k = self.rope(np.array([self.gain(norm_apply(k[g], inv, self.eps), m["kn"]) for g in range(KV)], dtype=object), pos)
        for g in range(KV):
            c["k"][i, pos, g], c["ksh"][i, pos, g] = quant8(k[g])
            c["v"][i, pos, g], c["vsh"][i, pos, g] = quant8(v[g])
        return wrap(rsr(obj(self.attention(q, c, i, pos)) * sigmoid_fix(gate), FRAC))

    def step(self, token, pos, cache, trace=None):
        x = self.embed(token)
        il = ia = 0
        for m, kind in zip(self.layers, self.kinds):
            y = self.mv(m["in"], x, rms_factor(x, self.inv_d, self.eps))
            if kind == "lin":
                a = self.delta(m, y, cache, il)
                il += 1
            else:
                a = self.gated_attention(m, y, cache, ia, pos)
                ia += 1
            x = wrap(x + self.mv(m["o"], a))
            gu = self.mv(m["gu"], x, rms_factor(x, self.inv_d, self.eps))
            h = wrap(rsr(silu_fix(gu[:self.F]) * gu[self.F:], FRAC))
            x = wrap(x + self.mv(m["down"], h))
            if trace is not None:
                trace.append(x.copy())
        xq, d, _ = quant_mx(x)
        return matvec_mx(self.head.q, self.head.s, xq, d)


HF_MODELS = {"tc1": ROOT / "data" / "tc1", "qwen3.5-4b": ROOT / "data" / "qwen3.5-4b"}


class Lazy(dict):
    def __getitem__(self, k):
        return super().__getitem__(k)()


def read_safetensors(path):
    raw = np.memmap(path, np.uint8, "r")
    n = int(raw[:8].view(np.uint64)[0])
    out = {}
    for name, t in json.loads(bytes(raw[8:8 + n])).items():
        if name != "__metadata__":
            assert t["dtype"] in ("BF16", "U8", "F32"), t["dtype"]
            a, b = t["data_offsets"]
            out[name] = raw[8 + n + a:8 + n + b].view({"BF16": np.uint16, "U8": np.uint8, "F32": np.float32}[t["dtype"]]).reshape(t["shape"])
    return out


def read_sharded(d):
    out = {}
    for f in sorted(set(json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values())):
        out.update(read_safetensors(d / f))
    return out


def bf16(a):
    return (np.asarray(a).astype(np.uint32) << 16).view(np.float32)


def qwen_config(name, max_seq=512):
    hc = json.loads((HF_MODELS[name] / "config.json").read_text())["text_config"]
    rp = hc["rope_parameters"]
    return {"vocab_size": hc["vocab_size"], "d_model": hc["hidden_size"], "n_layers": hc["num_hidden_layers"], "n_heads": hc["num_attention_heads"],
            "n_kv_heads": hc["num_key_value_heads"], "head_dim": hc["head_dim"], "d_ff": hc["intermediate_size"], "max_seq": max_seq,
            "rope_theta": float(rp["rope_theta"]), "rope_dim": int(hc["head_dim"] * rp["partial_rotary_factor"]), "norm_eps": hc["rms_norm_eps"],
            "layer_types": hc["layer_types"], "lk_heads": hc["linear_num_key_heads"], "lv_heads": hc["linear_num_value_heads"],
            "lk_dim": hc["linear_key_head_dim"], "lv_dim": hc["linear_value_head_dim"], "conv": hc["linear_conv_kernel_dim"]}


def tiny_model(seed=0, bitnet=False):
    cfg = {"vocab_size": 512, "d_model": 256, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2, "head_dim": 64, "d_ff": 768,
           "max_seq": 256, "rope_theta": 10000.0, "rope_scaling": None, "norm_eps": 1e-5,
           "act": "relu2" if bitnet else "silu", "subnorm": bitnet}
    r = np.random.default_rng(seed)
    g = lambda *shape: (r.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)
    norm = lambda: (1 + 0.1 * r.standard_normal(cfg["d_model"])).astype(np.float32)
    D, H, KV, hd, F = cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"], cfg["d_ff"]
    w = {"tok_emb": g(cfg["vocab_size"], D), "final_norm": norm()}
    for l in range(cfg["n_layers"]):
        w.update({f"l{l}.wq": g(H * hd, D), f"l{l}.wk": g(KV * hd, D), f"l{l}.wv": g(KV * hd, D), f"l{l}.wo": g(D, H * hd),
                  f"l{l}.attn_norm": norm(), f"l{l}.mlp_norm": norm(), f"l{l}.w_gate": g(F, D), f"l{l}.w_up": g(F, D),
                  f"l{l}.w_down": g(D, F), f"l{l}.attn_sub_norm": norm()[:H * hd], f"l{l}.ffn_sub_norm": np.resize(norm(), F)})
    return w, cfg, {"config": cfg}


def tiny_qwen(seed=0, wide=False):
    hd, dk = (128, 128) if wide else (64, 32)
    cfg = {"vocab_size": 512, "d_model": 256, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2, "head_dim": hd, "rope_dim": hd // 4, "d_ff": 512,
           "max_seq": 256, "rope_theta": 1e7, "norm_eps": 1e-6, "layer_types": ["linear_attention"] * 3 + ["full_attention"],
           "lk_heads": 2, "lv_heads": 4, "lk_dim": dk, "lv_dim": dk, "conv": 4}
    r = np.random.default_rng(seed)
    g = lambda *shape: (r.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)
    n = lambda k: (0.1 * r.standard_normal(k)).astype(np.float32)
    D, H, KV, hd, F, Hk, Hv, Dk, Dv = (cfg[k] for k in ("d_model", "n_heads", "n_kv_heads", "head_dim", "d_ff", "lk_heads", "lv_heads", "lk_dim", "lv_dim"))
    w = {"tok_emb": g(cfg["vocab_size"], D), "final_norm": n(D)}
    for l, t in enumerate(cfg["layer_types"]):
        p = f"l{l}."
        w.update({p + "attn_norm": n(D), p + "mlp_norm": n(D), p + "w_gate": g(F, D), p + "w_up": g(F, D), p + "w_down": g(D, F)})
        if t == "linear_attention":
            w.update({p + "in_qkv": g(2 * Hk * Dk + Hv * Dv, D), p + "in_z": g(Hv * Dv, D), p + "in_b": g(Hv, D), p + "in_a": g(Hv, D),
                      p + "out": g(D, Hv * Dv), p + "conv": (0.4 * r.standard_normal((2 * Hk * Dk + Hv * Dv, cfg["conv"]))).astype(np.float32),
                      p + "dt_bias": (1 + 0.2 * r.standard_normal(Hv)).astype(np.float32),
                      p + "A_log": np.log(r.uniform(0.5, 16, Hv)).astype(np.float32), p + "lnorm": (1 + n(Dv)).astype(np.float32)})
        else:
            w.update({p + "wq": g(H * hd, D), p + "wg": g(H * hd, D), p + "wk": g(KV * hd, D), p + "wv": g(KV * hd, D), p + "wo": g(D, H * hd),
                      p + "q_norm": n(hd), p + "k_norm": n(hd)})
    return w, cfg, {"config": cfg}


def rope_perm(hd, rd):
    return np.concatenate([np.stack([np.arange(rd // 2), np.arange(rd // 2, rd)], 1).reshape(-1), np.arange(rd, hd)])


def qwen_weights(name, max_seq):
    d = HF_MODELS[name]
    cfg = qwen_config(name, max_seq)
    H, KV, hd, rd = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"], cfg["rope_dim"]
    t = read_sharded(d)
    perm = rope_perm(hd, rd)
    f = lambda k: bf16(t[k]) if t[k].dtype == np.uint16 else np.asarray(t[k], np.float32)
    rows = lambda k, h: (lambda: f(k).reshape(h, hd, -1)[:, perm].reshape(h * hd, -1))
    m = "model.language_model."
    w = Lazy({"tok_emb": lambda: f(m + "embed_tokens.weight"), "final_norm": lambda: f(m + "norm.weight")})
    for l, kind in enumerate(cfg["layer_types"]):
        p, q = f"l{l}.", f"{m}layers.{l}."
        w[p + "attn_norm"] = lambda q=q: f(q + "input_layernorm.weight")
        w[p + "mlp_norm"] = lambda q=q: f(q + "post_attention_layernorm.weight")
        for a, b in (("w_gate", "gate_proj"), ("w_up", "up_proj"), ("w_down", "down_proj")):
            w[p + a] = lambda q=q, b=b: f(q + f"mlp.{b}.weight")
        if kind == "linear_attention":
            a = q + "linear_attn."
            for k, n in (("in_qkv", "in_proj_qkv.weight"), ("in_z", "in_proj_z.weight"), ("in_b", "in_proj_b.weight"), ("in_a", "in_proj_a.weight"),
                         ("out", "out_proj.weight"), ("dt_bias", "dt_bias"), ("A_log", "A_log"), ("lnorm", "norm.weight")):
                w[p + k] = lambda a=a, n=n: f(a + n)
            w[p + "conv"] = lambda a=a: f(a + "conv1d.weight")[:, 0, :]
        else:
            a = q + "self_attn."
            qg = lambda a=a: f(a + "q_proj.weight").reshape(H, 2, hd, -1)
            w[p + "wq"] = lambda qg=qg: qg()[:, 0][:, perm].reshape(H * hd, -1)
            w[p + "wg"] = lambda qg=qg: qg()[:, 1].reshape(H * hd, -1)
            w[p + "wk"] = rows(a + "k_proj.weight", KV)
            w[p + "wv"] = lambda a=a: f(a + "v_proj.weight")
            w[p + "wo"] = lambda a=a: f(a + "o_proj.weight")
            w[p + "q_norm"] = lambda a=a: f(a + "q_norm.weight")[perm]
            w[p + "k_norm"] = lambda a=a: f(a + "k_norm.weight")[perm]
    return w, cfg, {"config": cfg, "tokenizer": str(d / "tokenizer.json")}


def load_weights(name="tiny", max_seq=512):
    if name in ("tiny", "tiny-bitnet"):
        return tiny_model(bitnet=name == "tiny-bitnet")
    if name in ("tiny-qwen", "tiny-qwen-w"):
        return tiny_qwen(wide=name == "tiny-qwen-w")
    if name.startswith("qwen"):
        return qwen_weights(name, max_seq)
    if name == "tc1":
        d = HF_MODELS[name]
        cfg = json.loads((d / "config.json").read_text()) | {"max_seq": max_seq}
        return dict(read_safetensors(d / "model.safetensors")), cfg, {"config": cfg, "tokenizer": str(d / "tokenizer.json")}
    raise KeyError(name)


def chat_text(name, prompt):
    if name == "tc1":
        return prompt.strip()
    if name.startswith("qwen"):
        return f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    raise KeyError(name)


def chip_image(name, wbits=4, hbits=8, max_seq=1024):
    return HF_MODELS[name] / f"izfast_w{wbits}h{hbits}_s{max_seq}.bin"


def qwen_from_image(path):
    raw = np.memmap(path, np.uint8, "r")
    assert bytes(raw[:8]) == b"IZFQWEN1"
    off = 8
    take = lambda n, dt: (lambda a: (a, off + a.nbytes))(np.frombuffer(raw, dt, n, off))
    h, off = take(23, "<i4")
    L, D, H, KV, HD, RD, F, V, S, INV_D, HS, LS, WB, HB, _, EPS, Hk, Hv, Dk, Dv, K, _, _ = (int(x) for x in h)
    kinds, off = take(L, "<i4")
    E, off = take(2 + 4 * L, "<i4")
    fx, off = take(3 * L, "<i4")
    off += 4 * (192 + 256 + 1024 + 4096 + 256 + 256)
    cos, off = take(S * RD // 2, "<i4")
    sin, off = take(S * RD // 2, "<i4")
    m = object.__new__(FixedQwen)
    C = 2 * Hk * Dk + Hv * Dv
    m.kinds = ["lin" if k else "att" for k in kinds]
    m.c = {"max_seq": S, "layer_types": ["linear_attention" if k else "full_attention" for k in kinds]}
    m.rank, m.lora, m.gains, m.relu2, m.subnorm = 0, [], False, False, False
    m.wbits, m.hbits, m.L, m.d, m.F, m.H, m.KV, m.hd, m.rd = WB, HB, L, D, F, H, KV, HD, RD
    m.Hk, m.Hv, m.Dk, m.Dv, m.K, m.eps, m.inv_d, m.inv_f, m.hs, m.ls = Hk, Hv, Dk, Dv, K, EPS, INV_D, inv_count(F), HS, LS
    m.cos, m.sin, m.n_att = cos.reshape(S, RD // 2).astype(np.int64), sin.reshape(S, RD // 2).astype(np.int64), m.kinds.count("att")
    m.layers = []
    for l, kind in enumerate(m.kinds):
        lay = {}
        if kind == "lin":
            cw, off = take(C * K, "<i2")
            dt, off = take(Hv, "<i4")
            A, off = take(Hv, "<i2")
            nw, off = take(Dv, "<i2")
            lay.update(conv=(cw.reshape(C, K).astype(np.int64), int(fx[3 * l])), dt=dt.astype(np.int64),
                       A=(A.astype(np.int64), int(fx[3 * l + 1])), norm=(nw.astype(np.int64), int(fx[3 * l + 2])))
        else:
            qn, off = take(HD, "<i2")
            kn, off = take(HD, "<i2")
            lay.update(qn=(qn.astype(np.int64), int(fx[3 * l])), kn=(kn.astype(np.int64), int(fx[3 * l + 1])))
        m.layers.append(lay)
    shapes = [(V, D)]
    for kind in m.kinds:
        shapes += [(C + Hv * Dv + 2 * Hv if kind == "lin" else (2 * H + 2 * KV) * HD, D), (D, Hv * Dv if kind == "lin" else H * HD), (2 * F, D), (D, F)]
    shapes.append((V, D))
    bits = [HB] + [WB] * 4 * L + [HB]
    qs = []
    for (R, Cc), b in zip(shapes, bits):
        n = R * Cc // (1 if b == 8 else 2)
        a, off = take(n, np.int8)
        if b != 8:
            a = a.reshape(R, Cc // GROUP, GROUP // 2)
            q = np.empty((R, Cc // GROUP, 2, GROUP // 2), np.int8)
            q[:, :, 0], q[:, :, 1] = (a << 4) >> 4, a >> 4
            a = q
        qs.append(a.reshape(R, Cc // GROUP, GROUP))
    mats = []
    for (R, Cc), q, e in zip(shapes, qs, E):
        sc, off = take(R * Cc // GROUP, "<u2")
        mats.append(QMat(q, sc.reshape(R, Cc // GROUP).astype(np.int32), int(e)))
    assert off == raw.size
    m.emb, m.head = mats[0], mats[-1]
    for l in range(L):
        m.layers[l].update(zip(("in", "o", "gu", "down"), mats[1 + 4 * l:5 + 4 * l]))
    return m


def load(wbits=4, name="tiny", max_seq=512, hbits=None, lora=0):
    if name.startswith("qwen") and chip_image(name, wbits, hbits or 8).exists():
        return qwen_from_image(chip_image(name, wbits, hbits or 8)), {"tokenizer": str(HF_MODELS[name] / "tokenizer.json")}
    w, cfg, meta = load_weights(name, max_seq)
    if "layer_types" in cfg:
        return FixedQwen(w, cfg, wbits, hbits or 8), meta
    return FixedLlama(w, cfg, wbits, hbits, lora), meta

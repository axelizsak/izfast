import json
from dataclasses import dataclass, field

import numpy as np

from .ref import (EXP2, EXP2_HI, EXP2_LO, GROUP, L2E, RSQRT, SIGMOID, SOFTPLUS, FixedQwen, bank_layout, head_scale, inv_count, pack_trits,
                  pack_weight_word, row_scales)

OPS = {"END": 0, "EMBED": 1, "QMV": 2, "RESCALE": 3, "ATTN": 4, "ARGMAX": 5, "GAIN": 6, "HNORM": 7, "GATE": 8, "FEED": 9, "DELTA": 10}
RMS, RESIDUAL, SILU, WIDE, STORE = 1, 2, 4, 8, 16
COPY = 1
GAIN_ALIGN = 512
FIELDS = (("op", 8), ("a", 32), ("b", 32), ("c", 32), ("n", 32), ("m", 32), ("imm", 16), ("flags", 8), ("k", 32))


def encode(op, **kw):
    kw["op"] = OPS[op]
    kw["imm"] = kw.get("imm", 0) & 0xFFFF
    word, pos = 0, 0
    for name, width in FIELDS:
        value = kw.get(name, 0)
        assert 0 <= value < (1 << width), (op, name, value)
        word |= value << pos
        pos += width
    return word


@dataclass(frozen=True)
class Shape:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    subnorm: bool = False
    gains: bool = False
    wide: bool = False
    lora: int = 0


@dataclass
class Image:
    program: list = field(default_factory=list)
    banks: list = field(default_factory=list)
    emb: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    vmem: dict = field(default_factory=dict)
    words_per_lane: int = 0
    wide_words: int = 0
    gain_len: int = 0
    scales: list = field(default_factory=list)
    scale_depth: int = 0


def vmem_map(s, align=512):
    up = lambda n: -(-n // align) * align
    ATT = up(s.d_model)
    M = ATT + up(s.n_heads * s.head_dim)
    GX = M + up(s.d_ff)
    T = GX + (up(s.d_model) if s.gains else 0)
    return {"X": 0, "ATT": ATT, "M": M, "GX": GX, "T": T, "end": T + (up(s.lora) if s.lora else 0)}


def gain_layout(s):
    if not s.gains:
        return {}, 0
    n = {"qkv": s.d_model, "o": s.n_heads * s.head_dim, "gu": s.d_model, "down": s.d_ff}
    keys = ("qkv", "o", "gu", "down") if s.subnorm else ("qkv", "gu")
    base, out = 0, {}
    for l in range(s.n_layers):
        for k in keys:
            out[(l, k)] = base
            base += -(-n[k] // GAIN_ALIGN) * GAIN_ALIGN
    return out, base


def emit_program(img, s, mats, lanes):
    d, H, KV, hd, F = s.d_model, s.n_heads, s.n_kv_heads, s.head_dim, s.d_ff
    assert d % GROUP == 0 and F % GROUP == 0 and hd % GROUP == 0 and H % KV == 0
    v = img.vmem = vmem_map(s)
    base, hbase, sbase, bases = 0, 0, 0, {}
    for name, (rows, groups, E) in mats.items():
        if name == "emb":
            continue
        words = -(-rows // lanes) * groups
        if s.wide and is_wide(name):
            bases[name] = (hbase, rows, groups, E, sbase)
            hbase += words
        else:
            bases[name] = (base, rows, groups, E, sbase)
            base += words
        sbase += -(-rows // lanes)
    img.scale_depth, img.words_per_lane, img.wide_words = sbase, base, hbase

    def emit(op, **kw):
        img.program.append(encode(op, **kw))

    def qmv(name, src, n_in, rms, extra=0):
        b, rows, groups, E, sb = bases[name]
        assert groups * GROUP == n_in
        emit("QMV", a=src, b=b, c=sb, n=n_in, m=rows, k=inv_count(n_in) if rms else inv_d, flags=int(rms) | extra)
        return rows, E

    gb, img.gain_len = gain_layout(s)

    def gain(l, k, src, dst, n):
        if (l, k) not in gb:
            return src
        emit("GAIN", a=src, b=gb[(l, k)], c=dst, n=n)
        return dst

    wide = WIDE if s.wide else 0

    def residual(l, k, src, n_in, rms):
        if s.lora:
            _, E = qmv(f"l{l}.{k}.lA", src, n_in, rms, wide)
            emit("RESCALE", c=v["T"], n=s.lora, imm=E, flags=STORE | (RMS if rms else 0))
            emit("GAIN", a=src, c=src, n=n_in, flags=COPY)
        rows, E = qmv(f"l{l}.{k}", src, n_in, rms)
        emit("RESCALE", c=v["X"], n=rows, imm=E, flags=RESIDUAL | (RMS if rms else 0))
        if s.lora:
            emit("GAIN", a=v["T"], c=v["T"], n=s.lora, flags=COPY)
            rows, E = qmv(f"l{l}.{k}.lB", v["T"], s.lora, False, wide)
            emit("RESCALE", c=v["X"], n=rows, imm=E, flags=RESIDUAL)

    inv_d = inv_count(d)
    emit("EMBED", c=v["X"], n=d, imm=mats["emb"][2])
    for l in range(s.n_layers):
        rows, E = qmv(f"l{l}.qkv", gain(l, "qkv", v["X"], v["GX"], d), d, True)
        emit("RESCALE", n=rows, imm=E, flags=RMS)
        emit("ATTN", c=v["ATT"], imm=l, k=head_scale(hd))
        residual(l, "o", gain(l, "o", v["ATT"], v["ATT"], H * hd), H * hd, s.subnorm)
        rows, E = qmv(f"l{l}.gu", gain(l, "gu", v["X"], v["GX"], d), d, True)
        emit("RESCALE", c=v["M"], n=rows // 2, imm=E, flags=RMS | SILU)
        residual(l, "down", gain(l, "down", v["M"], v["M"], F), F, s.subnorm)
    rows, _ = qmv("head", v["X"], d, False, wide)
    emit("ARGMAX", n=rows)
    emit("END")
    return bases


@dataclass(frozen=True)
class QShape:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_dim: int
    d_ff: int
    vocab: int
    lk_heads: int
    lv_heads: int
    lk_dim: int
    lv_dim: int
    conv: int
    kinds: tuple
    wide: bool = True
    lora: int = 0
    gains: bool = False
    subnorm: bool = False


def qwen_in_rows(s, kind):
    if kind == "lin":
        return 2 * s.lk_heads * s.lk_dim + s.lv_heads * s.lv_dim + 2 * s.lv_heads
    return (2 * s.n_heads + 2 * s.n_kv_heads) * s.head_dim


def qwen_vmem(s, align=512):
    up = lambda n: -(-n // align) * align
    ATT = up(s.d_model)
    M = ATT + up(max(s.n_heads * s.head_dim, s.lv_heads * s.lv_dim))
    Y = M + up(s.d_ff)
    return {"X": 0, "ATT": ATT, "M": M, "Y": Y, "end": Y + up(max(qwen_in_rows(s, k) for k in set(s.kinds)))}


def qwen_gain_layout(s):
    out, base = {}, 0
    for l, k in enumerate(s.kinds):
        if k == "att":
            for t in ("q", "k"):
                out[(l, t)] = base
                base += -(-s.head_dim // GAIN_ALIGN) * GAIN_ALIGN
    return out, base


def mat_bases(mats, lanes, wide):
    base, hbase, sbase, bases = 0, 0, 0, {}
    for name, (rows, groups, E) in mats.items():
        if name == "emb":
            continue
        words = -(-rows // lanes) * groups
        if wide and is_wide(name):
            bases[name] = (hbase, rows, groups, E, sbase)
            hbase += words
        else:
            bases[name] = (base, rows, groups, E, sbase)
            base += words
        sbase += -(-rows // lanes)
    return bases, base, hbase, sbase


def emit_qwen(img, s, mats, lanes, shifts):
    d, H, KV, hd, F = s.d_model, s.n_heads, s.n_kv_heads, s.head_dim, s.d_ff
    assert d % GROUP == 0 and F % GROUP == 0 and hd % GROUP == 0 and H % KV == 0
    v = img.vmem = qwen_vmem(s)
    bases, img.words_per_lane, img.wide_words, img.scale_depth = mat_bases(mats, lanes, s.wide)
    gb, img.gain_len = qwen_gain_layout(s)
    emit = lambda op, **kw: img.program.append(encode(op, **kw))
    inv_d = inv_count(d)

    def qmv(name, src, n_in, rms, extra=0):
        b, rows, groups, E, sb = bases[name]
        assert groups * GROUP == n_in
        emit("QMV", a=src, b=b, c=sb, n=n_in, m=rows, k=inv_count(n_in) if rms else inv_d, flags=int(rms) | extra)
        return rows, E

    emit("EMBED", c=v["X"], n=d, imm=mats["emb"][2])
    il = ia = 0
    for l, kind in enumerate(s.kinds):
        rows, E = qmv(f"l{l}.in", v["X"], d, True)
        emit("RESCALE", c=v["Y"], n=rows, imm=E, flags=STORE | RMS)
        if kind == "lin":
            af, gf, cf = shifts.get(l, (14, 14, 14))
            emit("DELTA", a=v["Y"], c=v["ATT"], imm=il, k=af | gf << 8 | cf << 16)
            n_o, il = s.lv_heads * s.lv_dim, il + 1
        else:
            qf, kf = shifts.get(l, (14, 14))
            emit("HNORM", a=v["Y"], b=gb[(l, "q")], c=v["Y"], n=H * hd, m=hd, k=inv_count(hd), imm=qf, flags=1)
            emit("HNORM", a=v["Y"] + H * hd, b=gb[(l, "k")], c=v["Y"] + H * hd, n=KV * hd, m=hd, k=inv_count(hd), imm=kf, flags=1)
            emit("FEED", a=v["Y"], n=(H + 2 * KV) * hd)
            emit("ATTN", c=v["ATT"], imm=ia, k=head_scale(hd))
            emit("GATE", a=v["ATT"], b=v["Y"] + (H + 2 * KV) * hd, c=v["ATT"], n=H * hd)
            n_o, ia = H * hd, ia + 1
        rows, E = qmv(f"l{l}.o", v["ATT"], n_o, False)
        emit("RESCALE", c=v["X"], n=rows, imm=E, flags=RESIDUAL)
        rows, E = qmv(f"l{l}.gu", v["X"], d, True)
        emit("RESCALE", c=v["M"], n=rows // 2, imm=E, flags=RMS | SILU)
        rows, E = qmv(f"l{l}.down", v["M"], F, False)
        emit("RESCALE", c=v["X"], n=rows, imm=E, flags=RESIDUAL)
    rows, _ = qmv("head", v["X"], d, False, WIDE if s.wide else 0)
    emit("ARGMAX", n=rows)
    emit("END")
    return bases


def qwen_shape_mats(s, E=20):
    d, F = s.d_model, s.d_ff
    mats = {"emb": (s.vocab, d // GROUP, E)}
    for l, k in enumerate(s.kinds):
        mats[f"l{l}.in"] = (qwen_in_rows(s, k), d // GROUP, E)
        mats[f"l{l}.o"] = (d, (s.lv_heads * s.lv_dim if k == "lin" else s.n_heads * s.head_dim) // GROUP, E)
        mats[f"l{l}.gu"] = (2 * F, d // GROUP, E)
        mats[f"l{l}.down"] = (d, F // GROUP, E)
    mats["head"] = (s.vocab, d // GROUP, E)
    return mats


def qwen_model_mats(m):
    H, KV, hd = m.H, m.KV, m.hd
    for l, (layer, kind) in enumerate(zip(m.layers, m.kinds)):
        for k in ("in", "o", "gu", "down"):
            q, sc = layer[k].q, layer[k].s
            if k == "gu":
                order = np.arange(q.shape[0]).reshape(2, -1).T.reshape(-1)
                q, sc = q[order], sc[order]
            if k == "in" and kind == "att":
                order = np.concatenate([np.arange(H * hd), np.arange(2 * H * hd, (2 * H + 2 * KV) * hd), np.arange(H * hd, 2 * H * hd)])
                q, sc = q[order], sc[order]
            yield f"l{l}.{k}", layer[k], q, sc
    yield "head", m.head, m.head.q, m.head.s


def qwen_shape(m):
    return QShape(m.L, m.d, m.H, m.KV, m.hd, m.rd, m.F, m.head.rows, m.Hk, m.Hv, m.Dk, m.Dv, m.K, tuple(m.kinds), m.hbits != m.wbits)


def qwen_shifts(m):
    return {l: ((lay["A"][1], lay["norm"][1], lay["conv"][1]) if k == "lin" else (lay["qn"][1], lay["kn"][1]))
            for l, (lay, k) in enumerate(zip(m.layers, m.kinds))}


def qwen_tables(m):
    s = qwen_shape(m)
    t = ([(0, i, int(v)) for i, v in enumerate(RSQRT)] + [(1, i, int(v)) for i, v in enumerate(EXP2)]
         + [(2, i, int(v)) for i, v in enumerate(SIGMOID)]
         + [(3, i, int(v) & 0xFFFF) for i, v in enumerate(m.cos.reshape(-1))] + [(4, i, int(v) & 0xFFFF) for i, v in enumerate(m.sin.reshape(-1))]
         + [(6, i, int(v)) for i, v in enumerate(SOFTPLUS)] + [(7, i, int(v)) for i, v in enumerate(EXP2_HI)]
         + [(8, i, int(v)) for i, v in enumerate(EXP2_LO)])
    for (l, k), base in qwen_gain_layout(s)[0].items():
        t += [(5, base + j, int(v) & 0xFFFF) for j, v in enumerate(m.layers[l]["qn" if k == "q" else "kn"][0])]
    il = 0
    for lay, k in zip(m.layers, m.kinds):
        if k != "lin":
            continue
        cw = np.asarray(lay["conv"][0], np.int64)
        C = cw.shape[0]
        t += [(9, (il * C + c) * m.K + j, int(cw[c, j]) & 0xFFFF) for c in range(C) for j in range(m.K)]
        t += [(10, il * m.Hv + h, int(v) & 0xFFFFFFFF) for h, v in enumerate(lay["dt"])]
        t += [(11, il * m.Hv + h, int(v) & 0xFFFF) for h, v in enumerate(lay["A"][0])]
        t += [(12, il * m.Dv + j, int(v) & 0xFFFF) for j, v in enumerate(lay["norm"][0])]
        il += 1
    return t


def compile_qwen(m, lanes=8, banks=True):
    s = qwen_shape(m)
    img = Image()
    mats = {"emb": (m.emb.rows, m.emb.q.shape[1], m.emb.E)}
    for name, qm, _, _ in qwen_model_mats(m):
        mats[name] = (qm.rows, qm.q.shape[1], qm.E)
    bases = emit_qwen(img, s, mats, lanes, qwen_shifts(m))
    if banks:
        for name, _, q, sc in qwen_model_mats(m):
            bits = mat_bits(m, name)
            flag = 1 << 31 if s.wide and is_wide(name) else 0
            img.banks.extend((lane, (bases[name][0] + addr) | flag, word) for lane, addr, word in bank_layout(q, sc, lanes, bits))
        G = m.d // GROUP
        img.emb = [(tok * G + g, pack_weight_word(m.emb.q[tok, g], m.emb.s[tok, g], m.hbits))
                   for tok in range(m.emb.rows) for g in range(G)]
    img.tables = qwen_tables(m)
    return img


def shape_mats(s, E=20):
    d, H, KV, hd, F = s.d_model, s.n_heads, s.n_kv_heads, s.head_dim, s.d_ff
    mats = {"emb": (s.vocab, d // GROUP, E)}
    for l in range(s.n_layers):
        mats[f"l{l}.qkv"] = ((H + 2 * KV) * hd, d // GROUP, E)
        mats[f"l{l}.o"] = (d, H * hd // GROUP, E)
        mats[f"l{l}.gu"] = (2 * F, d // GROUP, E)
        mats[f"l{l}.down"] = (d, F // GROUP, E)
        if s.lora:
            mats[f"l{l}.o.lA"] = (s.lora, H * hd // GROUP, E)
            mats[f"l{l}.o.lB"] = (d, s.lora // GROUP, E)
            mats[f"l{l}.down.lA"] = (s.lora, F // GROUP, E)
            mats[f"l{l}.down.lB"] = (d, s.lora // GROUP, E)
    mats["head"] = (s.vocab, d // GROUP, E)
    return mats


def compile_shape(s, lanes):
    img = Image()
    if isinstance(s, QShape):
        emit_qwen(img, s, qwen_shape_mats(s), lanes, {})
    else:
        emit_program(img, s, shape_mats(s), lanes)
    return img


def model_mats(m):
    if isinstance(m, FixedQwen):
        yield from qwen_model_mats(m)
        return
    for l, layer in enumerate(m.layers):
        for k in ("qkv", "o", "gu", "down"):
            q, sc = layer[k].q, layer[k].s
            if k == "gu":
                order = np.arange(q.shape[0]).reshape(2, -1).T.reshape(-1)
                q, sc = q[order], sc[order]
            yield f"l{l}.{k}", layer[k], q, sc
            if m.lora and k in m.lora[l]:
                for tag, qm in zip(("lA", "lB"), m.lora[l][k]):
                    yield f"l{l}.{k}.{tag}", qm, qm.q, qm.s
    yield "head", m.head, m.head.q, m.head.s


def model_shape(m):
    if isinstance(m, FixedQwen):
        return qwen_shape(m)
    return Shape(m.L, m.d, m.H, m.KV, m.hd, m.F, m.head.rows, m.subnorm, m.gains, m.hbits != m.wbits, m.rank)


def is_wide(name):
    return name in ("head", "emb") or name.endswith((".lA", ".lB"))


def mat_bits(m, name):
    return m.hbits if is_wide(name) else m.wbits


def compile_model(m, lanes=8, banks=True):
    if isinstance(m, FixedQwen):
        return compile_qwen(m, lanes, banks)
    s = model_shape(m)
    img = Image()
    mats = {"emb": (m.emb.rows, m.emb.q.shape[1], m.emb.E)}
    for name, qm, _, _ in model_mats(m):
        mats[name] = (qm.rows, qm.q.shape[1], qm.E)
    bases = emit_program(img, s, mats, lanes)
    if banks:
        for name, _, q, sc in model_mats(m):
            bits = mat_bits(m, name)
            flag = 1 << 31 if s.wide and is_wide(name) else 0
            img.banks.extend((lane, (bases[name][0] + addr) | flag, word) for lane, addr, word in bank_layout(q, sc, lanes, bits))
            if bits == 2:
                img.scales.extend((lane, bases[name][4] + blk, v) for lane, blk, v in row_scales(sc, lanes))
        G = m.d // GROUP
        img.emb = [(tok * G + g, pack_weight_word(m.emb.q[tok, g], m.emb.s[tok, g], m.hbits))
                   for tok in range(m.emb.rows) for g in range(G)]
    img.tables = ([(0, i, int(v)) for i, v in enumerate(RSQRT)] + [(1, i, int(v)) for i, v in enumerate(EXP2)]
                  + [(2, i, int(v)) for i, v in enumerate(SIGMOID)]
                  + [(3, i, int(v) & 0x1FFFF) for i, v in enumerate(m.cos.reshape(-1))]
                  + [(4, i, int(v) & 0x1FFFF) for i, v in enumerate(m.sin.reshape(-1))])
    for (l, k), base in gain_layout(s)[0].items():
        img.tables += [(5, base + j, int(v) & 0xFFFF) for j, v in enumerate(m.gain[l][k])]
    return img


def pack_words(q, s, wbits):
    q = np.asarray(q).reshape(-1, GROUP)
    if wbits == 8:
        b = q.astype(np.uint8)
    elif wbits == 2:
        c = np.select([q == 1, q == -1], [1, 3], 0).astype(np.uint8).reshape(-1, GROUP // 4, 4)
        b = c[..., 0] | c[..., 1] << 2 | c[..., 2] << 4 | c[..., 3] << 6
    else:
        c = (q & 15).astype(np.uint8).reshape(-1, GROUP // 2, 2)
        b = c[..., 0] | c[..., 1] << 4
    s = np.asarray(s).reshape(-1).astype(np.int64)
    return np.concatenate([b.astype(np.uint8), np.stack([s & 255, s >> 8], 1).astype(np.uint8)], 1)


def pack_trit_words(q):
    t = np.asarray(q).reshape(-1, GROUP).astype(np.int64) + 1
    t = np.concatenate([t, np.ones((len(t), 3), np.int64)], 1).reshape(-1, 7, 5)
    return (t * 3 ** np.arange(5)).sum(-1).astype(np.uint8)


def word_bytes(bits):
    return 7 if bits == 2 else GROUP * bits // 8 + 2


def hex_rows(words):
    h = np.ascontiguousarray(words[:, ::-1]).tobytes().hex()
    w = 2 * words.shape[1]
    return "\n".join(h[i:i + w] for i in range(0, len(h), w)) + "\n"


def pow2(n):
    return 1 << (max(1, int(n)) - 1).bit_length()


def write_image(m, lanes, out):
    img = compile_model(m, lanes, banks=False)
    shape = model_shape(m)
    nw, hw = word_bytes(m.wbits), word_bytes(m.hbits)
    nw = nw if shape.wide else max(nw, hw)
    sdepth = pow2(img.scale_depth)
    scales = np.zeros((lanes, sdepth), np.int64)
    parts, sbase = {"w": [], "h": []}, 0
    for name, _, q, sc in model_mats(m):
        R, G = sc.shape
        nb = -(-R // lanes)
        qp = np.zeros((nb * lanes, G, GROUP), np.int8)
        sp = np.zeros((nb * lanes, G), np.int64)
        qp[:R], sp[:R] = q, sc
        bits = mat_bits(m, name)
        if bits == 2:
            words = pack_trit_words(qp)
            scales[:, sbase:sbase + nb] = sp[:, 0].reshape(nb, lanes).T
        else:
            words = pack_words(qp, sp, bits)
        sbase += nb
        tag, width = ("h", hw) if shape.wide and is_wide(name) else ("w", nw)
        words = np.pad(words, ((0, 0), (0, width - words.shape[1])))
        parts[tag].append(words.reshape(nb, lanes, G, -1).transpose(1, 0, 2, 3).reshape(lanes, nb * G, -1))
    for tag, ps in parts.items():
        if ps:
            words = np.concatenate(ps, 1)
            for l in range(lanes):
                (out / f"{tag}.{l}.hex").write_text(hex_rows(words[l]))
    (out / "e.hex").write_text(hex_rows(pack_words(m.emb.q, m.emb.s, m.hbits)))
    for l in range(lanes):
        (out / f"s.{l}.hex").write_text("".join(f"{v:04x}\n" for v in scales[l]))
    meta = {"program": [f"{w:064x}" for w in img.program], "tables": img.tables, "vmem": img.vmem,
            "wdepth": pow2(img.words_per_lane), "hdepth": pow2(max(1, img.wide_words)), "sdepth": sdepth,
            "edepth": pow2(m.emb.rows * (m.d // GROUP)), "words": img.words_per_lane}
    (out / "image.json").write_text(json.dumps(meta))
    return meta


def write_mats(f, mats, bits):
    for x, b in zip(mats, bits):
        for r in range(0, x.rows, 8192):
            q = x.q[r:r + 8192]
            if b != 8:
                q = q.reshape(len(q), -1, 2, GROUP // 2)
                q = (q[:, :, 0] & 15) | (q[:, :, 1] << 4)
            f.write(np.ascontiguousarray(q).tobytes())
    for x in mats:
        f.write(np.ascontiguousarray(x.s).astype("<u2").tobytes())


def write_qwen_emu(m, path):
    mats = [m.emb] + [layer[k] for layer in m.layers for k in ("in", "o", "gu", "down")] + [m.head]
    fmt = lambda b: 8 if b == 8 else 4
    hdr = [m.L, m.d, m.H, m.KV, m.hd, m.rd, m.F, m.head.rows, m.c["max_seq"], m.inv_d, m.hs, m.ls, fmt(m.wbits), fmt(m.hbits), L2E, m.eps,
           m.Hk, m.Hv, m.Dk, m.Dv, m.K, inv_count(m.hd), inv_count(m.Dv)]
    kinds = [int(k == "lin") for k in m.kinds]
    fx = [v for lay, k in zip(m.layers, m.kinds) for v in ((lay["conv"][1], lay["A"][1], lay["norm"][1]) if k == "lin" else (lay["qn"][1], lay["kn"][1], 0))]
    with open(path, "wb") as f:
        f.write(b"IZFQWEN1")
        f.write(np.array(hdr + kinds + [x.E for x in mats] + fx, "<i4").tobytes())
        for t in (RSQRT, EXP2, SIGMOID, SOFTPLUS, EXP2_HI, EXP2_LO, m.cos, m.sin):
            f.write(np.asarray(t).astype("<i4").tobytes())
        for lay, k in zip(m.layers, m.kinds):
            if k == "lin":
                f.write(np.asarray(lay["conv"][0]).astype("<i2").tobytes() + np.asarray(lay["dt"]).astype("<i4").tobytes()
                        + np.asarray(lay["A"][0]).astype("<i2").tobytes() + np.asarray(lay["norm"][0]).astype("<i2").tobytes())
            else:
                f.write(np.asarray(lay["qn"][0]).astype("<i2").tobytes() + np.asarray(lay["kn"][0]).astype("<i2").tobytes())
        write_mats(f, mats, [m.hbits] + [m.wbits] * 4 * m.L + [m.hbits])


def write_emu(m, path):
    if isinstance(m, FixedQwen):
        return write_qwen_emu(m, path)
    mats = [m.emb] + [layer[k] for layer in m.layers for k in ("qkv", "o", "gu", "down")] + [m.head]
    mats += [x for ad in m.lora for k in ("o", "down") for x in ad[k]]
    fmt = lambda b: 8 if b == 8 else 4
    hdr = [m.L, m.d, m.H, m.KV, m.hd, m.F, m.head.rows, m.c["max_seq"], m.inv_d, m.hs, fmt(m.wbits), L2E,
           int(m.relu2), int(m.subnorm), m.inv_a, m.inv_f, fmt(m.hbits), int(m.gains), m.rank]
    with open(path, "wb") as f:
        f.write(b"IZFEMU4\0")
        f.write(np.array(hdr + [x.E for x in mats], "<i4").tobytes())
        for t in (RSQRT, EXP2, SIGMOID, m.cos, m.sin):
            f.write(np.asarray(t).astype("<i4").tobytes())
        for g in m.gain:
            for k in ("qkv", "o", "gu", "down"):
                if k in g:
                    f.write(np.asarray(g[k]).astype("<i2").tobytes())
        for i, x in enumerate(mats):
            bits = m.hbits if i == 0 or i > 4 * m.L else m.wbits
            for r in range(0, x.rows, 8192):
                q = x.q[r:r + 8192]
                if bits != 8:
                    q = q.reshape(len(q), -1, 2, GROUP // 2)
                    q = (q[:, :, 0] & 15) | (q[:, :, 1] << 4)
                f.write(np.ascontiguousarray(q).tobytes())
        for x in mats:
            f.write(np.ascontiguousarray(x.s).astype("<u2").tobytes())

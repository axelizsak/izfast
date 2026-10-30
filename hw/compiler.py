import json
from dataclasses import dataclass, field

import numpy as np

from .ref import EXP2, GROUP, L2E, RSQRT, SIGMOID, bank_layout, head_scale, inv_count, pack_weight_word

OPS = {"END": 0, "EMBED": 1, "QMV": 2, "RESCALE": 3, "ATTN": 4, "ARGMAX": 5}
RMS, RESIDUAL, SILU = 1, 2, 4
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


@dataclass
class Image:
    program: list = field(default_factory=list)
    banks: list = field(default_factory=list)
    emb: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    vmem: dict = field(default_factory=dict)
    words_per_lane: int = 0


def vmem_map(s, align=512):
    up = lambda n: -(-n // align) * align
    ATT = up(s.d_model)
    M = ATT + up(s.n_heads * s.head_dim)
    return {"X": 0, "ATT": ATT, "M": M, "end": M + up(s.d_ff)}


def emit_program(img, s, mats, lanes):
    d, H, KV, hd, F = s.d_model, s.n_heads, s.n_kv_heads, s.head_dim, s.d_ff
    assert d % GROUP == 0 and F % GROUP == 0 and hd % GROUP == 0 and H % KV == 0
    v = img.vmem = vmem_map(s)
    base, bases = 0, {}
    for name, (rows, groups, E) in mats.items():
        if name == "emb":
            continue
        bases[name] = (base, rows, groups, E)
        base += -(-rows // lanes) * groups

    def emit(op, **kw):
        img.program.append(encode(op, **kw))

    def qmv(name, src, n_in, rms):
        b, rows, groups, E = bases[name]
        assert groups * GROUP == n_in
        emit("QMV", a=src, b=b, n=n_in, m=rows, k=inv_d, flags=int(rms))
        return rows, E

    inv_d = inv_count(d)
    emit("EMBED", c=v["X"], n=d, imm=mats["emb"][2])
    for l in range(s.n_layers):
        rows, E = qmv(f"l{l}.qkv", v["X"], d, True)
        emit("RESCALE", n=rows, imm=E, flags=RMS)
        emit("ATTN", c=v["ATT"], imm=l, k=head_scale(hd))
        rows, E = qmv(f"l{l}.o", v["ATT"], H * hd, False)
        emit("RESCALE", c=v["X"], n=rows, imm=E, flags=RESIDUAL)
        rows, E = qmv(f"l{l}.gu", v["X"], d, True)
        emit("RESCALE", c=v["M"], n=rows // 2, imm=E, flags=RMS | SILU)
        rows, E = qmv(f"l{l}.down", v["M"], F, False)
        emit("RESCALE", c=v["X"], n=rows, imm=E, flags=RESIDUAL)
    rows, _ = qmv("head", v["X"], d, False)
    emit("ARGMAX", n=rows)
    emit("END")
    return bases


def shape_mats(s, E=20):
    d, H, KV, hd, F = s.d_model, s.n_heads, s.n_kv_heads, s.head_dim, s.d_ff
    mats = {"emb": (s.vocab, d // GROUP, E)}
    for l in range(s.n_layers):
        mats[f"l{l}.qkv"] = ((H + 2 * KV) * hd, d // GROUP, E)
        mats[f"l{l}.o"] = (d, H * hd // GROUP, E)
        mats[f"l{l}.gu"] = (2 * F, d // GROUP, E)
        mats[f"l{l}.down"] = (d, F // GROUP, E)
    mats["head"] = (s.vocab, d // GROUP, E)
    return mats


def compile_shape(s, lanes):
    img = Image()
    emit_program(img, s, shape_mats(s), lanes)
    return img


def model_mats(m):
    for l, layer in enumerate(m.layers):
        for k in ("qkv", "o", "gu", "down"):
            q, sc = layer[k].q, layer[k].s
            if k == "gu":
                order = np.arange(q.shape[0]).reshape(2, -1).T.reshape(-1)
                q, sc = q[order], sc[order]
            yield f"l{l}.{k}", layer[k], q, sc
    yield "head", m.head, m.head.q, m.head.s


def compile_model(m, lanes=8, wbits=4, banks=True):
    s = Shape(m.L, m.d, m.H, m.KV, m.hd, m.F, m.head.rows)
    img = Image()
    mats = {"emb": (m.emb.rows, m.emb.q.shape[1], m.emb.E)}
    for name, qm, _, _ in model_mats(m):
        mats[name] = (qm.rows, qm.q.shape[1], qm.E)
    bases = emit_program(img, s, mats, lanes)
    img.words_per_lane = sum(-(-rows // lanes) * groups for name, (rows, groups, _) in mats.items() if name != "emb")
    if banks:
        for name, _, q, sc in model_mats(m):
            img.banks.extend((lane, bases[name][0] + addr, word) for lane, addr, word in bank_layout(q, sc, lanes, wbits))
        G = m.d // GROUP
        img.emb = [(tok * G + g, pack_weight_word(m.emb.q[tok, g], m.emb.s[tok, g], wbits))
                   for tok in range(m.emb.rows) for g in range(G)]
    img.tables = ([(0, i, int(v)) for i, v in enumerate(RSQRT)] + [(1, i, int(v)) for i, v in enumerate(EXP2)]
                  + [(2, i, int(v)) for i, v in enumerate(SIGMOID)]
                  + [(3, i, int(v) & 0x1FFFF) for i, v in enumerate(m.cos.reshape(-1))]
                  + [(4, i, int(v) & 0x1FFFF) for i, v in enumerate(m.sin.reshape(-1))])
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


def hex_rows(words):
    h = np.ascontiguousarray(words[:, ::-1]).tobytes().hex()
    w = 2 * words.shape[1]
    return "\n".join(h[i:i + w] for i in range(0, len(h), w)) + "\n"


def pow2(n):
    return 1 << (max(1, int(n)) - 1).bit_length()


def write_image(m, lanes, wbits, out):
    img = compile_model(m, lanes, wbits, banks=False)
    per_lane = []
    for _, _, q, sc in model_mats(m):
        R, G = sc.shape
        nb = -(-R // lanes)
        qp = np.zeros((nb * lanes, G, GROUP), np.int8)
        sp = np.zeros((nb * lanes, G), np.int64)
        qp[:R], sp[:R] = q, sc
        per_lane.append(pack_words(qp, sp, wbits).reshape(nb, lanes, G, -1).transpose(1, 0, 2, 3).reshape(lanes, nb * G, -1))
    words = np.concatenate(per_lane, 1)
    wdepth = pow2(words.shape[1])
    with open(out / "w.hex", "w") as f:
        for l in range(lanes):
            f.write(f"@{l * wdepth:x}\n")
            f.write(hex_rows(words[l]))
    (out / "e.hex").write_text(hex_rows(pack_words(m.emb.q, m.emb.s, wbits)))
    meta = {"program": [f"{w:064x}" for w in img.program], "tables": img.tables, "vmem": img.vmem,
            "wdepth": wdepth, "edepth": pow2(m.emb.rows * (m.d // GROUP)), "words": int(words.shape[1])}
    (out / "image.json").write_text(json.dumps(meta))
    return meta


def write_emu(m, wbits, path):
    mats = [m.emb] + [layer[k] for layer in m.layers for k in ("qkv", "o", "gu", "down")] + [m.head]
    hdr = [m.L, m.d, m.H, m.KV, m.hd, m.F, m.head.rows, m.c["max_seq"], m.inv_d, m.hs, 8 if wbits == 8 else 4, L2E]
    with open(path, "wb") as f:
        f.write(b"IZFEMU1\0")
        f.write(np.array(hdr + [x.E for x in mats], "<i4").tobytes())
        for t in (RSQRT, EXP2, SIGMOID, m.cos, m.sin):
            f.write(np.asarray(t).astype("<i4").tobytes())
        for x in mats:
            for r in range(0, x.rows, 8192):
                q = x.q[r:r + 8192]
                if wbits != 8:
                    q = q.reshape(len(q), -1, 2, GROUP // 2)
                    q = (q[:, :, 0] & 15) | (q[:, :, 1] << 4)
                f.write(np.ascontiguousarray(q).tobytes())
        for x in mats:
            f.write(np.ascontiguousarray(x.s).astype("<u2").tobytes())

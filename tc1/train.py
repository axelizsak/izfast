import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "tinystories"
OUT = ROOT / "data" / "tc1"
CFG = {"vocab_size": 4096, "d_model": 256, "n_layers": 8, "n_heads": 4, "n_kv_heads": 2, "head_dim": 64, "d_ff": 768,
       "max_seq": 256, "rope_theta": 10000.0, "rope_scaling": None, "norm_eps": 1e-5, "act": "relu2", "subnorm": True}
EOT = "<|endoftext|>"
LINEAR = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
NORMS = ("attn_norm", "mlp_norm", "attn_sub_norm", "ffn_sub_norm")


def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    path = OUT / "tokenizer.json"
    if path.exists():
        return Tokenizer.from_file(str(path))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=CFG["vocab_size"], special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    with open(DATA / "train.txt", encoding="utf-8") as f:
        tok.train_from_iterator((line for _, line in zip(range(2_000_000), f)), trainer)
    OUT.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def encode(tok, name, max_bytes):
    path = OUT / f"{name}.u16"
    if not path.exists():
        with open(DATA / f"{name}.txt", encoding="utf-8") as f:
            stories = [s.strip() for s in f.read(max_bytes).split(EOT) if s.strip()]
        eot = tok.token_to_id(EOT)
        parts = [np.array([eot] + e.ids, np.uint16) for i in range(0, len(stories), 10000)
                 for e in tok.encode_batch(stories[i:i + 10000])]
        np.concatenate(parts).tofile(path)
    return np.memmap(path, np.uint16, "r")


def ternary(w):
    s = w.abs().mean().clamp(min=1e-5)
    return w + ((w / s).round().clamp(-1, 1) * s - w).detach()


def act8(x):
    s = 127 / x.abs().amax(-1, keepdim=True).clamp(min=1e-5)
    return x + ((x * s).round().clamp(-128, 127) / s - x).detach()


class BitLinear(nn.Linear):
    def forward(self, x):
        return F.linear(act8(x), ternary(self.weight))


class Norm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + CFG["norm_eps"]) * self.weight


def rope(x, cos, sin):
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        D, H, KV, hd, Fd = c["d_model"], c["n_heads"], c["n_kv_heads"], c["head_dim"], c["d_ff"]
        self.shape = H, KV, hd
        self.attn_norm, self.mlp_norm, self.attn_sub_norm, self.ffn_sub_norm = Norm(D), Norm(D), Norm(H * hd), Norm(Fd)
        self.wq, self.wk, self.wv = BitLinear(D, H * hd, bias=False), BitLinear(D, KV * hd, bias=False), BitLinear(D, KV * hd, bias=False)
        self.wo = BitLinear(H * hd, D, bias=False)
        self.w_gate, self.w_up, self.w_down = BitLinear(D, Fd, bias=False), BitLinear(D, Fd, bias=False), BitLinear(Fd, D, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        H, KV, hd = self.shape
        h = self.attn_norm(x)
        q = rope(self.wq(h).view(B, T, H, hd), cos, sin).transpose(1, 2)
        k = rope(self.wk(h).view(B, T, KV, hd), cos, sin).transpose(1, 2).repeat_interleave(H // KV, 1)
        v = self.wv(h).view(B, T, KV, hd).transpose(1, 2).repeat_interleave(H // KV, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, H * hd)
        x = x + self.wo(self.attn_sub_norm(a))
        h = self.mlp_norm(x)
        return x + self.w_down(self.ffn_sub_norm(F.relu(self.w_gate(h)).square() * self.w_up(h)))


class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.tok_emb = nn.Embedding(c["vocab_size"], c["d_model"])
        self.blocks = nn.ModuleList(Block(c) for _ in range(c["n_layers"]))
        self.final_norm = Norm(c["d_model"])
        hd = c["head_dim"]
        f = torch.outer(torch.arange(c["max_seq"]).float(), 1.0 / c["rope_theta"] ** (torch.arange(0, hd, 2).float() / hd))
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x, self.cos[:T], self.sin[:T])
        return self.final_norm(x) @ self.tok_emb.weight.T


def batches(data, n, T, rng, dev):
    ix = rng.integers(0, len(data) - T - 1, n)
    x = torch.from_numpy(np.stack([data[i:i + T + 1] for i in ix]).astype(np.int64)).to(dev)
    return x[:, :-1], x[:, 1:]


def val_loss(m, val, T, dev, n=40):
    rng = np.random.default_rng(1)
    with torch.no_grad():
        return float(np.mean([F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item()
                              for x, y in (batches(val, 16, T, rng, dev) for _ in range(n))]))


def export(m):
    w = {"tok_emb": m.tok_emb.weight, "final_norm": m.final_norm.weight}
    for l, b in enumerate(m.blocks):
        for k in LINEAR:
            p = getattr(b, k).weight
            s = p.abs().mean().clamp(min=1e-5)
            w[f"l{l}.{k}"] = (p / s).round().clamp(-1, 1) * s
        for k in NORMS:
            w[f"l{l}.{k}"] = getattr(b, k).weight
    from safetensors.numpy import save_file
    save_file({k: v.detach().cpu().float().numpy() for k, v in w.items()}, str(OUT / "model.safetensors"))
    (OUT / "config.json").write_text(json.dumps(CFG))


def train(a):
    tok = tokenizer()
    data, val = encode(tok, "train", a.bytes), encode(tok, "valid", 1 << 30)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    m = Model(CFG).to(dev)
    print(f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M parameters, {len(data) / 1e6:.0f}M training tokens", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    T, rng = CFG["max_seq"], np.random.default_rng(0)
    steps = a.tokens // (a.batch * T)
    t0 = time.time()
    for step in range(steps):
        lr = a.lr * min(1.0, (step + 1) / 500) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batches(data, a.batch, T, rng, dev)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            done = (step + 1) * a.batch * T
            print(f"step {step}/{steps} loss {loss.item():.3f} lr {lr:.2e} {done / (time.time() - t0):.0f} tokens/s", flush=True)
        if (step + 1) % 2000 == 0 or step == steps - 1:
            print(f"validation loss {val_loss(m, val, T, dev):.3f}", flush=True)
            torch.save(m.state_dict(), OUT / "ckpt.pt")
            export(m)


def chunks(n, length):
    tok = tokenizer()
    val = encode(tok, "valid", 1 << 30)
    starts = np.flatnonzero(np.asarray(val) == tok.token_to_id(EOT))
    return [np.asarray(val[s:s + length]).astype(np.int64) for s in starts[:n]]


def evaluate(a):
    from hw.eval import chip_logprobs
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = Model(CFG).to(dev)
    m.load_state_dict(torch.load(OUT / "ckpt.pt", map_location=dev))
    cs = chunks(a.chunks, a.length)
    with torch.no_grad():
        lp_f, am_f = [], []
        for c in cs:
            z = torch.log_softmax(m(torch.from_numpy(c[None]).to(dev))[0].float(), -1)
            lp_f.append(z[torch.arange(len(c) - 1), torch.from_numpy(c[1:]).to(dev)].cpu().numpy())
            am_f.append(z[:-1].argmax(-1).cpu().numpy())
    lp_f, am_f = np.concatenate(lp_f), np.concatenate(am_f)
    res = {"model": "tc1", "tokens": int(lp_f.size), "reference": {"perplexity": float(np.exp(-lp_f.mean()))}}
    for spec in a.wbits.split(","):
        lp, am = chip_logprobs("tc1", spec, [list(c) for c in cs])
        res[f"chip_w{spec}"] = {"perplexity": float(np.exp(-lp.mean())), "same_next_token": float((am == am_f).mean())}
    print(json.dumps(res, indent=1))


def sample(a):
    tok = tokenizer()
    subprocess.run([sys.executable, str(ROOT / "hw" / "run.py"), "emu", "tc1", "--wbits", "2", "--hbits", "8"], check=True, capture_output=True)
    exe, img = ROOT / "hw" / "build" / "emu" / "emu", ROOT / "hw" / "build" / "emu" / "tc1_w2h8_s1024.bin"
    ids = [tok.token_to_id(EOT)] + tok.encode(a.prompt).ids
    out = subprocess.run([str(exe), str(img)], input=f"t {a.tokens} {len(ids)} 1 {' '.join(map(str, ids))} {tok.token_to_id(EOT)}\n",
                         capture_output=True, text=True, check=True).stdout.split("\n")
    print(a.prompt + tok.decode([int(f.split()[1]) for f in out if f.startswith("t ")]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--tokens", type=int, default=300_000_000)
    t.add_argument("--bytes", type=int, default=1_200_000_000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-3)
    e = sub.add_parser("eval")
    e.add_argument("--chunks", type=int, default=16)
    e.add_argument("--length", type=int, default=256)
    e.add_argument("--wbits", default="2h8")
    s = sub.add_parser("sample")
    s.add_argument("--prompt", default="Once upon a time, a little robot")
    s.add_argument("--tokens", type=int, default=120)
    a = p.parse_args()
    {"train": train, "eval": evaluate, "sample": sample}[a.cmd](a)

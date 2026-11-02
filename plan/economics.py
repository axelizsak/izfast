import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "results" / "measurements.json").read_text())

SOURCES = {
    "horowitz": "Horowitz, ISSCC 2014, energy table (45 nm)",
    "llmet": "LLMET, arXiv 2607.26491 (2026), LPDDR5 modeled as HBM2E",
    "tom": "TOM, arXiv 2602.20662 (2026), 7 nm compiler ROM",
    "n7sram": "TSMC N7 HD bitcell 0.027 um2, IEDM 2016",
    "sram28": "28 nm HD bitcell 0.120 um2, VLSI 2011",
    "rom28": "derived: 7 nm compiler ROM / 3.28 (TOM 28->7 factor)",
    "wafer": "Silicon Analysts wafer pricing 2024-2026; CSET 2020",
    "mask": "Silicon Analysts; UC Davis photomask handout",
    "mpw": "Europractice 2026 MPW price list",
    "d0": "TSMC N5 ramp D0 0.10/cm2 (AnandTech 2020), applied to all nodes",
    "pkg": "Silicon Analysts fabless tape-out guide: FCBGA $5-20",
}
A = {
    "pj_bit": {"on-chip SRAM/ROM (45 nm, 8 KB macro)": (0.16, "horowitz"), "LPDDR5": (6.6, "llmet")},
    "density_mb_mm2": {"rom_7nm": (34.4, "tom"), "sram_7nm": (37.0 * 0.7, "n7sram"), "rom_28nm": (34.4 / 3.28, "rom28"),
                       "sram_28nm": (1 / 0.120 * 0.7, "sram28")},
    "logic_scale_28nm_vs_7nm": 6.0,
    "wafer_usd": {"7nm": (9500, "wafer"), "28nm": (3000, "wafer")},
    "masks_usd": {"7nm": ((3.5e6, 10e6), "mask"), "28nm": ((1e6, 2e6), "mask")},
    "mpw_eur_mm2": {"TSMC 28 nm HPC+": (9190, "mpw")},
    "d0_cm2": (0.1, "d0"),
    "pkg_test_usd": (12, "pkg"),
    "overhead": 0.15,
}
D, L, F, V, ATT, LIN = 2560, 32, 9216, 248320, 8, 24
HK, HV, DK, DV, H, KV, HD, CK = 16, 32, 128, 128, 16, 4, 256, 4
C = 2 * HK * DK + HV * DV
LAYER_W = LIN * ((C + HV * DV + 2 * HV) * D + D * HV * DV) + ATT * ((2 * H + 2 * KV) * HD * D + D * H * HD) + L * 3 * F * D
HEAD_W = V * D
KV_MBIT = ATT * 1024 * KV * HD * 2 * 8 / 1e6
STATE_MBIT = LIN * HV * DK * DV * 32 / 1e6
CONV_MBIT = LIN * (CK - 1) * C * 48 / 1e6
LANES = 1024
UNITS = {"lanes": "mvu_asap7_w4h8_l8", "vector unit": "vpu_asap7_vk1", "linear-attention unit": "dnu_asap7_vk1", "attention unit": "attn_asap7_vk1_bitnet"}
SCALE = {"vector unit": (4.0, "4 times wider than the synthesized VK 1 version"),
         "linear-attention unit": (32.0, "8 heads of 128 at once against 1 head of 32 in the synthesized version"),
         "attention unit": (2.0, "head 256 against head 128 in the synthesized version")}
MARKET = [("Taalas HC1, datacenter card (815 mm2, 6 nm)", "Llama 3.1 8B hard-wired", 14357, "independent test, heise 2026"),
          ("Qualcomm Snapdragon 8 Elite Gen 5, phone", "3B-class model", 220, "Qualcomm claim, 2025"),
          ("Apple M5 Max, laptop", "Qwen3.5-4B, 4-bit", 148, "third-party benchmark aggregate, 2026"),
          ("NVIDIA Jetson AGX Thor, robotics module", "Qwen3-4B, FP4", 73.7, "NVIDIA TensorRT Edge-LLM benchmark, batch 1"),
          ("Hailo-10H, 2.5 W accelerator", "2B models", 10, "Hailo claim")]


def syn(name):
    return M["synthesis"].get(name, {})


def lane():
    s4, s8 = syn("mvu_asap7_w4h8_l4"), syn("mvu_asap7_w4h8_l8")
    area = (s8["area_um2"] - s4["area_um2"]) / 4
    pj = (s8["power_w_at_lib_period"]["dynamic"] - s4["power_w_at_lib_period"]["dynamic"]) / 4 * 500e-12 * 1e12
    return area, pj, min(s4["fmax_mhz"], s8["fmax_mhz"])


def dies(area_mm2, wafer_usd):
    dpw = math.pi * 150 ** 2 / area_mm2 - math.pi * 300 / math.sqrt(2 * area_mm2)
    ad = area_mm2 / 100 * A["d0_cm2"][0]
    y = ((1 - math.exp(-ad)) / ad) ** 2
    return dpw, y, wafer_usd / (dpw * y)


def cycles():
    per = M["cycles"]["qwen3.5-4b/l1024"]["per_ctx"]
    pts = sorted((int(k), v) for k, v in per.items())
    def at(p):
        for (a, ca), (b, cb) in zip(pts, pts[1:]):
            if p <= b:
                return ca + (cb - ca) * (p - a) / (b - a)
        return pts[-1][1]
    return pts, at


def product():
    la, lpj, lmhz = lane()
    clocks = {"lanes": lmhz} | {k: syn(v)["fmax_mhz"] for k, v in UNITS.items() if k != "lanes"}
    mhz = min(clocks.values())
    slow = min(clocks, key=clocks.get)
    w_mbit = (LAYER_W * 4.5 + HEAD_W * 8.5) / 1e6
    rom = w_mbit / A["density_mb_mm2"]["rom_7nm"][0]
    sram = (KV_MBIT + STATE_MBIT + CONV_MBIT) / A["density_mb_mm2"]["sram_7nm"][0]
    units = {k: syn(v)["area_um2"] * SCALE[k][0] / 1e6 for k, v in UNITS.items() if k != "lanes"}
    logic = LANES * la / 1e6 + sum(units.values())
    die = (rom + sram + logic) * (1 + A["overhead"])
    dpw, y, cost = dies(die, A["wafer_usd"]["7nm"][0])
    pts, at = cycles()
    c0 = pts[0][1]
    tps = mhz * 1e6 / c0
    ttft = lambda p: sum(at(i) for i in range(p)) / (mhz * 1e6)
    e_lanes = LANES * lpj * c0 * 1e-12
    e_units = sum(syn(v)["power_w_at_lib_period"]["dynamic"] * SCALE[k][0] for k, v in UNITS.items() if k != "lanes") * 500e-12 * c0
    e_reads = w_mbit * 1e6 * A["pj_bit"]["on-chip SRAM/ROM (45 nm, 8 KB macro)"][0] * 1e-12
    e_state = 4 * STATE_MBIT * 1e6 * A["pj_bit"]["on-chip SRAM/ROM (45 nm, 8 KB macro)"][0] * 1e-12
    e = e_lanes + e_units + e_reads + e_state
    return {"clocks": clocks, "mhz": mhz, "slow": slow, "w_mbit": w_mbit, "rom": rom, "sram": sram, "units": units, "lane": (la, lpj, lmhz),
            "logic": logic, "die": die, "dpw": dpw, "yield": y, "cost": cost, "pts": pts, "tps": tps, "ttft": ttft, "e": e,
            "e_parts": {"lanes (measured per lane)": e_lanes, "vector, linear-attention and attention units (estimate)": e_units,
                        "weight reads (assumption)": e_reads, "linear-attention state reads and writes (assumption)": e_state},
            "e_lpddr": w_mbit * 1e6 * A["pj_bit"]["LPDDR5"][0] * 1e-12}


def test_chip():
    la, _, _ = lane()
    d, ff, vocab, hk, hv, dk, h, kv, hd = 256, 768, 4096, 2, 4, 64, 4, 2, 64
    lin = (2 * hk * dk + 2 * hv * dk + 2 * hv) * d + d * hv * dk + 3 * ff * d
    att = (2 * h + 2 * kv) * hd * d + d * h * hd + 3 * ff * d
    params = 6 * lin + 2 * att
    rom = (params * 4.5 + vocab * d * 8.5) / 1e6 / A["density_mb_mm2"]["rom_28nm"][0]
    logic = (64 * la + syn("vpu_asap7_vk1")["area_um2"] + syn("dnu_asap7_vk1")["area_um2"] + syn("attn_asap7_vk1_tc1").get("area_um2", 0)) / 1e6
    logic *= A["logic_scale_28nm_vs_7nm"]
    area = (rom + logic) * (1 + A["overhead"])
    return params + vocab * d, rom, logic, area


def main():
    p = product()
    la, lpj, lmhz = p["lane"]
    q = M["quality"].get("eval_qwen_w4h8", {})
    chk = M["rtl_checks"].get("qwen3.5-4b_w4h8_l128", {})
    out = ["# izfast economics model", "",
           "Generated by `plan/economics.py` from `results/measurements.json`. Every number is marked: measured (RTL simulation or synthesis), "
           "estimate (scaled from a measured smaller version, the factor is given) or assumption (published source, listed at the end).", "",
           "## Product: Qwen3.5-4B on one 7 nm chip", "",
           "| Quantity | Value | Kind |", "|---|---|---|",
           f"| Model | Qwen3.5-4B, 24 linear-attention and 8 attention layers, {(LAYER_W + HEAD_W) / 1e9:.2f}B weights on chip | |",
           f"| Weights | 4-bit layers (groups of 32, 16-bit scale), 8-bit output head: {p['w_mbit'] / 8e3:.2f} GB | |",
           f"| Quality | perplexity {q.get('chip_w4h8', {}).get('perplexity', float('nan')):.3f} against {q.get('reference', {}).get('perplexity', float('nan')):.3f} for the original model, "
           f"same next token {100 * q.get('chip_w4h8', {}).get('same_next_token', float('nan')):.1f}% ({q.get('tokens', 0):,} tokens) | measured |",
           f"| RTL check | {chk.get('positions_bit_exact', 0)} of {chk.get('positions', 0)} positions bit-exact against the C emulator, answer \"{chk.get('answer', '')}\" | measured |"]
    out += [f"| Cycles per token, {LANES:,} lanes, context {c:,} | {v:,} | measured |" for c, v in p["pts"]]
    out += [f"| Clock of the {k} | {v:,.1f} MHz | measured, synthesis before placement |" for k, v in p["clocks"].items()]
    out += [f"| Chip clock (slowest unit, the {p['slow']}) | {p['mhz']:,.1f} MHz | measured |",
            f"| Tokens per second, one user, context 0 | {p['tps']:,.0f} | measured cycles x measured clock |",
            f"| First token after a 20-token prompt | {p['ttft'](20) * 1e3:.2f} ms | measured cycles x measured clock |",
            f"| First token after a 500-token prompt | {p['ttft'](500) * 1e3:.1f} ms | measured cycles x measured clock |",
            f"| Weight memory (ROM) | {p['rom']:,.0f} mm2 | assumption: density |",
            f"| KV cache, linear-attention state and convolution state (SRAM) | {p['sram']:,.1f} mm2 | assumption: density |",
            f"| Lanes ({LANES:,} x {la:,.0f} um2) | {LANES * la / 1e6:.2f} mm2 | measured per lane |"]
    out += [f"| {k.capitalize()} | {v:.2f} mm2 | estimate: {SCALE[k][1]} |" for k, v in p["units"].items()]
    out += [f"| Die, with {A['overhead']:.0%} overhead | {p['die']:,.0f} mm2 | |",
            f"| Good dies per 300 mm wafer | {p['dpw'] * p['yield']:,.0f} | assumption: defect density |",
            f"| Die cost | ${p['cost']:,.0f} | assumption: wafer price |",
            f"| Unit cost with package and test | ${p['cost'] + A['pkg_test_usd'][0]:,.0f} | |",
            f"| Energy per token | {p['e'] * 1e3:.2f} mJ | sum below |"]
    out += [f"| - {k} | {v * 1e3:.2f} mJ | |" for k, v in p["e_parts"].items()]
    out += [f"| Power at full speed | {p['e'] * p['tps']:.1f} W | |",
            f"| Power at 100 tokens/s | {p['e'] * 100:.2f} W | |",
            f"| Same weights read from LPDDR5 instead | {p['e_lpddr'] * 1e3:.0f} mJ per token | assumption |", "",
            "## The same model size on today's hardware (published figures, one user)", "",
            "| Hardware | Model | Tokens/s | Source | izfast speed-up |", "|---|---|---|---|---|"]
    out += [f"| {n} | {m} | {t:,} | {s} | {p['tps'] / t:,.1f}x |" for n, m, t, s in MARKET]
    params, rom, logic, area = test_chip()
    out += ["", "## Test chip (shared wafer run, plan)", "",
            f"A small model of the same family (6 linear-attention and 2 attention layers, width 256, {params / 1e6:.1f}M weights, 4-bit layers and 8-bit head, "
            "to be trained) on 64 lanes, "
            "with the synthesized vector, linear-attention and attention units. Logic areas are the ASAP7 syntheses scaled to 28 nm.", "",
            "| Process | ROM | Logic | Die | MPW cost |", "|---|---|---|---|---|"]
    for k, (eur, _) in A["mpw_eur_mm2"].items():
        out.append(f"| {k} | {rom:.2f} mm2 | {logic:.2f} mm2 | {area:.1f} mm2 | EUR {area * eur:,.0f} |")
    fp = M.get("fpga", {})
    if fp:
        out += ["", "## FPGA prototype (Yosys synthesis for AMD UltraScale+, estimate before Vivado place and route)", "",
                "| Configuration | LUTs | Flip-flops | DSP blocks | On-chip memory |", "|---|---|---|---|---|"]
        out += [f"| {k} | {v['luts']:,} | {v['ffs']:,} | {v['dsps']:,} | {v['mbit']:.1f} Mbit |" for k, v in fp.items()]
    out += ["", "## Assumptions", "", "| Assumption | Value | Source |", "|---|---|---|"]
    out += [f"| Read energy, {k} | {v} pJ/bit | {SOURCES[s]} |" for k, (v, s) in A["pj_bit"].items()]
    out += [f"| Density, {k} | {v:.1f} Mbit/mm2 | {SOURCES[s]} |" for k, (v, s) in A["density_mb_mm2"].items()]
    out += [f"| 300 mm wafer, {k} | ${v:,} | {SOURCES[s]} |" for k, (v, s) in A["wafer_usd"].items()]
    out += [f"| Mask set, {k} | ${v[0] / 1e6:.1f}-{v[1] / 1e6:.1f}M | {SOURCES[s]} |" for k, (v, s) in A["masks_usd"].items()]
    out += [f"| MPW, {k} | EUR {v:,}/mm2 | {SOURCES[s]} |" for k, (v, s) in A["mpw_eur_mm2"].items()]
    out += [f"| Defect density | {A['d0_cm2'][0]}/cm2 | {SOURCES['d0']} |", f"| Package and test | ${A['pkg_test_usd'][0]} | {SOURCES['pkg']} |",
            f"| Logic area at 28 nm vs 7 nm | x{A['logic_scale_28nm_vs_7nm']:.0f} | assumption |",
            "| Embedding table (8-bit) in external flash, one row read per token | | assumption |",
            "| Unit energies are dynamic power at 50% input activity charged on every cycle, an upper bound | | assumption |",
            f"| Die overhead (I/O, clocking, routing) | {A['overhead']:.0%} | assumption |", ""]
    (ROOT / "plan" / "economics.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()

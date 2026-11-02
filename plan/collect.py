import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "hw" / "build"
OUT = ROOT / "results" / "measurements.json"


def last_json(path):
    text = path.read_text()
    try:
        return json.loads(text[text.rfind("\n{") + 1:])
    except ValueError:
        return None


def main():
    m = json.loads(OUT.read_text()) if OUT.exists() else {}
    cycles = m.get("cycles", {})
    for f in sorted(BUILD.glob("bench_*/bench.json")):
        model, lanes, vk, at, seq, sh = re.match(r"bench_(.+)_l(\d+)_v(\d+)_a(\d+)_s(\d+)(?:_h(\d+))?$", f.parent.name).groups()
        key = f"{model}/l{lanes}"
        cycles[key] = {"model": model, "lanes": int(lanes), "vk": int(vk), "att_tok": int(at), "max_seq": int(seq), **({"heads_parallel": int(sh)} if sh else {}),
                       "per_ctx": {str(r["ctx"]): r["cycles"] for r in json.loads(f.read_text())}}
    synth = m.get("synthesis", {})
    for sta in sorted(BUILD.glob("synth/*/sta.txt")):
        d, t = sta.parent, sta.read_text()
        stat = (d / "stat.txt").read_text() if (d / "stat.txt").exists() else ""
        fmax = re.search(r"FMAX_MHZ ([\d.]+)", t)
        area = re.search(r"Chip area for module '\\\w+': ([\d.]+)", stat)
        pw = re.search(r"^Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", t, re.M)
        synth[d.name] = {"fmax_mhz": float(fmax.group(1)) if fmax else None, "area_um2": float(area.group(1)) if area else None,
                         **({"power_w_at_lib_period": {"dynamic": float(pw.group(1)) + float(pw.group(2)), "leakage": float(pw.group(3))}}
                            if pw else {})}
    quality = m.get("quality", {})
    for name in ("eval_qwen_w4h8.log",):
        f = BUILD / "exp" / name
        if f.exists() and (r := last_json(f)):
            quality[name.removesuffix(".log")] = r
    checks = m.get("rtl_checks", {})
    for f in sorted((BUILD / "exp").glob("check_*.json")):
        checks[f.stem.removeprefix("check_")] = json.loads(f.read_text())
    fpga = m.get("fpga", {})
    fpga.update({f.stem.removeprefix("fpga_"): json.loads(f.read_text()) for f in sorted((BUILD / "exp").glob("fpga_*.json"))})
    m.update({"cycles": cycles, "synthesis": synth, "quality": quality, "rtl_checks": checks, "fpga": fpga})
    OUT.write_text(json.dumps(m, indent=1, sort_keys=True))
    print(f"{OUT}: {len(cycles)} cycle tables, {len(synth)} syntheses, {len(quality)} quality runs")


if __name__ == "__main__":
    main()

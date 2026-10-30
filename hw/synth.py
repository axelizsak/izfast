import argparse
import json
import os
import re
import subprocess
from pathlib import Path

HW = Path(__file__).resolve().parent
ABC = "+strash;&get,-n;&nf,{D};&put;buffer;upsize,{D};dnsize,{D}"
EDA = Path(os.environ.get("EDA", Path.home() / "eda"))
LIBS = {
    "sky130hd": {"files": ["sky130hd/sky130_fd_sc_hd__tt_025C_1v80.lib"], "seq": 0, "period": 4.0, "scale": 1000,
                 "dont_use": ["sky130_fd_sc_hd__probe*", "sky130_fd_sc_hd__lpflow_*"],
                 "tie": "-hicell sky130_fd_sc_hd__conb_1 HI -locell sky130_fd_sc_hd__conb_1 LO",
                 "constr": "set_driving_cell sky130_fd_sc_hd__buf_1\nset_load 5\n",
                 "buf": ("sky130_fd_sc_hd__buf_8", "A", "X")},
    "asap7": {"files": [f"asap7/asap7sc7p5t_{g}_RVT_TT_nldm_{v}.lib" for g, v in
                        (("AO", "211120"), ("INVBUF", "220122"), ("OA", "211120"), ("SEQ", "220123"), ("SIMPLE", "211120"))],
              "seq": 3, "period": 500.0, "scale": 1,
              "dont_use": ["*x1p*_ASAP7*", "*xp*_ASAP7*", "SDF*", "ICG*"],
              "tie": "-hicell TIEHIx1_ASAP7_75t_R H -locell TIELOx1_ASAP7_75t_R L",
              "constr": "set_driving_cell BUFx2_ASAP7_75t_R\nset_load 3.898\n",
              "buf": ("BUFx8_ASAP7_75t_R", "A", "Y")},
}


def cut_memories(src, dst, top):
    d = json.loads(src.read_text())
    m = d["modules"][top]
    for name, cell in list(m["cells"].items()):
        if cell["type"] != "$mem_v2":
            continue
        tag = re.sub(r"\W", "_", name).strip("_")
        for pin, bits in cell["connections"].items():
            nets = list(dict.fromkeys(b for b in bits if isinstance(b, int)))
            if nets and not pin.endswith("CLK") and "RST" not in pin:
                m["ports"][f"mem_{tag}_{pin}"] = {"direction": "input" if pin == "RD_DATA" else "output", "bits": nets}
        del m["cells"][name]
    dst.write_text(json.dumps(d))


def buffer_fanout(src, dst, top, buf, fanout=12):
    d = json.loads(src.read_text())
    m = d["modules"][top]
    cell_type, pin_in, pin_out = buf
    sinks, nid = {}, 1 + max([b for c in m["cells"].values() for bits in c["connections"].values() for b in bits
                              if isinstance(b, int)] + [b for p in m["ports"].values() for b in p["bits"] if isinstance(b, int)])
    for name, c in m["cells"].items():
        for pin, bits in c["connections"].items():
            if c["port_directions"][pin] == "input":
                for k, b in enumerate(bits):
                    if isinstance(b, int):
                        sinks.setdefault(b, []).append((name, pin, k))
    clk = set(m["ports"]["clk"]["bits"])
    count = 0
    for net, loads in list(sinks.items()):
        if net in clk or len(loads) <= fanout:
            continue
        while len(loads) > fanout:
            nxt = []
            for k in range(0, len(loads), fanout):
                count += 1
                name = f"fanout_buf_{count}"
                m["cells"][name] = {"type": cell_type, "port_directions": {pin_in: "input", pin_out: "output"},
                                    "connections": {pin_in: [net], pin_out: [nid]}, "parameters": {}, "attributes": {}}
                for cname, pin, bit in loads[k:k + fanout]:
                    m["cells"][cname]["connections"][pin][bit] = nid
                nxt.append((name, pin_in, 0))
                nid += 1
            loads = nxt
    dst.write_text(json.dumps(d))
    return count


def run(top, lib, params):
    L = LIBS[lib]
    out = HW / "build" / "synth" / f"{top}_{lib}"
    out.mkdir(parents=True, exist_ok=True)
    libs = [str(EDA / "pdk" / f) for f in L["files"]]
    dont = " ".join(f"-dont_use {c}" for c in L["dont_use"])
    rtl = sorted(str(p) for p in (HW / "rtl").glob("*.sv"))
    subprocess.run(["sv2v", "-DSYNTHESIS", f"--top={top}", f"--write={out / 'design.v'}", *rtl], check=True)
    chp = " ".join(f"-chparam {k} {v}" for k, v in params.items())
    (out / "pre.ys").write_text(f"read_verilog -sv {out / 'design.v'}\nhierarchy -top {top} {chp}\nproc\nflatten\n"
                                f"opt_clean\nmemory -nomap\nopt_clean\nwrite_json {out / 'pre.json'}\n")
    subprocess.run(["yosys", "-q", "-s", str(out / "pre.ys")], check=True)
    cut_memories(out / "pre.json", out / "cut.json", top)
    (out / "abc.constr").write_text(L["constr"])
    rl = "".join(f"read_liberty -lib {f}\n" for f in libs)
    al = " ".join(f"-liberty {f}" for f in libs)
    (out / "syn.ys").write_text(
        f"{rl}read_json {out / 'cut.json'}\nsynth -top {top} -booth -noalumacc -noabc -extra-map +/choices/kogge-stone.v\n"
        f"dfflibmap -liberty {libs[L['seq']]} {dont}\nabc -D {L['period'] * L['scale']:.0f} -constr {out / 'abc.constr'} -script {ABC} {al} {dont}\n"
        f"opt_clean -purge\nsetundef -zero\nhilomap -singleton {L['tie']}\nopt_clean -purge\n"
        f"tee -o {out / 'stat.txt'} stat {al}\nwrite_json {out / 'mapped.json'}\n")
    subprocess.run(["yosys", "-q", "-s", str(out / "syn.ys")], check=True)
    nbuf = buffer_fanout(out / "mapped.json", out / "buffered.json", top, L["buf"])
    (out / "net.ys").write_text(f"read_json {out / 'buffered.json'}\nwrite_verilog -noattr -noexpr -nohex -nodec {out / 'net.v'}\n")
    subprocess.run(["yosys", "-q", "-s", str(out / "net.ys")], check=True)
    (out / "sta.tcl").write_text(
        "suppress_msg 1212\n" + "".join(f"read_liberty {f}\n" for f in libs)
        + f"read_verilog {out / 'net.v'}\nlink_design {top}\ncreate_clock -name clk -period {L['period']} [get_ports clk]\n"
        "set_input_delay 0 -clock clk [all_inputs -no_clocks]\nset_output_delay 0 -clock clk [all_outputs]\n"
        "report_checks -path_delay max -digits 3\nset ws [worst_slack -max]\n"
        f"puts [format \"FMAX_MHZ %.1f\" [expr {{1.0e-6 / [sta::time_ui_sta [expr {{{L['period']} - $ws}}]]}}]]\n")
    r = subprocess.run([str(EDA / "bin" / "sta"), "-no_splash", "-no_init", "-exit", str(out / "sta.tcl")],
                       capture_output=True, text=True, check=True)
    (out / "sta.txt").write_text(r.stdout)
    print(f"{nbuf} fanout buffers inserted")
    return float(re.search(r"FMAX_MHZ ([\d.]+)", r.stdout).group(1)), out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("top")
    p.add_argument("--lib", choices=LIBS, default="sky130hd")
    p.add_argument("--param", nargs="*", default=[])
    a = p.parse_args()
    fmax, out = run(a.top, a.lib, dict(kv.split("=") for kv in a.param))
    print(f"{a.top} {a.lib}: fmax {fmax:.1f} MHz, report in {out / 'sta.txt'}")

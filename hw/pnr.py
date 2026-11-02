import argparse
import re
import subprocess
from pathlib import Path

HW = Path(__file__).resolve().parent
EDA = Path.home() / "eda"
PLAT = EDA / "orfs" / "flow" / "platforms" / "asap7"
LIBS = [EDA / "pdk" / "asap7" / f"asap7sc7p5t_{g}_RVT_TT_nldm_{v}.lib" for g, v in
        (("AO", "211120"), ("INVBUF", "220122"), ("OA", "211120"), ("SIMPLE", "211120"), ("SEQ", "220123"))]


def run(name, mhz, util):
    synth = HW / "build" / "synth" / name
    out = HW / "build" / "pnr" / name
    out.mkdir(parents=True, exist_ok=True)
    top = name.split("_")[0]
    period = 1e6 / mhz
    tcl = "\n".join([
        f"read_lef {PLAT / 'lef' / 'asap7_tech_1x_201209.lef'}",
        f"read_lef {PLAT / 'lef' / 'asap7sc7p5t_28_R_1x_220121a.lef'}",
        *[f"read_liberty {f}" for f in LIBS],
        f"read_verilog {synth / 'net.v'}",
        f"link_design {top}",
        f"create_clock -name clk -period {period:.1f} [get_ports clk]",
        "set_input_delay 0 -clock clk [delete_from_list [all_inputs] [get_ports clk]]",
        "set_output_delay 0 -clock clk [all_outputs]",
        f"source {PLAT / 'setRC.tcl'}",
        "puts PRE",
        "report_checks -path_delay max -from [all_registers -clock_pins] -to [all_registers -data_pins] -digits 3",
        f"initialize_floorplan -utilization {util} -aspect_ratio 1 -core_space 2 -site asap7sc7p5t",
        f"source {PLAT / 'openRoad' / 'make_tracks.tcl'}",
        "place_pins -hor_layers M4 -ver_layers M5 -min_distance 0.048",
        "tapcell -distance 25 -tapcell_master TAPCELL_ASAP7_75t_R -endcap_master TAPCELL_ASAP7_75t_R",
        "global_placement -density 0.6",
        "estimate_parasitics -placement",
        "repair_design",
        "repair_timing -setup",
        "detailed_placement",
        "clock_tree_synthesis -root_buf BUFx4_ASAP7_75t_R -buf_list {BUFx4_ASAP7_75t_R BUFx8_ASAP7_75t_R}",
        "set_propagated_clock [all_clocks]",
        "detailed_placement",
        "set_routing_layers -signal M2-M7 -clock M2-M7",
        "global_route",
        "estimate_parasitics -global_routing",
        "puts POST",
        "report_checks -path_delay max -from [all_registers -clock_pins] -to [all_registers -data_pins] -digits 3",
        "report_design_area",
        "exit"])
    (out / "pnr.tcl").write_text(tcl + "\n")
    r = subprocess.run([str(EDA / "bin" / "openroad"), "-no_splash", "-exit", str(out / "pnr.tcl")], capture_output=True, text=True)
    (out / "pnr.log").write_text(r.stdout + r.stderr)
    pre, post = (r.stdout.split(tag, 1)[1] for tag in ("\nPRE", "\nPOST"))
    fmax = lambda t: 1e6 / (period - float(re.search(r"([-\d.]+)\s+slack", t).group(1)))
    area = re.search(r"Design area ([\d.]+) u\^2", r.stdout)
    print(f"{name}: register to register {fmax(pre):.1f} MHz after synthesis, {fmax(post):.1f} MHz after placement, "
          f"clock tree and global routing; placed area {area.group(1) if area else '?'} um2")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--mhz", type=float, required=True)
    p.add_argument("--util", type=float, default=50)
    a = p.parse_args()
    run(a.name, a.mhz, a.util)

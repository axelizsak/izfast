import json
import os
import time
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge, Timer

from hw.compiler import OPS

IMG = json.loads((Path(os.environ["IMAGE"]) / "image.json").read_text())
REQ = Path(os.environ["REQ"])


async def write_all(dut, we, fields, rows):
    for row in rows:
        for name, value in zip(fields, row):
            getattr(dut, name).value = value
        getattr(dut, we).value = 1
        await RisingEdge(dut.clk)
    getattr(dut, we).value = 0


async def token(dut, tok, pos):
    dut.tok_in.value, dut.pos_in.value, dut.start.value = tok, pos, 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.done)
    await ReadOnly()
    nxt, cycles = int(dut.tok_out.value), int(dut.cycles.value)
    await Timer(1, unit="ns")
    per_op = {}
    for name, code in OPS.items():
        dut.perf_sel.value = code
        await Timer(1, unit="ns")
        per_op[name] = int(dut.perf_val.value)
    return nxt, cycles, per_op


@cocotb.test()
async def serve(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    for name in ("imem_we", "emb_we", "w_we", "tbl_we", "start", "tok_in", "pos_in", "perf_sel", "vm_raddr"):
        getattr(dut, name).value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await write_all(dut, "imem_we", ("imem_addr", "imem_data"), ((a, int(w, 16)) for a, w in enumerate(IMG["program"])))
    await write_all(dut, "tbl_we", ("tbl_sel", "tbl_addr", "tbl_data"), IMG["tables"])
    (REQ / "ready").write_text("1")
    while True:
        f = REQ / "req.json"
        if not f.exists():
            time.sleep(0.01)
            continue
        req = json.loads(f.read_text())
        f.unlink()
        if req.get("quit"):
            break
        toks, n, stop, out = list(req["tokens"]), req["n"], set(req.get("stop", [])), []
        p = len(toks)
        t0 = time.time()
        for pos in range(min(p + n - 1, req["max_seq"])):
            nxt, cycles, per_op = await token(dut, toks[pos], pos)
            gen = pos >= p - 1
            out.append({"pos": pos, "cycles": cycles, "per_op": per_op, "generated": gen})
            if req.get("dump"):
                x = []
                for k in range(req["dump"]):
                    dut.vm_raddr.value = IMG["vmem"]["X"] + k
                    await Timer(1, unit="ns")
                    x.append(dut.vm_rdata.value.to_signed())
                out[-1]["x"] = x
            if gen:
                toks.append(nxt)
                tmp = REQ / "progress.tmp"
                tmp.write_text(json.dumps({"tokens": toks[p:], "sim_seconds": time.time() - t0}))
                tmp.replace(REQ / "progress.json")
                if nxt in stop:
                    break
        tmp = REQ / "resp.tmp"
        tmp.write_text(json.dumps({"tokens": toks, "prompt_len": p, "steps": out, "sim_seconds": time.time() - t0}))
        tmp.replace(REQ / "resp.json")

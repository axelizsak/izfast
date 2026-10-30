import json
import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge, Timer

from hw.compiler import OPS, Shape, compile_shape

SHAPE = Shape(**json.loads(os.environ["SHAPE"]))
LANES = int(os.environ["LANES"])
CTX = [int(c) for c in os.environ["CTX"].split(",")]
OUT = Path(os.environ["OUT"])


@cocotb.test()
async def bench(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    for name in ("imem_we", "emb_we", "w_we", "tbl_we", "start", "tok_in", "pos_in", "perf_sel"):
        getattr(dut, name).value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    img = compile_shape(SHAPE, LANES)
    for addr, word in enumerate(img.program):
        dut.imem_we.value, dut.imem_addr.value, dut.imem_data.value = 1, addr, word
        await RisingEdge(dut.clk)
    dut.imem_we.value = 0
    results = []
    for pos in CTX:
        dut.tok_in.value, dut.pos_in.value, dut.start.value = 0, pos, 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        await RisingEdge(dut.done)
        await ReadOnly()
        cycles = int(dut.cycles.value)
        await Timer(1, unit="ns")
        per_op = {}
        for name, code in OPS.items():
            dut.perf_sel.value = code
            await Timer(1, unit="ns")
            per_op[name] = int(dut.perf_val.value)
        results.append({"ctx": pos, "cycles": cycles, "per_op": per_op, "instructions": len(img.program)})
        dut._log.info(f"ctx {pos}: {cycles} cycles")
    OUT.write_text(json.dumps(results, indent=1))

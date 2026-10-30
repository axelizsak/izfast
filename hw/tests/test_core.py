import os

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge, Timer

from hw.compiler import compile_model
from hw.ref import load

WBITS = int(os.environ.get("WBITS", "4"))
LANES = int(os.environ.get("LANES", "8"))
MODEL, META = load(WBITS)
IMG = compile_model(MODEL, LANES, WBITS)


async def write_all(dut, we, fields, rows):
    for row in rows:
        for name, value in zip(fields, row):
            getattr(dut, name).value = value
        getattr(dut, we).value = 1
        await RisingEdge(dut.clk)
    getattr(dut, we).value = 0


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    for name in ("imem_we", "emb_we", "w_we", "tbl_we", "start", "tok_in", "pos_in", "perf_sel", "vm_raddr"):
        getattr(dut, name).value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await write_all(dut, "imem_we", ("imem_addr", "imem_data"), enumerate(IMG.program))
    await write_all(dut, "emb_we", ("emb_addr", "emb_data"), IMG.emb)
    await write_all(dut, "tbl_we", ("tbl_sel", "tbl_addr", "tbl_data"), IMG.tables)
    await write_all(dut, "w_we", ("w_lane", "w_addr", "w_data"), IMG.banks)
    await RisingEdge(dut.clk)


async def run_token(dut, token, pos):
    dut.tok_in.value, dut.pos_in.value, dut.start.value = token, pos, 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.done)
    await ReadOnly()
    tok, cycles, err = int(dut.tok_out.value), int(dut.cycles.value), int(dut.error.value)
    await Timer(1, unit="ns")
    assert err == 0, "core raised its error flag"
    return tok, cycles


async def read_vmem(dut, base, n):
    out = []
    for k in range(n):
        dut.vm_raddr.value = base + k
        await Timer(1, unit="ns")
        out.append(dut.vm_rdata.value.to_signed())
    return np.array(out, dtype=object)


@cocotb.test()
async def test_tokens(dut):
    await setup(dut)
    cache = MODEL.new_cache()
    for pos, tok in enumerate([1, 7, 42, 99, 3, 250, 511, 0, 64, 128, 17, 300]):
        trace = []
        logits = MODEL.step(tok, pos, cache, trace)
        got, cycles = await run_token(dut, tok, pos)
        x = await read_vmem(dut, IMG.vmem["X"], MODEL.d)
        bad = np.flatnonzero(x != trace[-1])
        assert bad.size == 0, f"pos {pos}: {bad.size} residual values differ, first {bad[0]}: {x[bad[0]]} vs {trace[-1][bad[0]]}"
        assert got == int(np.argmax(logits)), f"pos {pos}: token {got} vs {int(np.argmax(logits))}"
        dut._log.info(f"pos {pos:2d}: {tok} -> {got}, bit-exact, {cycles} cycles")


@cocotb.test()
async def test_generation(dut):
    await setup(dut)
    prompt = [5, 77, 300, 12, 9, 400, 31, 2]
    cache = MODEL.new_cache()
    ref, hw = list(prompt), list(prompt)
    for pos in range(len(prompt) + 99):
        logits = MODEL.step(ref[pos], pos, cache)
        got, _ = await run_token(dut, hw[pos], pos)
        if pos >= len(prompt) - 1:
            ref.append(int(np.argmax(logits)))
            hw.append(got)
    dut._log.info(f"reference: {ref}")
    dut._log.info(f"core:      {hw}")
    assert ref == hw

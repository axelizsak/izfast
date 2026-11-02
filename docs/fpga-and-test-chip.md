# FPGA prototype and test chip TC2

Two steps sit between the RTL and a product chip. The FPGA prototype shows the architecture working on real hardware. The test chip TC2 replaces the published numbers the product estimates rely on with measured silicon.

## FPGA prototype

**What it is.** The same RTL in a reduced configuration, on a commercial AMD FPGA board: 16 lanes, the vector unit, the attention unit and the linear-attention unit, running a small model of the Qwen3.5 family (8 layers, 6 linear-attention and 2 attention, width 256, 4,096-token vocabulary). All weights, the embedding table included, sit in the FPGA's on-chip memory.

**What it shows.** The hybrid architecture running on hardware, token by token identical to the C emulator, with a time per token fixed by the program. An FPGA runs at a fraction of an ASIC clock, so the prototype does not show the product speed; that stays RTL cycles times the synthesized clock until the test chip.

**Resources** (Yosys `synth_xilinx` for UltraScale+, memories counted separately, before Vivado place and route; `hw/fpga.py`):

The configuration needs about 704,000 LUTs, 317,000 flip-flops, 6,310 DSP blocks and 53 Mbit of on-chip memory (weights 39.6, embedding 8.9, linear-attention and convolution state 3.6, KV cache for 256 tokens 1.0). Share of each board used:

| Board | Price (approximate list price) | LUTs | DSP blocks | On-chip memory | Fits |
|---|---|---|---|---|---|
| AMD Kria KV260 | about $250 | 601% | 506% | 225% | no |
| AMD ZCU104 (Zynq UltraScale+ ZU7EV) | about $1,700 | 306% | 365% | 137% | no |
| AMD VCU118 (Virtex UltraScale+ VU9P) | about $15,000 | 60% | 92% | 15% | yes |
| AWS EC2 F2 (Virtex UltraScale+ VU47P), rented | about $2 per hour | 54% | 70% | 15% | yes |

**Recommended board: AMD VCU118 board (Virtex UltraScale+ VU9P), about $15,000.** It is the smallest board listed that holds the whole configuration. Its DSP blocks are 92% used in this estimate, so the bring-up may move some multiplies into LUTs (40% of them are free). Development can start on a rented AWS F2 instance, with more DSP margin, before the board arrives. The FPGA clock and the final resource counts come from Vivado.

## Test chip TC2

The first silicon. Its job is to measure what the estimates take from publications today.

| Estimate | Today | Measured on TC2 |
|---|---|---|
| Weights in ROM next to the lanes | ROM density from publications (7 nm) | ROM density and read energy at 28 nm |
| Tokens per second per user | RTL cycles x synthesized clock | real clock after place and route, real tokens per second |
| Energy per token | lane and vector-unit energy from synthesis, published read energy | joules per token on a power supply |
| Bit-exact answers | RTL simulation against the C emulator | silicon against the same emulator, token by token |
| Linear-attention unit | RTL and synthesis | clock, area and energy in silicon |

### Specification

| Item | Choice |
|---|---|
| Model | a small model of the Qwen3.5 family: 8 layers (6 Gated DeltaNet, 2 attention with output gate), width 256, 4,096-token vocabulary, about 7.9M parameters, 4-bit layers and 8-bit head, weights in ROM. Still to be trained; the compiler and the RTL already support the architecture |
| Compute | 64 lanes, vector unit, attention unit, linear-attention unit, all from the current RTL |
| Process | TSMC 28 nm HPC+ through Europractice (default) or ST 28 nm FD-SOI through CMP |
| Area | about 13 mm2 (estimate, `plan/economics.md`: ROM at published density, logic from 7 nm synthesis scaled by 6 to 28 nm, 15% overhead) |
| Interfaces | SPI for tokens in and out, JTAG for test, one clock input, current sense on the core supply |
| Package and quantity | QFN or ceramic package, 50-100 samples, test board designed in-house |
| Run cost | about EUR 122k for the wafer slot (Europractice price per mm2), before packaging, boards and physical design |

### Schedule

Europractice runs TSMC 28 nm HPC+ several times a year. The 2026 schedule had runs in February, September, November and December, with final GDS about three months before parts ship ([Europractice 2026 schedule](https://europractice-ic.com/wp-content/uploads/2025/11/TSMC_EP_MPW_schedule_2026_V1_3b.pdf)). Design registration is due at least three weeks before the GDS deadline. CMP offers ST 28 nm FD-SOI runs as an alternative.

### What has to happen before tape-out

1. Train the TC2 model and check it bit-exact on the Python reference, the C emulator and the RTL (`make check` does this today for Qwen3.5-4B).
2. Close timing after place and route; the vector unit is the critical path today (637 MHz at 7 nm, against 1,035 MHz for the lanes).
3. Integrate the foundry ROM and SRAM macros in place of the behavioural memories.
4. Add test structures: scan, a ROM read-energy test mode, a supply-current sense point.
5. Run the RTL checks on the final netlist with gate-level simulation.

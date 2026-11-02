# Technical brief

Everything in this document is measured on the design unless it says "estimate", "assumption" or cites a source. Raw data: `results/measurements.json`. Cost and energy model: `plan/economics.md`.

## 1. What the chip does

izfast is a single-stream inference engine: one user, one token at a time, every weight read once per token, all weights on the chip. The program is the same for every token and fixed at compile time, so the time of a token depends only on its position in the conversation, never on the data.

The product model is Qwen3.5-4B (Alibaba, Apache 2.0, 2026), text only. It has 32 layers: 24 Gated DeltaNet layers (linear attention, with a memory of the conversation of fixed size) and 8 attention layers with an output gate, partial rotary embedding and per-head normalization of queries and keys. 4.2B weights sit on the chip: 3.57B layer weights at 4 bits and the 0.64B output head at 8 bits. The embedding table is read from external flash, one row per token.

## 2. Architecture

```
            token
              |
   +----------v-----------+        +--------------------------+
   | sequencer (static     |------->| vector unit (VK x 32 /   |
   | program, same for     |        | cycle): embed, quantize, |
   | every token)          |        | norms, gates, residual,  |
   +----------+------------+        | SiLU, argmax             |
              |                     +----^----------+------+---+
   +----------v------------------------- | ----+    |      |
   | matrix unit: LANES lanes             |     |    v      v
   |  lane = weight bank (ROM/SRAM)       |     | +--------------------+ +----------------------+
   |       + 32 multipliers + adder tree  | rows| | attention unit:    | | linear-attention     |
   |       + scale, shift, accumulator    +-----+ | RoPE, KV write,    | | unit: convolution,   |
   |  input vector broadcast to all lanes |       | scores, softmax,   | | gates, state update, |
   +--------------------------------------+       | values; KV cache   | | output norm; state   |
                                                  +--------------------+ +----------------------+
```

- **Matrix unit.** Each lane owns the rows of every weight matrix and reads one group of 32 weights per cycle. Layer weights are 4-bit with a 16-bit scale per group of 32; the output head sits in a second, 8-bit bank, and each instruction selects the datapath. Products go through an adder tree, a scale multiply, a power-of-two shift and a 64-bit accumulator, in a 7-stage pipeline. A matrix with R rows and C columns takes ceil(R / LANES) x C / 32 cycles.
- **Vector unit.** Processes VK x 32 numbers per cycle. It consumes matrix rows as they leave the lanes (scaling, normalization factor, residual add, SiLU gating), so most vector work hides behind weight reads. It also normalizes queries and keys per head, applies the attention output gate and feeds the attention unit. Every vector it writes is tracked on the fly (group maxima, sum of squares), so the next quantization and normalization are ready when the next matrix product starts.
- **Attention unit.** Rotary embedding on the first quarter of each head, KV-cache write and attention for a whole layer in one instruction. Scores are computed for ATT_TOK cached tokens per cycle for all query heads that share a KV head. Softmax uses a 2^-x table and an integer divider.
- **Linear-attention unit.** Runs one Gated DeltaNet layer in one instruction: a causal convolution of width 4 over 8,192 channels with its own state memory, L2 normalization of queries and keys per head, the gates beta = sigmoid(b) and decay = exp(-A softplus(a + dt)) from lookup tables, then the delta-rule update of a 128 x 128 state per head (decay, read with the key, correction, rank-1 write, read with the query), and the output RMS normalization with a gain and a SiLU gate. It processes several heads in parallel (8 in the product) and is pipelined; the state of the 24 layers stays on the chip.
- **Sequencer.** Runs a static program. Per linear-attention layer: `QMV` (input projections), `RESCALE`, `DELTA`, `QMV` (output, added to the residual), then the MLP (`QMV`, `RESCALE` with SiLU, `QMV`, `RESCALE`). Per attention layer: `QMV`, `RESCALE`, `HNORM` (queries), `HNORM` (keys), `FEED`, `ATTN`, `GATE`, `QMV`, then the MLP. Then the output head and `ARGMAX`.

**Number formats.** Weights: 4-bit or 8-bit in groups of 32 with a 16-bit scale and one exponent per matrix. Activations into the lanes: 8-bit with one power-of-two exponent per group of 32. Vectors: 48-bit fixed point (16 fractional bits). KV cache: 8-bit with one exponent per token and head. Linear-attention state: 32-bit fixed point (16 fractional bits), saturated. Normalizations, softmax, sigmoid, softplus and exponentials use small lookup tables.

**Also supported.** Llama-family transformers (grouped-query attention, SwiGLU, RoPE with Llama 3 scaling), ternary weights with an 8-bit head, and low-rank adapters; these paths stay covered by `make test`.

## 3. Verification

Three implementations of the same integer arithmetic must agree bit for bit: `hw/ref.py` (Python reference), `hw/emu.c` (C emulator, used by the demo) and `hw/rtl/` (SystemVerilog, simulated with Verilator and cocotb).

| Check | Result |
|---|---|
| Float reference (`hw/eval.py`) against the published model | "The capital of France is **Paris**."; perplexity 6.99 on 511 tokens of Python documentation |
| `make test`: random Qwen-shaped models (VK 1 and 4, 1, 2 and 4 heads in parallel) and the earlier paths (4/8-bit, ternary with 8-bit head, adapters) | residual stream identical after each of 12 tokens and 100 generated tokens identical, 9 of 9 configurations |
| Python reference against the C emulator, Qwen3.5-4B | identical at every position tested |
| `make check`: Qwen3.5-4B on the RTL, 128 lanes, 8 heads in parallel | 22 of 22 positions identical, same answer ("The capital of France"), 1,082,639 cycles per token at 128 lanes |

## 4. Measured results

### Cycles per token (RTL simulation, 1,024 lanes, VK 4, 8 heads in parallel)

| Context | Cycles | Matrix products | Linear attention | Attention | Output head |
|---|---|---|---|---|---|
| 0 | 184,602 | 113,972 | 43,800 | 7,352 | 19,439 |
| 511 | 189,634 | | | 12,384 | |
| 1,023 | 194,754 | | | 17,504 | |

The weight-read floor at 1,024 lanes is 136,816 cycles (every matrix row spread over the lanes, 32 weights per lane per cycle). Long conversations cost little: 24 of the 32 layers keep a memory of fixed size, so 1,023 tokens of context add 5.5% to a token.

### Clocks and area (Yosys synthesis, OpenSTA timing, ASAP7 7 nm, before placement and routing)

| Unit (synthesized configuration) | Clock | Area |
|---|---|---|
| Lanes, 4-bit layers and 8-bit head (slope between 4 and 8 lanes) | 1,035 MHz | 3,698 um² per lane |
| Vector unit, VK 1 | 637 MHz | 0.32 mm² |
| Linear-attention unit, 1 head of 32 at a time, VK 1 | 700 MHz | 0.36 mm² |
| Attention unit, head 128, 4 query heads per KV head, VK 1 | 937 MHz | 1.09 mm² |
| Chip clock (slowest unit) | 637 MHz | |

Memories are cut out as macro ports, so these numbers cover logic only. The product units are wider than the synthesized ones; their areas in `plan/economics.md` are scaled estimates. An earlier place-and-route of the lanes (October 2026) lost 19% of their synthesized clock.

### Quality (perplexity on 2,044 tokens of English technical prose, Python documentation)

| | Perplexity | Same next token as the original |
|---|---|---|
| Qwen3.5-4B, BF16 weights, float arithmetic | 6.39 | |
| izfast: 4-bit layers, 8-bit head, 8-bit activations, integer arithmetic | 6.67 | 89% |

## 5. From design to product

| | Qwen3.5-4B on izfast, 7 nm |
|---|---|
| Tokens per second, one user | 3,450 at 637 MHz |
| First token after a 20-token prompt | 5.8 ms |
| A 500-token answer | 147 ms |
| Energy per token | 11.2 mJ (upper bound, `plan/economics.md`) |
| Die | about 760 mm² (estimate: published ROM density, measured logic) |
| Unit cost | about $300 (estimate) |

The weight memory dominates the die: 21.5 Gbit of ROM at the published 7 nm density is about 624 mm². Options to shrink it are listed below.

## 6. Silicon roadmap

1. **FPGA prototype.** The same RTL in a reduced configuration on an FPGA board, running a small model of the same family. See `fpga-and-test-chip.md`.
2. **Test chip TC2.** Shared-wafer run at TSMC 28 nm: 64 lanes, the vector, attention and linear-attention units, a small hybrid model in ROM. Goal: measured clock, energy per token, ROM density and read energy.
3. **Product chip.** Qwen3.5-4B class model in ROM, 7 or 6 nm.

## 7. Open problems and limits

- Timing is before placement and routing; wires will cost some frequency.
- The vector unit sets the chip clock at 637 MHz while the lanes reach 1,035 MHz. Splitting its longest stages, and those of the linear-attention unit (700 MHz), is the next speed gain; it is not counted in any number here.
- The product die is large (about 760 mm²), close to the reticle limit. Options: two chiplets in one package, a denser ROM (published ternary ROM reaches 120 Mbit/mm² at 7 nm), 3-bit weights, or a smaller model of the same family (Qwen3.5-2B).
- Memory macros are modelled, not synthesized. ROM density and read energy come from publications until the test chip measures them.
- 4-bit weights cost 4.4% of perplexity with plain round-to-nearest quantization; better quantization methods are open.
- Text only: the vision encoder of Qwen3.5 is not supported.

## 8. Reproduce

```
make setup && make models     # Python env, Qwen3.5-4B converted to the chip format
make test                     # 9 RTL configurations
make check                    # Qwen3.5-4B on the RTL against the emulator, about 40 min
make bench                    # cycles per token
make synth                    # clocks and areas
.venv/bin/python plan/collect.py && .venv/bin/python plan/economics.py
```

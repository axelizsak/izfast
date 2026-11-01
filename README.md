# izfast

Today's computing hardware was not built to run AI models. GPUs and phone NPUs keep the model in a separate memory chip, and for every word they write they read the whole model across a narrow connection. izfast is a chip built around one model, so the model runs as fast as it can, locally, where latency matters: robots, vehicles, machines, defence systems.

The chip runs Qwen3.5-4B (Apache 2.0), answering one person at a time. Every weight sits in a memory bank next to its multipliers, and the chip runs the same fixed program for every token. The model mixes 24 linear-attention layers, which keep a fixed-size memory of the conversation, with 8 classic attention layers, and the chip has a unit for each.

There is no silicon yet. Cycle counts come from simulating the RTL with Verilator, clock speeds from synthesis with Yosys and OpenSTA for a 7 nm process (ASAP7), before place and route.

Demo: https://axelizsak.github.io/izfast/

| | Qwen3.5-4B on izfast |
|---|---|
| Cycles per token, 1,024 lanes | 184,602 at the start of a conversation, 194,754 after 1,023 tokens |
| Clock (slowest unit, 7 nm synthesis) | 637 MHz |
| Tokens per second, one user | 3,450 |
| A 500-token answer | 147 ms |
| Quality, 4-bit weights and 8-bit output head | perplexity 6.67 against 6.39 for the original model |

The answer computed by the chip's arithmetic is the same, bit for bit, in the Python reference, the C emulator that the demo uses and the RTL simulation.

For comparison, published speeds for one user on models of this size: NVIDIA Jetson AGX Thor 74 tokens/s (Qwen3-4B), Apple M5 Max about 148 tokens/s (Qwen3.5-4B, 4-bit), Qualcomm Snapdragon 8 Elite Gen 5 up to 220 tokens/s (3B-class, vendor claim).

## Where the ideas come from

Running a language model for one person is mostly a memory problem. Every token needs every weight, and fetching them from DRAM takes much longer than the math itself. A few companies built their chips around that.

Groq keeps all the weights in SRAM spread over many chips, and its compiler decides every cycle in advance. Taalas turns one model into one chip, with the weights built into the silicon. Both target the datacenter.

izfast applies the same idea to a single local chip: the program is fixed before the chip runs, the weights live next to the multipliers, and the chip is shaped around one model.

## The code

This repository holds the online demo and the first version of the design, which runs Llama 3.2 1B. The Qwen3.5-4B version, with its linear-attention unit, is not public.

The online demo replays answers computed by the bit-exact C emulator, timed with the cycle counts measured on the RTL and the clock from synthesis.

To run the first version you need Verilator, Yosys, sv2v, llama.cpp and Python 3.11 or newer.

    make setup     # python env
    make models    # Llama 3.2 1B weights and GGUF files, about 4 GB
    make test      # RTL against the reference on a small random model
    make check     # Llama 3.2 1B on the RTL against the emulator, about 15 min
    make bench     # cycles per token

Qwen3.5-4B is released by Alibaba Cloud under the Apache 2.0 license. Built with Llama: Llama 3.2 is licensed under the Llama 3.2 Community License.

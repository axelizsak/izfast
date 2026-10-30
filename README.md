# izfast

The goal of this project is to cut the runtime of an LLM down as far as it can go.

izfast is a chip design in SystemVerilog built around one model, Llama 3.2 1B, answering one person at a time. Every weight sits in a memory bank right next to its multipliers, and the chip runs the same fixed program for every token, so nearly all of its cycles go into reading weights and doing the math.

There is no silicon yet. Cycle counts come from simulating the RTL with Verilator, clock speeds from synthesis with Yosys and OpenSTA, before place and route.

Demo: https://axelizsak.github.io/izfast/

Cycles per token at the start of a conversation:

| lanes | cycles per token | tokens/s at 7 nm (607 MHz) |
|---|---|---|
| 256 | 155,591 | 3,899 |
| 1,024 | 42,839 | 14,160 |
| 4,096 | 17,230 | 35,206 |

At 130 nm (SKY130) the clock drops to 81 MHz. Llama 3.2 1B running on the RTL gives exactly the same numbers, bit for bit, as the reference in `hw/ref.py` and the C emulator in `hw/emu.c` that the demo uses.

## Running it

You need Verilator, Yosys, sv2v, llama.cpp and Python 3.11 or newer.

    make setup     # python env
    make models    # Llama 3.2 1B weights and GGUF files, about 4 GB
    make test      # RTL against the reference on a small random model
    make check     # Llama 3.2 1B on the RTL against the emulator, about 15 min
    make bench     # cycles per token
    make synth     # clock speed, needs OpenSTA in ~/eda
    make demo      # ask your own questions and race your own Mac

`make record` refreshes the recorded races used by the online demo.

Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License.

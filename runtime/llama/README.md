# llama.cpp — Windows x64 (Spark-X2.5)

llama.cpp build from the [XHToken/llama.cpp](https://github.com/XHToken/llama.cpp)
fork, with support for the **Spark-X2.5** model (GGUF architecture `spark2_5`).

Backends included in this package: CPU (all x86-64 variants), NVIDIA CUDA 12.4, Vulkan.

## Installation

1. Extract the archive into any folder (e.g. `C:\llama-spark`).
2. For CUDA: extract `cuda-runtime-win-x64-12.4.zip` next to it (or copy all
   `.dll` files from it into the folder with `llama-cli.exe`).
   Without the CUDA runtime, CPU and Vulkan still work (Vulkan only needs a
   graphics driver with Vulkan support).

Windows 10/11 x64 required. The Visual C++ Runtime is usually already present
on Windows 10/11; if a program reports a missing `vcruntime140.dll` — install
the [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

## Checking devices

```bat
llama-cli.exe --list-devices
```

Lists the available devices: `CPU`, `CUDA0: <gpu>`, `Vulkan0: <gpu>`.

## Running

Interactive chat (CUDA):

```bat
llama-cli.exe -m Spark-X2.5-4B.gguf --device CUDA0
```

Interactive chat (Vulkan):

```bat
llama-cli.exe -m Spark-X2.5-4B.gguf --device Vulkan0
```

Server with the OpenAI-compatible API (CUDA, all layers on GPU):

```bat
llama-server.exe -m Spark-X2.5-4B.gguf --device CUDA0 -ngl 99 --port 8080
```

Server (Vulkan):

```bat
llama-server.exe -m Spark-X2.5-4B.gguf --device Vulkan0 -ngl 99 --port 8080
```

Then open the web UI: http://localhost:8080
API: `http://localhost:8080/v1/chat/completions`

Useful options:
- `-ngl N` — number of layers to offload to the GPU (99 = all).
- `-c N` — context size (the model supports up to 1M tokens; for 4B a
  reasonable range is 8192–32768 depending on VRAM).
- `--threads N` — number of CPU threads (default: logical core count).
- `-sm lora`, etc. — see `llama-cli.exe --help`.

## Quantization

The package includes `llama-quantize.exe`. You can download ready-made quants
from HuggingFace (e.g. community repos) or make your own:

```bat
llama-quantize.exe --output Spark-X2.5-4B-Q4_K_M.gguf Spark-X2.5-4B-F16.gguf Q4_K_M
```

## Package contents

| File | Purpose |
|------|---------|
| `llama-cli.exe` | interactive chat / generation |
| `llama-server.exe` | server (OpenAI API + web UI) |
| `llama-app.exe` | unified binary (cli + server + ...) |
| `llama-quantize.exe` | GGUF quantization |
| `llama-bench.exe` | benchmarks |
| `llama-completion.exe`, `llama-embedding.exe`, `llama-perplexity.exe`, ... | utilities |
| `llama.dll`, `ggml*.dll`, `libggml-cpu-*.dll` | core and CPU backends (variants: x64, sse42, sandybridge, haswell, skylakex, cannonlake, cascadelake, icelake, alderlake — the runtime picks the best one) |
| `ggml-cuda.dll` | CUDA backend (requires `cuda-runtime-*.zip`) |
| `ggml-vulkan.dll` | Vulkan backend |

Built from XHToken/llama.cpp @ 4a3635c (XHToken fork, master branch).
Build date: 2026-09-08 09:31 UTC.
Backends: CPU (all x86-64 variants), CUDA 12.4, Vulkan.

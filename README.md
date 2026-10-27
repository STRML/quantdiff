# quantdiff

Diff a quantized or fine-tuned model's weights against the upstream model it claims to come from. A clean quant differs from upstream by rounding noise. A backdoor or an abliteration edit differs by a few low-rank directions, and quantdiff finds those.

It exists because of [How abliterated models can get you pwned](https://projectdiscovery.io/research/how-abliterated-models-can-get-you-pwned). A LoRA-merged backdoor ships as plain weights, no scanner has code to inspect, and the article's advice is to diff against the base model. This does that for MLX quants too.

```
python quantdiff.py --base Qwen/Qwen2.5-0.5B-Instruct \
                    --candidate mlx-community/Qwen2.5-0.5B-Instruct-4bit \
                    [--only REGEX] [--json out.json]
```

Arguments are directories or Hugging Face repo ids (local cache first). Exit 0 is no findings, 1 is flagged, 2 is could not run. Needs `torch`, `safetensors` and `huggingface_hub`; `pip install -r requirements.txt` also adds `numpy` and `pytest`. The tool itself doesn't import numpy, but the tests do (through `safetensors`).

## Check without downloading

```
python quantdiff.py --stream --sample 4 \
    --base Qwen/Qwen2.5-0.5B-Instruct \
    --candidate huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3
```

`--stream` reads repo ids that aren't in your cache with HTTP Range requests: the safetensors header first, then one tensor at a time. Nothing is written to disk, and memory holds the tensors being checked plus `--prefetch-mb` (default 256) fetched ahead. Both the upstream and the candidate stream, so you can vet a quant before you download it. On a terminal it draws a live view on stderr: a byte bar with rate and time left, a tensor bar, and one cell per tensor grouped by layer, red where a tensor is flagged. `--progress off` turns it off, and it stays off when stderr isn't a terminal. The upstream is pinned to a commit hash and the hash is printed.

`--sample K` checks K layers spread through the stack, first and last included. `--layers 5,10,20` picks them yourself, and `--global` adds the embeddings and `lm_head`. A sampled run says so in its output, and it only covers the layers it names.

| Candidate (streamed, `--sample 4`) | Fetched | Time | Result |
|---|---|---|---|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | 153 of 1,266 MB | 50 s | No findings, 48 tensors |
| `huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3` | 239 of 1,976 MB | 49 s | Flagged, 6 of 48: `o_proj` and `down_proj` in layers 8, 15 and 23 (layer 0 is untouched in this model) |

Each request costs about 2 s of latency here, so the prefetch matters: one request at a time took 4m20 for the second row.

## How it decides

```
quantized tensor   dequantize, subtract upstream
                   ├─ any element past the rounding bound? (0.5 step, 1 step on the two edge codes)
                   ├─ error vs a re-quantization of upstream (about 1.0x when clean)
                   └─ share of the error in one direction, vs the same share for that re-quantization
plain tensor       must equal upstream after a dtype cast; if not, how much of the change is one direction
whole model        tensors the candidate adds are flagged, tensors it lacks are listed
```

Rounding noise spreads over many directions. A refusal-direction edit is rank 1.

## Block-scaled upstreams and stacked experts

DeepSeek-V4 ships its weights the way it runs them, so the upstream is decoded before the diff:

| Upstream tensor | Scale beside it | Decoded as |
|---|---|---|
| `F8_E4M3` | `.scale` or `.weight_scale_inv`, one per block (32x32 or 128x128, from the shapes) | FP8 x scale |
| `I8` holding two FP4 E2M1 codes a byte, low nibble first | one E8M0 scale per 32 inputs | FP4 x scale |

Scale tensors are folded into their weight and never listed as missing. Upstream stores one tensor per expert (`experts.7.w1.weight`); a pack that stacks them (`experts.w1.weight`, `[E, out, in]`) matches a virtual stacked tensor, and `--experts K` (default 8) compares K experts spread through the stack, first and last included. The result names them: `layers.3.ffn.experts.w1.weight[experts 0,55,110,...]`.

MLX's microscaling modes are read too: `mxfp8` and `mxfp4` (uint32 codes, a uint8 E8M0 scale per 32, no biases). They're judged on error against a power-of-two re-quantization of upstream and on the one-direction share. An FP8 upstream lands on the mxfp8 grid exactly, so a clean copy has zero error.

## Measured

Qwen2.5-0.5B-Instruct, 2026-10-20:

| Candidate | Result |
|---|---|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | No findings. 0 of 290 tensors flagged, error 1.01x a clean re-quantization. |
| `huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3` (bf16) | Flagged. 46 of 290: `o_proj` and `down_proj` in layers 1-23, each change fully one direction. |
| The abliterated model re-quantized to 4-bit | Flagged. 46 of the 48 edited quantized tensors, 0 false positives. The other 2 sit at a 0.012 one-direction share against 0.011 for a clean quant. |

## What a clean result means

No findings means no tampering this method can see. It is not a safety certificate.

- An edit smaller than the rounding noise and spread over many directions passes.
- The thresholds were set on one real 4-bit MLX quant. Mixed-bit packs, imatrix-tuned quants, GGUF and larger models are untested, so recalibrate before you trust it.
- The upstream must be the exact revision the candidate was made from. A different revision flags everything.
- A sample misses edits in the layers it skips. Four layers caught this abliteration because it touches nearly every layer. A backdoor confined to a few layers needs `--layers` aimed at them or a full run.
- Stacked mixture-of-experts tensors match by name: upstream's fused `experts.gate_up_proj`, or one tensor per expert stacked in order. A repo that reorders the experts flags them.
- With `--experts K`, the experts it skips are unchecked, the same way `--sample` skips layers.
- Safetensors only. NVFP4, MXFP8 and W4A8 upstreams are not read (DeepSeek's FP8 and FP4 are).
- Tokenizer, chat template and config tampering are out of scope.

## Tests

```
python -m pytest -q
```

16 tests on small synthetic models. No network.

MIT.

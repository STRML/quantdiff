# quantdiff

Diff a quantized or fine-tuned model's weights against the upstream model it claims to come from. A clean quant differs from upstream by rounding noise. A backdoor or an abliteration edit differs by a few low-rank directions, and quantdiff finds those.

It exists because of [How abliterated models can get you pwned](https://projectdiscovery.io/research/how-abliterated-models-can-get-you-pwned). A LoRA-merged backdoor ships as plain weights, no scanner has code to inspect, and the article's advice is to diff against the base model. This does that for MLX quants too.

```
python quantdiff.py --base Qwen/Qwen2.5-0.5B-Instruct \
                    --candidate mlx-community/Qwen2.5-0.5B-Instruct-4bit \
                    [--only REGEX] [--json out.json]
```

Arguments are directories or Hugging Face repo ids (local cache first). Exit 0 is no findings, 1 is flagged, 2 is could not run. Needs `torch`, `safetensors` and `huggingface_hub`.

## Check without downloading

```
python quantdiff.py --stream --sample 4 \
    --base Qwen/Qwen2.5-0.5B-Instruct \
    --candidate huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3
```

`--stream` reads repo ids that aren't in your cache with HTTP Range requests: the safetensors header first, then one tensor at a time. Nothing is written to disk, and memory holds the tensors being checked plus `--prefetch-mb` (default 256) fetched ahead. Both the upstream and the candidate stream, so you can vet a quant before you download it. The upstream is pinned to a commit hash and the hash is printed.

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
- Stacked mixture-of-experts tensors match by name only. A repo that re-stacks the experts reports them as missing.
- Safetensors only. NVFP4 and W4A8 files are not read.
- Tokenizer, chat template and config tampering are out of scope.

## Tests

```
python -m pytest -q
```

16 tests on small synthetic models. No network.

MIT.

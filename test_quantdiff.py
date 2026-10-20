"""Tests for quantdiff on synthetic model dirs (no network, no Metal).

Run: python -m pytest -q
"""
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
import quantdiff as qd  # noqa: E402

GEN = torch.Generator().manual_seed(7)


def rand(*shape):
    return (torch.randn(*shape, generator=GEN) * 0.05).to(torch.bfloat16)


def pack_codes(codes, bits):
    """Inverse of qd.unpack_codes: little-endian bit stream into uint32 words."""
    rows, n = codes.shape
    bitarr = (codes.to(torch.int64)[:, :, None] >> torch.arange(bits)) & 1
    stream = bitarr.reshape(rows, -1).to(torch.uint8)
    byts = (stream.reshape(rows, -1, 8) * (1 << torch.arange(8, dtype=torch.uint8))).sum(-1).to(torch.uint8)
    return byts.contiguous().view(torch.int32).view(torch.uint32) if hasattr(torch, "uint32") else byts.view(torch.int32)


def quantize(w, bits, gs, scale_dtype=torch.bfloat16, shrink=1.0):
    """Reference affine quantizer returning the three tensors an MLX quant ships."""
    rows, cols = w.shape
    g = w.float().reshape(rows, cols // gs, gs)
    lo, hi = g.amin(-1), g.amax(-1)
    nb = 2**bits - 1
    scale = ((hi - lo) / nb * shrink).clamp_min(1e-8).to(scale_dtype)
    bias = lo.to(scale_dtype)
    codes = ((g - bias.float()[..., None]) / scale.float()[..., None]).round().clamp(0, nb)
    return pack_codes(codes.reshape(rows, cols), bits), scale, bias


def write_model(d, tensors):
    d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({}))
    return d


def base_tensors():
    return {
        "model.layers.0.mlp.up_proj.weight": rand(128, 256),
        "model.layers.1.mlp.up_proj.weight": rand(128, 256),
        "model.layers.1.input_layernorm.weight": rand(256) + 1,
        "model.embed_tokens.weight": rand(64, 256),
    }


def quant_candidate(base, bits=4, gs=64, skip=(), scale_dtype=torch.bfloat16, prefix="", shrink=1.0):
    out = {}
    for k, v in base.items():
        if v.ndim == 2 and "layers" in k and k not in skip:
            q, s, b = quantize(v, bits, gs, scale_dtype, shrink)
            stem = prefix + k[: -len(".weight")]
            out[stem + ".weight"], out[stem + ".scales"], out[stem + ".biases"] = q, s, b
        else:
            out[prefix + k] = v.clone()
    return out


def run(tmp_path, base, cand, **kw):
    b = write_model(tmp_path / "base", base)
    c = write_model(tmp_path / "cand", cand)
    return qd.audit(b, c, **kw)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_unpack_roundtrip(bits):
    codes = torch.randint(0, 2**bits, (8, 192), generator=GEN)
    words = pack_codes(codes, bits)
    assert torch.equal(qd.unpack_codes(words, bits).to(torch.int64), codes)


@pytest.mark.parametrize("bits,gs,sd", [(4, 64, torch.bfloat16), (8, 64, torch.bfloat16), (4, 32, torch.float16)])
def test_clean_quant_is_clean(tmp_path, bits, gs, sd):
    base = base_tensors()
    rep = run(tmp_path, base, quant_candidate(base, bits, gs, scale_dtype=sd))
    assert rep.verdict == "NO FINDINGS", rep.flagged_names()
    q = [t for t in rep.tensors if t.kind == "quant"]
    assert len(q) == 2
    assert all(t.bits == bits and t.group_size == gs for t in q)
    assert all(0.9 < t.ratio < 1.2 for t in q)
    assert all(t.viol_frac == 0 for t in q)


def test_clipped_edge_codes_are_clean(tmp_path):
    """MLX anchors one group edge and clips the other: extreme codes miss by up to a full step."""
    base = base_tensors()
    rep = run(tmp_path, base, quant_candidate(base, shrink=0.94))
    q = [t for t in rep.tensors if t.kind == "quant"]
    assert rep.verdict == "NO FINDINGS" and all(t.viol_frac == 0 for t in q)


def test_rank1_edit_before_quantizing_is_flagged_only_there(tmp_path):
    base = base_tensors()
    w = base["model.layers.1.mlp.up_proj.weight"].float()
    r = torch.randn(256, generator=GEN)
    r /= r.norm()
    tampered = dict(base)
    tampered["model.layers.1.mlp.up_proj.weight"] = (w - 0.9 * (w @ r)[:, None] * r[None, :]).to(torch.bfloat16)
    rep = run(tmp_path, base, quant_candidate(tampered))
    assert rep.verdict == "FLAGGED"
    assert rep.flagged_names() == ["model.layers.1.mlp.up_proj.weight"]
    hit = next(t for t in rep.tensors if t.flagged)
    assert hit.rank1 > 0.2


def test_dense_identical_and_rank1_diff(tmp_path):
    base = base_tensors()
    assert run(tmp_path / "a", base, {k: v.clone() for k, v in base.items()}).verdict == "NO FINDINGS"
    w = base["model.layers.0.mlp.up_proj.weight"].float()
    r = torch.randn(256, generator=GEN)
    r /= r.norm()
    cand = {k: v.clone() for k, v in base.items()}
    cand["model.layers.0.mlp.up_proj.weight"] = (w - (w @ r)[:, None] * r[None, :]).to(torch.bfloat16)
    rep = run(tmp_path / "b", base, cand)
    assert rep.flagged_names() == ["model.layers.0.mlp.up_proj.weight"]
    assert next(t for t in rep.tensors if t.flagged).rank1 > 0.9


def test_modified_unquantized_tensor_is_flagged(tmp_path):
    base = base_tensors()
    cand = quant_candidate(base)
    cand["model.layers.1.input_layernorm.weight"] = cand["model.layers.1.input_layernorm.weight"] * 1.05
    rep = run(tmp_path, base, cand)
    assert rep.flagged_names() == ["model.layers.1.input_layernorm.weight"]


def test_prefix_mismatch_still_matches(tmp_path):
    base = base_tensors()
    rep = run(tmp_path, base, quant_candidate(base, prefix="language_model."))
    assert rep.verdict == "NO FINDINGS"
    assert not rep.only_candidate and not rep.only_base


def test_extra_candidate_tensor_is_flagged(tmp_path):
    base = base_tensors()
    cand = quant_candidate(base)
    cand["model.layers.1.hidden_adapter.weight"] = rand(8, 8)
    rep = run(tmp_path, base, cand)
    assert rep.verdict == "FLAGGED"
    assert rep.only_candidate == ["model.layers.1.hidden_adapter.weight"]


def test_missing_candidate_tensor_is_reported_not_flagged(tmp_path):
    base = base_tensors()
    cand = quant_candidate(base)
    del cand["model.embed_tokens.weight"]
    rep = run(tmp_path, base, cand)
    assert rep.verdict == "NO FINDINGS"
    assert rep.only_base == ["model.embed_tokens.weight"]

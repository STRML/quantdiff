#!/usr/bin/env python3
"""Diff a quantized (or fine-tuned, or abliterated) model against its upstream weights.

    quantdiff.py --base <upstream dir|hf repo> --candidate <dir|hf repo> [--stream] [--sample K | --layers N,N] [--json out.json] [--only REGEX]

Defends against weight-level tampering (https://projectdiscovery.io/research/how-abliterated-models-can-get-you-pwned):
a LoRA-merged backdoor or an abliteration edit changes a few tensors by a structured, low-rank amount. A clean
quant differs from upstream only by rounding noise, which has two properties the checks use.

  quantized tensors   Dequantize and subtract upstream. Rounding error is bounded by half a quant step per
                      element, so any element past that bound (`viol`) cannot come from quantizing. The
                      error norm is also compared with a re-quantization of upstream at the same bits and
                      group size (`ratio`, about 1.0 when clean). Rounding noise is spread over every
                      direction, while an edit concentrates in a few: `rank1` is the share of the error's
                      energy in its top singular direction.
  plain tensors       Norms, embeddings and anything the quantizer skipped must match upstream (after a
                      dtype cast). Any difference is a modification. Dense-vs-dense diffs (a fine-tune or
                      an abliterated bf16 release) take this path too, and `rank1` shows the edit shape.

--stream reads a repo id that is not in the local cache with HTTP Range requests: the safetensors header, then one
tensor at a time. Nothing is written to disk and memory holds one tensor pair. --sample K checks K layers spread
through the stack and says so in the verdict, which makes a first pass cheap.

Exit status: 0 no findings, 1 flagged, 2 could not run. Needs torch and safetensors; huggingface_hub to resolve repo ids.
NO FINDINGS means no sign of tampering that this method can see. It is not a safety certificate.
Tensors only: tokenizer, chat template and config tampering are not covered.
"""
import argparse
import http.client
import json
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

CAST_TOL = 2.0**-8  # relative error a dtype cast (bf16 <-> f16 <-> f32) may introduce
VIOL_FRAC = 1e-4  # share of elements past the rounding bound that flags a quantized tensor
RATIO_MAX = 1.5  # error vs re-quantized upstream
RANK1_EXCESS = 0.03  # share of a quantized tensor's error in one direction beyond what re-quantizing upstream gives (clean quants: <= 0.01)
SVD_MAX_ELEMS = 3e8
SVD_SLICES = 8  # 3-D (stacked expert) tensors: slices sampled for the rank-1 check


@dataclass
class TensorResult:
    name: str
    kind: str  # quant | plain
    shape: list
    rel_err: float = 0.0
    bits: int = 0
    group_size: int = 0
    ratio: float = 0.0
    viol_frac: float = 0.0
    rank1: float = 0.0
    rank1_ref: float = 0.0
    reasons: list = field(default_factory=list)

    @property
    def flagged(self):
        return bool(self.reasons)


@dataclass
class Report:
    tensors: list
    only_candidate: list
    only_base: list
    layers: list = None  # layers checked when sampling; None means every tensor
    fetched: int = 0  # bytes streamed over HTTP (base and candidate); nothing is written to disk
    remote_total: int = 0  # bytes in the streamed repos

    @property
    def verdict(self):
        return "FLAGGED" if self.flagged_names() or self.only_candidate else "NO FINDINGS"

    def flagged_names(self):
        return [t.name for t in self.tensors if t.flagged]


class Store:
    """Lazy tensor access over a directory of safetensors shards."""

    def __init__(self, root):
        self.root = Path(root)
        files = sorted(self.root.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no *.safetensors in {root}")
        self.where = {}
        self.handles = {}
        for f in files:
            with safe_open(str(f), framework="pt") as h:
                for k in h.keys():
                    self.where[k] = f
        self.names = set(self.where)

    def get(self, name):
        f = self.where[name]
        if f not in self.handles:
            self.handles[f] = safe_open(str(f), framework="pt")
        return self.handles[f].get_tensor(name)


SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
    "I8": torch.int8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "U8": torch.uint8, "U16": torch.uint16, "U32": torch.uint32, "U64": torch.uint64, "BOOL": torch.bool,
}


class _DropAuthOffHost(urllib.request.HTTPRedirectHandler):
    """Hub downloads redirect to a CDN; the bearer token must not follow them to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_DropAuthOffHost)


class RemoteStore:
    """Tensor access over HTTP Range requests. Reads each shard's safetensors header, then fetches one
    tensor at a time, so nothing touches disk and memory holds one tensor."""

    def __init__(self, files, url_for, headers=None):
        self.headers = dict(headers or {})
        self.fetched = 0
        self.total_bytes = 0
        self.targets = {}
        self.lock = threading.Lock()
        self.index = {}
        for f in files:
            url = url_for(f)
            (n,) = struct.unpack("<Q", self._range(url, 0, 7))
            if n > 1 << 28:
                raise RuntimeError(f"{url}: safetensors header claims {n} bytes")
            header, base = json.loads(self._range(url, 8, 7 + n)), 8 + n
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                start, end = meta["data_offsets"]
                self.index[name] = (url, base + start, base + end, SAFETENSORS_DTYPES[meta["dtype"]], meta["shape"])
                self.total_bytes += end - start
        if not self.index:
            raise FileNotFoundError("no safetensors tensors found")
        self.names = set(self.index)

    def _open(self, url, first, last):
        """Request first..last of `url`, following the Hub's redirect to its CDN once per file and reusing that
        target afterwards. The token goes only to the host the caller named."""
        target = self.targets.get(url, url)
        headers = dict(self.headers)
        if urllib.parse.urlparse(target).netloc != urllib.parse.urlparse(url).netloc:
            headers.pop("Authorization", None)
        req = urllib.request.Request(target, headers={**headers, "Range": f"bytes={first}-{last}"})
        try:
            r = _OPENER.open(req, timeout=60)
        except urllib.error.HTTPError as e:
            if target != url and e.code in (400, 401, 403):  # the signed CDN link expired
                self.targets.pop(url, None)
                return self._open(url, first, last)
            raise
        self.targets[url] = r.geturl()
        return r

    def _range(self, url, first, last):
        """Bytes first..last inclusive. A connection cut mid-body keeps what arrived and asks for the rest."""
        want, buf, failure = last - first + 1, bytearray(), None
        for attempt in range(5):
            try:
                with self._open(url, first + len(buf), last) as r:
                    if r.status != 206:
                        raise RuntimeError(f"{url} ignored the Range header (HTTP {r.status}); not downloading the whole file")
                    buf += r.read()
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    raise RuntimeError(f"{url}: HTTP {e.code}") from e
                failure = e
            except http.client.IncompleteRead as e:
                buf += e.partial
                failure = e
            except (OSError, http.client.HTTPException) as e:
                failure = e
            if len(buf) == want:
                with self.lock:
                    self.fetched += want
                return bytes(buf)
            if len(buf) > want:
                raise RuntimeError(f"{url}: asked for {want} bytes, got {len(buf)}")
            time.sleep(1 + attempt)
        raise RuntimeError(f"{url}: {failure}")

    def size(self, name):
        return self.index[name][2] - self.index[name][1]

    def get(self, name):
        url, start, end, dtype, shape = self.index[name]
        if end == start:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(bytearray(self._range(url, start, end - 1)), dtype=dtype).reshape(shape)


def cached_dir(spec):
    from huggingface_hub import snapshot_download

    try:
        d = Path(snapshot_download(spec, allow_patterns=["*.safetensors", "*.json"], local_files_only=True))
    except Exception:  # noqa: BLE001 - not in the cache
        return None
    return d if any(d.glob("*.safetensors")) else None


def open_store(spec, stream=False, revision=None):
    """A directory, else the local HF cache, else (stream) Range requests, else a full download."""
    p = Path(spec).expanduser()
    if p.is_dir():
        return Store(p)
    d = cached_dir(spec) if revision is None else None
    if d:
        return Store(d)
    from huggingface_hub import HfApi, get_token, hf_hub_url, snapshot_download

    if not stream:
        return Store(Path(snapshot_download(spec, allow_patterns=["*.safetensors", "*.json"], revision=revision)))
    api = HfApi()
    sha = api.model_info(spec, revision=revision).sha
    files = sorted(f for f in api.list_repo_files(spec, revision=sha) if f.endswith(".safetensors"))
    token = get_token()
    store = RemoteStore(files, lambda f: hf_hub_url(spec, f, revision=sha), {"Authorization": f"Bearer {token}"} if token else None)
    store.revision = sha
    return store


def pick_layers(layers, k):
    """k layers spread evenly across the stack, first and last included."""
    layers = sorted(layers)
    if k >= len(layers):
        return layers
    if k == 1:
        return layers[-1:]
    n = len(layers)
    return [layers[i] for i in sorted({round(j * (n - 1) / (k - 1)) for j in range(k)})]


def suffix_index(names):
    """Every dotted suffix of >= 2 segments that identifies exactly one name."""
    seen = {}
    for n in names:
        segs = n.split(".")
        for i in range(len(segs) - 1):
            seen.setdefault(".".join(segs[i:]), []).append(n)
    return {s: v[0] for s, v in seen.items() if len(v) == 1}


def match_name(name, base_names, index):
    if name in base_names:
        return name
    segs = name.split(".")
    for i in range(len(segs) - 1):
        hit = index.get(".".join(segs[i:]))
        if hit:
            return hit
    return None


def unpack_codes(words, bits):
    """MLX packs `bits`-wide codes little-endian into uint32 words. Returns int codes [..., n]."""
    b = words.contiguous().view(torch.uint8)
    if bits == 8:
        return b
    if bits == 4:
        return torch.stack([b & 0xF, b >> 4], -1).reshape(*b.shape[:-1], -1)
    if bits == 2:
        return torch.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], -1).reshape(*b.shape[:-1], -1)
    flat = b.reshape(-1, b.shape[-1])
    shifts = torch.arange(8, dtype=torch.uint8)
    weights = 1 << torch.arange(bits, dtype=torch.int16)
    out = []
    for i in range(0, flat.shape[0], 4096):
        bitarr = (flat[i : i + 4096, :, None] >> shifts) & 1
        out.append((bitarr.reshape(bitarr.shape[0], -1, bits).to(torch.int16) * weights).sum(-1))
    return torch.cat(out).reshape(*b.shape[:-1], -1)


def dequantize(words, scales, biases, bits, gs):
    codes = unpack_codes(words, bits).float()
    g = codes.reshape(*codes.shape[:-1], -1, gs)
    return (g * scales.float()[..., None] + biases.float()[..., None]).reshape(codes.shape)


def requantize(w, bits, gs):
    """Min/max affine round trip of upstream; the noise floor a clean quant should sit near."""
    g = w.float().reshape(*w.shape[:-1], -1, gs)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = ((hi - lo) / (2**bits - 1)).clamp_min(1e-12)
    return (((g - lo) / scale).round() * scale + lo).reshape(w.shape)


def rel(delta, ref):
    return float(delta.double().pow(2).sum().sqrt() / ref.double().pow(2).sum().sqrt().clamp_min(1e-30))


def rank1_share(delta):
    """Share of the error's energy in its top singular direction (max over sampled 2-D slices)."""
    if delta.numel() > SVD_MAX_ELEMS or delta.ndim < 2:
        return 0.0
    mats = delta.reshape(-1, *delta.shape[-2:])
    step = max(1, len(mats) // SVD_SLICES)
    best = 0.0
    for m in mats[::step][:SVD_SLICES]:
        total = float(m.double().pow(2).sum())
        if total == 0:
            continue
        s = torch.svd_lowrank(m.float(), q=min(4, *m.shape), niter=3)[1]
        best = max(best, float(s[0].double() ** 2) / total)
    return best


def check_quant(name, base_w, words, scales, biases):
    in_dim = base_w.shape[-1]
    gs = in_dim // scales.shape[-1]
    bits = words.shape[-1] * 32 // in_dim
    res = TensorResult(name, "quant", list(base_w.shape), bits=bits, group_size=gs)
    if bits not in (2, 3, 4, 5, 6, 8) or gs * scales.shape[-1] != in_dim:
        res.reasons.append(f"layout does not decode (bits={bits}, group_size={gs})")
        return res
    codes = unpack_codes(words, bits)
    deq = dequantize(words, scales, biases, bits, gs)
    delta = deq - base_w.float()
    res.rel_err = rel(delta, base_w)
    ref_delta = requantize(base_w, bits, gs) - base_w.float()
    floor = rel(ref_delta, base_w)
    res.ratio = res.rel_err / floor if floor else float("inf")
    # Half a step for interior codes. MLX anchors one edge of a group exactly and clips the other, so the two
    # extreme codes may be off by a full step. The stored scale and bias rounding adds a little (bf16 keeps 8 bits).
    n_bins = 2**bits - 1
    s = scales.float().abs()[..., None]
    g = codes.reshape(*codes.shape[:-1], -1, gs)
    reach = torch.where((g == 0) | (g == n_bins), 1.0, 0.5)
    tol = s * (reach + n_bins / 256) + biases.float().abs()[..., None] / 256 + 1e-12
    res.viol_frac = float((delta.reshape(g.shape).abs() > tol).double().mean())
    res.rank1, res.rank1_ref = rank1_share(delta), rank1_share(ref_delta)
    if res.viol_frac > VIOL_FRAC:
        res.reasons.append(f"{res.viol_frac:.2%} of elements past half a quant step")
    if res.ratio > RATIO_MAX:
        res.reasons.append(f"error {res.ratio:.1f}x a clean re-quantization")
    if res.rank1 - res.rank1_ref > RANK1_EXCESS:
        res.reasons.append(f"{res.rank1:.0%} of the error is one direction, {res.rank1_ref:.0%} for a clean quant")
    return res


def check_plain(name, base_w, cand_w):
    res = TensorResult(name, "plain", list(base_w.shape))
    if base_w.shape != cand_w.shape:
        res.reasons.append(f"shape {list(cand_w.shape)} against upstream {list(base_w.shape)}")
        return res
    if torch.equal(base_w, cand_w):
        return res
    delta = cand_w.float() - base_w.float()
    res.rel_err = rel(delta, base_w)
    res.rank1 = rank1_share(delta)
    if res.rel_err > CAST_TOL:
        res.reasons.append(f"differs from upstream by {res.rel_err:.2%}")
        if res.rank1 > 0.5:
            res.reasons.append(f"{res.rank1:.0%} of the change is one direction")
    return res


def stream_units(units, budget, workers=8):
    """Yield (unit, tensors) in order. Remote tensors are fetched ahead on a thread pool, at most `budget` bytes
    beyond the unit being checked (one request costs ~2 s of latency, so one at a time is slow)."""
    size = lambda u: sum(s.size(n) for s, n in u[1] if hasattr(s, "size"))  # noqa: E731
    if not any(hasattr(s, "size") for u in units[:1] for s, _ in u[1]):
        for u in units:
            yield u, [s.get(n) for s, n in u[1]]
        return
    with ThreadPoolExecutor(workers) as pool:
        pending, nxt, held = {}, 0, 0
        for i, u in enumerate(units):
            while nxt < len(units) and (nxt <= i or held + size(units[nxt]) <= budget):
                pending[nxt] = [pool.submit(s.get, n) for s, n in units[nxt][1]]
                held += size(units[nxt])
                nxt += 1
            tensors = [f.result() for f in pending.pop(i)]
            held -= size(u)
            yield u, tensors


def audit(base, cand, only=None, layers=None, include_global=False, prefetch_bytes=256 << 20):
    """`base` and `cand` are stores or directories. `layers` limits the check to those layer indices."""
    base = base if hasattr(base, "get") else Store(base)
    cand = cand if hasattr(cand, "get") else Store(cand)

    def in_scope(name):
        if layers is None:
            return True
        layer = layer_of(name)
        return include_global if layer is None else layer in layers

    index = suffix_index(base.names)
    quant_stems = {n[: -len(".scales")] for n in cand.names if n.endswith(".scales") and n[: -len(".scales")] + ".weight" in cand.names}
    skip = {s + suffix for s in quant_stems for suffix in (".scales", ".biases")}
    pat = re.compile(only) if only else None
    units, only_cand, matched = [], [], set()
    for name in sorted(cand.names - skip):
        stem = name[: -len(".weight")] if name.endswith(".weight") else None
        target = match_name(name, base.names, index)
        if target is None:
            only_cand.append(name)
            continue
        matched.add(target)
        if not in_scope(target) or (pat and not pat.search(target)):
            continue
        fetch = [(base, target), (cand, name)]
        if stem in quant_stems:
            fetch += [(cand, stem + ".scales"), (cand, stem + ".biases")]
        units.append((target, fetch))
    results = []
    for (target, fetch), tensors in stream_units(units, prefetch_bytes):
        results.append(check_quant(target, *tensors) if len(tensors) == 4 else check_plain(target, *tensors))
    return Report(results, only_cand, sorted(n for n in base.names - matched if in_scope(n)),
                  None if layers is None else sorted(layers),
                  sum(getattr(s, "fetched", 0) for s in (base, cand)),
                  sum(getattr(s, "total_bytes", 0) for s in (base, cand)))


def layer_of(name):
    m = re.search(r"(?:^|\.)(?:layers|blocks|h)\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def render(rep, top=15):
    out = []
    flagged = sorted((t for t in rep.tensors if t.flagged), key=lambda t: -(t.ratio or t.rel_err))
    q = [t for t in rep.tensors if t.kind == "quant"]
    out.append(f"{rep.verdict}: {len(flagged)} of {len(rep.tensors)} tensors flagged, "
               f"{len(rep.only_candidate)} not in upstream, {len(rep.only_base)} missing from candidate")
    if rep.layers is not None:
        out.append(f"sampled layers {rep.layers} only; every other layer is unchecked, so this is not a full result")
    if rep.remote_total:
        out.append(f"streamed {rep.fetched / 1e6:,.0f} MB of {rep.remote_total / 1e6:,.0f} MB, nothing written to disk")
    if q:
        bits = sorted({(t.bits, t.group_size) for t in q})
        out.append(f"quantized: {len(q)} tensors, (bits, group) {bits}, median error ratio "
                   f"{sorted(t.ratio for t in q)[len(q) // 2]:.2f}x a clean re-quantization")
    by_layer = {}
    for t in rep.tensors:
        by_layer.setdefault(layer_of(t.name), []).append(t)
    layers = sorted(k for k in by_layer if k is not None)
    if layers:
        out.append("\nflagged tensors per layer")
        for k in layers:
            n = sum(t.flagged for t in by_layer[k])
            out.append(f"  {k:>3} {'█' * n}{'░' * (len(by_layer[k]) - n)} {n}/{len(by_layer[k])}")
    if flagged:
        out.append("\nworst tensors")
        for t in flagged[:top]:
            out.append(f"  ✗ {t.name}  {'; '.join(t.reasons)}")
    for label, names in (("not in upstream", rep.only_candidate), ("missing from candidate", rep.only_base)):
        if names:
            out.append(f"\n{label} ({len(names)}):")
            out += [f"  {n}" for n in names[:top]]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="upstream model dir or HF repo id")
    ap.add_argument("--candidate", required=True, help="quantized / fine-tuned model dir or HF repo id")
    ap.add_argument("--stream", action="store_true", help="repo ids not in the local cache are read with HTTP Range requests, one tensor at a time, instead of downloaded")
    ap.add_argument("--base-revision", help="upstream branch, tag or commit (printed as a commit hash when streaming)")
    ap.add_argument("--candidate-revision", help="candidate branch, tag or commit")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--sample", type=int, metavar="K", help="check K layers spread evenly through the stack (first and last included)")
    sel.add_argument("--layers", metavar="N,N,...", help="check exactly these layer indices")
    ap.add_argument("--prefetch-mb", type=int, default=256, help="streaming: megabytes of tensors fetched ahead of the one being checked (memory ceiling is about this plus the largest tensor)")
    ap.add_argument("--global", dest="include_global", action="store_true", help="with --sample/--layers, also check tensors outside the layers (embeddings, lm_head, final norm)")
    ap.add_argument("--only", help="regex on upstream tensor names; others are skipped")
    ap.add_argument("--json", help="write the full per-tensor report here")
    args = ap.parse_args(argv)
    try:
        base = open_store(args.base, args.stream, args.base_revision)
        cand = open_store(args.candidate, args.stream, args.candidate_revision)
        for label, st in (("upstream", base), ("candidate", cand)):
            if hasattr(st, "revision"):
                print(f"{label} pinned at {st.revision}")
        layers = None
        if args.layers:
            layers = {int(x) for x in args.layers.split(",")}
        elif args.sample:
            layers = set(pick_layers({layer_of(n) for n in base.names} - {None}, args.sample))
        rep = audit(base, cand, args.only, layers, args.include_global, args.prefetch_mb << 20)
    except Exception as e:  # noqa: BLE001
        print(f"quantdiff: {e}", file=sys.stderr)
        return 2
    print(render(rep))
    if args.json:
        Path(args.json).write_text(json.dumps({"verdict": rep.verdict, "only_candidate": rep.only_candidate,
                                               "only_base": rep.only_base,
                                               "layers": rep.layers,
                                               "tensors": [asdict(t) for t in rep.tensors]}, indent=1))
    return 0 if rep.verdict == "NO FINDINGS" else 1


if __name__ == "__main__":
    sys.exit(main())

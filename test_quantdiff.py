"""Tests for quantdiff on synthetic model dirs (no network, no Metal).

Run: python -m pytest -q
"""
import json
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


class RangeHandler(BaseHTTPRequestHandler):
    """Serves files under `root` with HTTP Range support; `honor_range=False` mimics a server that ignores it."""

    def __init__(self, *a, root, honor_range=True, **kw):
        self.root, self.honor_range = Path(root), honor_range
        super().__init__(*a, **kw)

    def log_message(self, *a):
        pass

    def do_GET(self):
        data = (self.root / self.path.lstrip("/")).read_bytes()
        rng = self.headers.get("Range")
        if rng and self.honor_range:
            a, b = rng.split("=")[1].split("-")
            part = data[int(a) : int(b) + 1]
            self.send_response(206)
        else:
            part = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)


@pytest.fixture
def serve():
    servers = []

    def start(root, honor_range=True):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, root=root, honor_range=honor_range))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}"

    yield start
    for s in servers:
        s.shutdown()


def remote(url):
    return qd.RemoteStore(["model.safetensors"], lambda f: f"{url}/{f}")


def test_remote_store_matches_local_store(tmp_path, serve):
    base = base_tensors()
    d = write_model(tmp_path / "base", base)
    local, rem = qd.Store(d), remote(serve(d))
    assert rem.names == local.names
    for n in sorted(local.names):
        assert torch.equal(rem.get(n), local.get(n)), n


def test_remote_audit_matches_local_audit(tmp_path, serve):
    base = base_tensors()
    w = base["model.layers.1.mlp.up_proj.weight"].float()
    r = torch.randn(256, generator=GEN)
    r /= r.norm()
    tampered = dict(base)
    tampered["model.layers.1.mlp.up_proj.weight"] = (w - 0.9 * (w @ r)[:, None] * r[None, :]).to(torch.bfloat16)
    b = write_model(tmp_path / "base", base)
    c = write_model(tmp_path / "cand", quant_candidate(tampered))
    url = serve(tmp_path)
    got = qd.audit(remote(f"{url}/base"), remote(f"{url}/cand"))
    want = qd.audit(b, c)
    assert got.verdict == want.verdict == "FLAGGED"
    assert got.flagged_names() == want.flagged_names()


def test_server_ignoring_range_is_an_error_not_a_full_download(tmp_path, serve):
    d = write_model(tmp_path / "base", base_tensors())
    with pytest.raises(RuntimeError, match="Range"):
        remote(serve(d, honor_range=False))


def test_sampled_layers_fetch_fewer_bytes_and_say_so(tmp_path, serve):
    base = base_tensors()
    w = base["model.layers.1.mlp.up_proj.weight"].float()
    r = torch.randn(256, generator=GEN)
    r /= r.norm()
    tampered = dict(base)
    tampered["model.layers.1.mlp.up_proj.weight"] = (w - 0.9 * (w @ r)[:, None] * r[None, :]).to(torch.bfloat16)
    write_model(tmp_path / "base", base)
    write_model(tmp_path / "cand", quant_candidate(tampered))
    url = serve(tmp_path)
    full_base = remote(f"{url}/base")
    hit = qd.audit(full_base, remote(f"{url}/cand"), layers={1})
    assert hit.flagged_names() == ["model.layers.1.mlp.up_proj.weight"]
    assert hit.layers == [1]
    miss_base = remote(f"{url}/base")
    miss = qd.audit(miss_base, remote(f"{url}/cand"), layers={0})
    assert miss.verdict == "NO FINDINGS" and miss.layers == [0]
    assert [t.name for t in miss.tensors] == ["model.layers.0.mlp.up_proj.weight"]
    assert miss_base.fetched < full_base.fetched
    assert not miss.only_base  # tensors outside the sample are not "missing"


def test_pick_layers_spreads_evenly_and_keeps_the_ends():
    assert qd.pick_layers(list(range(24)), 4) == [0, 8, 15, 23]
    assert qd.pick_layers(list(range(3)), 8) == [0, 1, 2]
    assert qd.pick_layers([], 4) == []


def test_connection_cut_mid_body_resumes_from_the_bytes_that_arrived(tmp_path, serve):
    d = write_model(tmp_path / "base", base_tensors())
    cut = []

    class Cutting(RangeHandler):
        def do_GET(self):
            data = (self.root / self.path.lstrip("/")).read_bytes()
            a, b = self.headers["Range"].split("=")[1].split("-")
            part = data[int(a) : int(b) + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(part)))
            self.end_headers()
            if len(part) > 1000 and not cut:
                cut.append(1)
                self.wfile.write(part[: len(part) - 7])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(part)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Cutting, root=d))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rem = qd.RemoteStore(["model.safetensors"], lambda f: f"http://127.0.0.1:{srv.server_port}/{f}")
        want = qd.Store(d)
        for n in sorted(want.names):
            assert torch.equal(rem.get(n), want.get(n)), n
        assert cut
    finally:
        srv.shutdown()


def test_redirect_is_followed_once_and_the_token_stays_on_the_first_host(tmp_path, serve):
    d = write_model(tmp_path / "base", base_tensors())
    seen = {"hub": [], "cdn": []}

    class Hub(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen["hub"].append(self.headers.get("Authorization"))
            self.send_response(302)
            self.send_header("Location", f"{cdn}{self.path}")
            self.end_headers()

    class Spy(RangeHandler):
        def do_GET(self):
            seen["cdn"].append(self.headers.get("Authorization"))
            super().do_GET()

    spy = ThreadingHTTPServer(("127.0.0.1", 0), partial(Spy, root=d))
    cdn = f"http://127.0.0.1:{spy.server_port}"
    hub = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    for srv in (spy, hub):
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rem = qd.RemoteStore(["model.safetensors"], lambda f: f"http://127.0.0.1:{hub.server_port}/{f}", {"Authorization": "Bearer t"})
        want = qd.Store(d)
        for n in sorted(want.names):
            assert torch.equal(rem.get(n), want.get(n)), n
    finally:
        spy.shutdown()
        hub.shutdown()
    assert seen["hub"] == ["Bearer t"]  # one redirect for the whole file
    assert set(seen["cdn"]) == {None} and len(seen["cdn"]) > 1


import io  # noqa: E402


def make_progress(width=80, color=False):
    return qd.Progress(out=io.StringIO(), color=color, width=width, labels=("Qwen/base", "me/quant"))


def test_progress_shows_every_tensor_by_layer_with_status_glyphs():
    p = make_progress()
    units = [(f"model.layers.{l}.mlp.{n}.weight", []) for l in (0, 1) for n in ("down_proj", "up_proj", "gate_proj")]
    p.begin(units, (), {0, 1})
    p.status[:] = ["ok", "flagged", "ok", "fetching", "queued", "queued"]
    p.latest = ("flagged", "model.layers.0.mlp.up_proj.weight", "100% of the change is one direction")
    text = "\n".join(p.lines())
    assert text.count("■") == 2 + 1  # two ok cells, plus the legend entry
    assert text.count("✗") == 1 + 1 + 1  # flagged cell, the latest line, the legend entry
    assert "◐" in text and "·" in text
    assert "tensors" in text and "3 / 6" in text  # done = clean + flagged
    assert "Qwen/base" in text and "me/quant" in text
    assert "\x1b" not in text


def test_progress_lines_never_exceed_the_terminal_width():
    p = make_progress(width=40)
    units = [(f"model.layers.{l}.self_attn.q_proj.weight", []) for l in range(30)]
    p.begin(units, (), set(range(30)))
    p.latest = ("ok", "model.layers.29.self_attn.q_proj.weight" * 3, "")
    assert all(len(line) <= 39 for line in p.lines())


def test_progress_redraws_in_place_and_ends_on_a_final_frame():
    p = make_progress(color=True)
    p.begin([("model.layers.0.a.weight", [])], (), {0})
    p.draw(force=True)
    p.draw(force=True)
    out = p.out.getvalue()
    assert out.count("\x1b[") > 0 and "F" in out  # second frame moves the cursor back up first


def test_audit_drives_the_progress_view(tmp_path):
    base = base_tensors()
    w = base["model.layers.1.mlp.up_proj.weight"].float()
    r = torch.randn(256, generator=GEN)
    r /= r.norm()
    tampered = dict(base)
    tampered["model.layers.1.mlp.up_proj.weight"] = (w - 0.9 * (w @ r)[:, None] * r[None, :]).to(torch.bfloat16)
    b = write_model(tmp_path / "base", base)
    c = write_model(tmp_path / "cand", quant_candidate(tampered))
    p = make_progress()
    rep = qd.audit(b, c, progress=p)
    final = "\n".join(p.lines())
    assert rep.verdict == "FLAGGED"
    n = len(rep.tensors)
    assert f"{n} / {n}" in final
    assert "✗ model.layers.1.mlp.up_proj.weight" in final  # last tensor checked, and it is the flagged one
    assert final.count("✗") == 3  # its cell, the latest line, the legend


def test_headers_of_many_shards_load_in_parallel_and_report_progress(tmp_path, serve):
    d = tmp_path / "base"
    d.mkdir()
    tensors = base_tensors()
    items = sorted(tensors.items())
    for i, (k, v) in enumerate(items):
        save_file({k: v}, str(d / f"model-0000{i}-of-0000{len(items)}.safetensors"))
    files = sorted(p.name for p in d.glob("*.safetensors"))
    seen = []
    url = serve(d)
    rem = qd.RemoteStore(files, lambda f: f"{url}/{f}", on_progress=lambda done, total: seen.append((done, total)))
    assert rem.names == set(tensors)
    assert seen[-1] == (len(files), len(files)) and [x[0] for x in seen] == list(range(1, len(files) + 1))

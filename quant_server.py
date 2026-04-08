#!/usr/bin/env python3
"""Thin FastAPI server for the tensor viewer. All heavy logic lives in gpu_quant."""

import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

import sys
_here = Path(__file__).resolve().parent
_parent = _here.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
import gpu_quant as Q
app = FastAPI()

# ── Model cache ───────────────────────────────────────────────────────
_cache = {}  # path -> {"src": source_or_None, "metas": [TensorMeta], "fmt": str}


def _open_remote(path):
    """Open an HF repo or remote URL, caching shard metadata."""
    import requests as http_requests

    repo_id = Q.extract_repo_id(path) if hasattr(Q, "extract_repo_id") else None
    if repo_id:
        url = f"https://huggingface.co/api/models/{repo_id}"
        siblings = http_requests.get(url, timeout=60).json().get("siblings", [])
        st_files = sorted(s["rfilename"] for s in siblings if s["rfilename"].endswith(".safetensors"))
        if not st_files:
            raise ValueError(f"No .safetensors in repo {repo_id}")

        # Try index for lightweight tensor list
        idx_url = f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors.index.json"
        weight_map = {}
        try:
            r = http_requests.get(idx_url, timeout=30)
            r.raise_for_status()
            weight_map = r.json().get("weight_map", {})
        except Exception:
            pass

        all_tensors = []
        shard_cache = {}
        if weight_map:
            for tname, shard in weight_map.items():
                all_tensors.append(Q.TensorMeta(tname, [], "?", 0, 0, source=None))
                shard_cache.setdefault(shard, []).append(tname)

        _cache[path] = {"src": None, "metas": all_tensors, "fmt": "repo",
                        "repo_id": repo_id, "weight_map": weight_map,
                        "shard_cache": {}, "shard_parsed": {}}
        return None, all_tensors

    # Direct remote file
    resolved = Q.resolve_url(path)
    src = Q.HTTPSource(resolved)
    if ".safetensors" in resolved.lower():
        tensors, _ = Q.parse_safetensors_header(src)
    else:
        tensors, _ = Q.parse_gguf_header(src)
    _cache[path] = {"src": src, "metas": tensors, "fmt": "remote"}
    return src, tensors


def _resolve_repo_tensor(path, name):
    """Lazy-parse a shard to resolve a placeholder tensor in a repo."""
    import requests as http_requests
    c = _cache.get(path)
    if not c or c["fmt"] != "repo":
        return None
    shard_name = c["weight_map"].get(name)
    if not shard_name:
        return None
    if shard_name not in c["shard_parsed"]:
        url = f"https://huggingface.co/{c['repo_id']}/resolve/main/{shard_name}"
        src = Q.HTTPSource(url)
        ts, _ = Q.parse_safetensors_header(src)
        for t in ts:
            t.source = src
        c["shard_parsed"][shard_name] = {t.name: t for t in ts}
        # Update main metas list
        parsed = c["shard_parsed"][shard_name]
        for i, m in enumerate(c["metas"]):
            if m.name in parsed:
                c["metas"][i] = parsed[m.name]
    parsed = c["shard_parsed"].get(shard_name, {})
    return parsed.get(name)


def get_sm(path):
    """Get (source, metas) for a path, with caching."""
    if path in _cache:
        c = _cache[path]
        return c["src"], c["metas"]

    # Local dir
    p = Path(path).expanduser().resolve() if not path.startswith("http") else None
    if p and p.is_dir():
        src, metas = Q.open_model(str(p))
        _cache[path] = {"src": src, "metas": metas, "fmt": "local_dir"}
        return src, metas

    # Local file
    if p and p.exists():
        src, metas = Q.open_model(str(p))
        _cache[path] = {"src": src, "metas": metas, "fmt": "local"}
        return src, metas

    # Remote / repo
    repo_id = None
    if not path.startswith("http"):
        m = re.match(r"^([^/]+/[^/]+)$", path)
        if m:
            repo_id = m.group(1)
    if path.startswith("http") or repo_id:
        return _open_remote(path)

    raise ValueError(f"Cannot open: {path}")


# ── API routes ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (_here / "quant_viewer.html").read_text()


@app.get("/api/browse")
async def browse(dir: str = Query(".")):
    p = Path(dir).expanduser().resolve()
    if not p.is_dir():
        return JSONResponse({"error": f"Not a directory: {dir}"}, 400)
    items = []
    if p.parent != p:
        items.append({"name": "..", "path": str(p.parent), "type": "dir"})
    try:
        children = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return JSONResponse({"error": f"Permission denied: {dir}"}, 403)
    exts = {".gguf", ".safetensors"}
    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            has = any(child.glob("*.gguf")) or any(child.glob("*.safetensors"))
            items.append({"name": child.name + "/", "path": str(child), "type": "dir", "has_gguf": has})
        elif child.suffix in exts:
            sz = child.stat().st_size
            s = f"{sz/1048576:.1f} MB" if sz < 1073741824 else f"{sz/1073741824:.2f} GB"
            items.append({"name": child.name, "path": str(child), "type": "file", "size": sz, "size_str": s})
    return {"dir": str(p), "items": items}


@app.get("/api/open")
async def open_file(path: str = Query(...)):
    try:
        src, tensors = get_sm(path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 400)

    blocks = {}
    for t in tensors:
        m = re.match(r"(blk\.\d+)", t.name) or re.match(r"(model\.layers\.\d+)", t.name)
        key = m.group(1) if m else "other"
        suf = t.name[len(key)+1:] if m else t.name
        blocks.setdefault(key, []).append({
            "name": t.name, "suffix": suf, "type": t.type_name,
            "shape": list(t.shape), "bytes": t.n_bytes,
        })

    def bk(k):
        if k == "other": return (1, 0)
        m = re.match(r"(?:blk|model\.layers)\.(\d+)", k)
        return (0, int(m.group(1))) if m else (1, 0)

    result = []
    for key in sorted(blocks, key=bk):
        ts = blocks[key]
        types = {}
        for t in ts: types[t["type"]] = types.get(t["type"], 0) + 1
        result.append({"name": key, "n_tensors": len(ts), "total_bytes": sum(t["bytes"] for t in ts),
                        "types": types, "tensors": ts})
    return {"path": path, "n_tensors": len(tensors), "blocks": result}


@app.get("/api/tensor")
async def get_tensor(path: str = Query(...), name: str = Query(...),
                     offset: int = Query(0), limit: int = Query(500),
                     mode: str = Query("local")):
    try:
        src, metas = get_sm(path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 400)

    meta = next((m for m in metas if m.name == name), None)
    if not meta:
        return JSONResponse({"error": f"Not found: {name}"}, 404)
    if meta.n_bytes == 0:
        resolved = _resolve_repo_tensor(path, name)
        if resolved:
            meta = resolved
        else:
            return JSONResponse({"error": f"Could not resolve: {name}"}, 404)

    total = meta.n_elements
    if mode == "web":
        raw_slice = Q.fetch_raw_slice(src, meta, offset, limit)
        chunk = Q.to_f32(raw_slice, meta.dtype, metas, name, src)[:limit]
        sample_size = min(50000, total)
        sample_start = max(0, (total - sample_size) // 2)
        raw_sample = Q.fetch_raw_slice(src, meta, sample_start, sample_size)
        sample = Q.to_f32(raw_sample, meta.dtype, metas, name, src)
        sample = sample[np.isfinite(sample)]
        code_dist = None
    else:
        raw = Q.fetch_raw(src, meta)
        values = Q.to_f32(raw, meta.dtype, metas, name, src)
        total = len(values)
        chunk = values[offset:offset + limit]
        max_sample = 300000
        if total > max_sample:
            sample = values[np.random.default_rng(42).choice(total, max_sample, replace=False)]
        else:
            sample = values
        sample = sample[np.isfinite(sample)]
        code_dist = None
        if Q.NVFP4_TYPE is not None and not isinstance(meta.dtype, str) and Q.safe_ggml_type(meta.dtype) == Q.NVFP4_TYPE:
            flat = raw.reshape(-1, 36)
            qs = flat[:, 4:]
            lo = (qs & 0x0F).flatten(); hi = (qs >> 4).flatten()
            u, cc = np.unique(np.concatenate([lo, hi]), return_counts=True)
            code_dist = {"codes": u.tolist(), "counts": cc.tolist(),
                         "labels": [f"{Q.E2M1_TABLE[c]:.1f}" for c in u]}

    hc, he = (np.histogram(sample, bins=150) if len(sample) > 0 else (np.array([]), np.array([])))
    stats = {}
    outlier_pct = None
    sparse_pct = None
    if len(sample) > 0:
        mean, std = float(np.mean(sample)), float(np.std(sample))
        abs_sample = np.abs(sample)
        outlier_pct = float(np.sum(abs_sample > mean + 3 * std) / len(sample) * 100) if std > 0 else 0.0
        sparse_pct = float(np.sum(abs_sample < std * 0.1) / len(sample) * 100) if std > 0 else 0.0
        stats = {"mean": mean, "std": std,
                 "min": float(np.min(sample)), "max": float(np.max(sample)),
                 "zeros": int(np.sum(sample == 0)),
                 "zero_pct": float(np.sum(sample == 0) / len(sample) * 100)}

    return {"name": name, "type": meta.type_name, "shape": list(meta.shape),
            "total_elements": total, "offset": offset, "limit": limit,
            "values": [round(float(v), 8) for v in chunk],
            "histogram": {"counts": hc.tolist(), "edges": he.tolist()},
            "stats": stats, "outlier_pct": outlier_pct, "sparse_pct": sparse_pct}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)

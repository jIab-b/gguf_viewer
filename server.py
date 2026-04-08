#!/usr/bin/env python3
"""GGUF tensor viewer — local files only, no external dependencies beyond numpy/fastapi."""

import re
import struct
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

_here = Path(__file__).resolve().parent
app = FastAPI()


# ── GGUF parsing ─────────────────────────────────────────────────────

GGUF_MAGIC = 0x46554747

# From ggml: (block_size, type_size) per quant type enum value
# Only the types we need for reading headers + dequanting BF16/F16/F32
GGML_TYPES = {
    0: ("F32", 1, 4),    1: ("F16", 1, 2),   30: ("BF16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    28: ("Q4_0_4_4", 32, 18), 29: ("Q4_0_4_8", 32, 18),
    31: ("NVFP4", 64, 36),
}


class TensorMeta:
    def __init__(self, name, shape, dtype, n_bytes, data_offset):
        self.name, self.shape, self.dtype = name, shape, dtype
        self.n_bytes, self.data_offset = n_bytes, data_offset

    @property
    def type_name(self):
        info = GGML_TYPES.get(self.dtype)
        return info[0] if info else f"UNKNOWN({self.dtype})"

    @property
    def n_elements(self):
        r = 1
        for s in self.shape:
            r *= s
        return r


def parse_gguf(path):
    """Parse a GGUF file header. Returns list of TensorMeta."""
    with open(path, "rb") as f:
        buf = bytearray(f.read(10 * 1024 * 1024))
    pos = 0

    def ensure(n):
        nonlocal buf
        if pos + n > len(buf):
            with open(path, "rb") as f:
                f.seek(len(buf))
                buf.extend(f.read(max(n, 4 * 1024 * 1024)))

    def u32():
        nonlocal pos; ensure(4); v = struct.unpack_from("<I", buf, pos)[0]; pos += 4; return v
    def u64():
        nonlocal pos; ensure(8); v = struct.unpack_from("<Q", buf, pos)[0]; pos += 8; return v
    def read_str():
        nonlocal pos; length = u64(); ensure(length)
        s = buf[pos:pos+length].decode("utf-8"); pos += length; return s

    VAL_SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    def skip_value(vtype):
        nonlocal pos
        if vtype == 8: read_str()
        elif vtype == 9:
            et = u32(); cnt = u64()
            for _ in range(cnt): skip_value(et)
        else:
            sz = VAL_SIZES.get(vtype)
            if sz is None: raise ValueError(f"Unknown GGUF value type {vtype}")
            ensure(sz); pos += sz

    if u32() != GGUF_MAGIC:
        raise ValueError("Not a GGUF file")
    u32()  # version
    tc, kc = u64(), u64()
    for _ in range(kc):
        read_str(); skip_value(u32())

    tensors = []
    for _ in range(tc):
        name = read_str()
        nd = u32()
        shape = [u64() for _ in range(nd)]
        dtype = u32()
        offset = u64()
        info = GGML_TYPES.get(dtype)
        if info:
            bs, ts = info[1], info[2]
        else:
            bs, ts = 1, 1
        ne = 1
        for s in shape:
            ne *= s
        tensors.append(TensorMeta(name, shape, dtype, ne * ts // bs, offset))

    alignment = 32
    ds = pos + (alignment - pos % alignment) % alignment
    for t in tensors:
        t.data_offset = ds + t.data_offset
    return tensors


def fetch_raw(path, meta):
    with open(path, "rb") as f:
        f.seek(meta.data_offset)
        return np.frombuffer(f.read(meta.n_bytes), dtype=np.uint8)


# ── E2M1 / UE4M3 for NVFP4 dequant ─────────────────────────────────
E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_TABLE = np.zeros(16, dtype=np.float32)
E2M1_TABLE[:8] = E2M1_POS
E2M1_TABLE[8:] = -E2M1_POS


def ue4m3_decode(code):
    code = code.astype(np.uint8) & 0x7F
    exp = (code >> 3).astype(np.int32)
    m = (code & 0x07).astype(np.float32)
    out = np.zeros(code.shape, dtype=np.float32)
    s = exp == 0
    out[s] = m[s] / 8.0 * (2.0 ** -6)
    n = exp > 0
    out[n] = (1.0 + m[n] / 8.0) * (2.0 ** (exp[n] - 7).astype(np.float32))
    return out


def dequant_nvfp4(raw, scale2=1.0):
    flat = raw.reshape(-1, 36)
    n = flat.shape[0]
    scales = ue4m3_decode(flat[:, :4])
    qs = flat[:, 4:]
    lo = (qs & 0x0F).astype(np.uint8)
    hi = (qs >> 4).astype(np.uint8)
    codes = np.empty((n, 64), dtype=np.uint8)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    vals = E2M1_TABLE[codes].reshape(n, 4, 16)
    vals *= scales[:, :, np.newaxis]
    return vals.reshape(-1) * scale2


def to_f32(raw, dtype, metas=None, name=None, path=None):
    """Convert raw bytes to float32."""
    info = GGML_TYPES.get(dtype)
    tname = info[0] if info else None
    if tname == "F32":  return raw.view(np.float32)
    if tname == "BF16": return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if tname == "F16":  return raw.view(np.float16).astype(np.float32)
    if tname == "NVFP4":
        s2 = 1.0
        if metas and name and path:
            sn = name.replace(".weight", ".scale")
            for m in metas:
                if m.name == sn:
                    s2 = fetch_raw(path, m).view(np.float32)[0]; break
        return dequant_nvfp4(raw, s2)
    return raw.astype(np.float32)


# ── Model cache ──────────────────────────────────────────────────────
_cache = {}  # path -> {"metas": [TensorMeta]}


def get_metas(path):
    if path in _cache:
        return _cache[path]
    p = Path(path).expanduser().resolve()
    if not p.exists() or not str(p).lower().endswith(".gguf"):
        raise ValueError(f"Not a GGUF file: {path}")
    metas = parse_gguf(str(p))
    _cache[path] = metas
    return metas


# ── API routes ───────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (_here / "viewer.html").read_text()


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
    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            has = any(child.glob("*.gguf"))
            items.append({"name": child.name + "/", "path": str(child), "type": "dir", "has_gguf": has})
        elif child.suffix == ".gguf":
            sz = child.stat().st_size
            s = f"{sz/1048576:.1f} MB" if sz < 1073741824 else f"{sz/1073741824:.2f} GB"
            items.append({"name": child.name, "path": str(child), "type": "file", "size": sz, "size_str": s})
    return {"dir": str(p), "items": items}


@app.get("/api/open")
async def open_file(path: str = Query(...)):
    try:
        metas = get_metas(path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 400)

    blocks = {}
    for t in metas:
        m = re.match(r"(blk\.\d+)", t.name)
        key = m.group(1) if m else "other"
        suf = t.name[len(key)+1:] if m else t.name
        blocks.setdefault(key, []).append({
            "name": t.name, "suffix": suf, "type": t.type_name,
            "shape": list(t.shape), "bytes": t.n_bytes,
        })

    def bk(k):
        if k == "other": return (1, 0)
        m = re.match(r"blk\.(\d+)", k)
        return (0, int(m.group(1))) if m else (1, 0)

    result = []
    for key in sorted(blocks, key=bk):
        ts = blocks[key]
        types = {}
        for t in ts: types[t["type"]] = types.get(t["type"], 0) + 1
        result.append({"name": key, "n_tensors": len(ts), "total_bytes": sum(t["bytes"] for t in ts),
                        "types": types, "tensors": ts})
    return {"path": path, "n_tensors": len(metas), "blocks": result}


@app.get("/api/tensor")
async def get_tensor(path: str = Query(...), name: str = Query(...),
                     offset: int = Query(0), limit: int = Query(500)):
    try:
        metas = get_metas(path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 400)

    meta = next((m for m in metas if m.name == name), None)
    if not meta:
        return JSONResponse({"error": f"Not found: {name}"}, 404)

    raw = fetch_raw(path, meta)
    values = to_f32(raw, meta.dtype, metas, name, path)
    total = len(values)
    chunk = values[offset:offset + limit]

    max_sample = 300000
    if total > max_sample:
        sample = values[np.random.default_rng(42).choice(total, max_sample, replace=False)]
    else:
        sample = values
    sample = sample[np.isfinite(sample)]

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

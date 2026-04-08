# gguf_viewer

Web-based tensor inspector for GGUF and safetensors models. Browse model files locally or from HuggingFace, inspect per-tensor distributions, stats, and raw values.

## Usage

```bash
python quant_server.py
# http://localhost:8877
```

Supports local `.gguf` files, local `.safetensors` files/directories, and remote HuggingFace models via range requests.

## Dependencies

```
numpy
fastapi
uvicorn
```

Also requires `gpu_quant.py` from the parent directory for format parsing (GGUF header, safetensors header, dtype conversion).

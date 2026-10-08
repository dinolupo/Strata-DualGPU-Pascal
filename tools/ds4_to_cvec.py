#!/usr/bin/env python3
"""Convert between DS4 directional steering vectors (.f32) and GGUF control vectors (.gguf).

Supports:
1. DS4 format: flat binary float32 array of shape (n_layer, n_embd), optionally
   accompanied by a JSON metadata file with {"shape": [L, N], "format": "ds4-directional-steering-v1"}.
2. GGUF format: general.architecture='controlvector' with tensors 'direction.<layer>' of shape [n_embd].

Examples:
    # Convert DS4 .f32 to GGUF
    python tools/ds4_to_cvec.py --from-ds4 dir-steering/out/verbosity.f32 --to-gguf dir-steering/out/verbosity.gguf

    # Convert GGUF control vector to DS4 .f32
    python tools/ds4_to_cvec.py --from-gguf vector.gguf --to-ds4 vector.f32

    # Verify round-trip conversion parity
    python tools/ds4_to_cvec.py --check-roundtrip vector.gguf
"""

from __future__ import annotations

import argparse
import array
import json
import os
import pathlib
import struct
import sys
from typing import Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUF_MAGIC, GGUFFile  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402

KNOWN_PROFILES = {
    (48, 2560): "qwen3.8-flash-next",
    (47, 2560): "qwen3.8-flash-next-headless",
    (43, 4096): "deepseek-v4-flash",
    (45, 4096): "glm-5.3-flash",
}


def infer_shape(f32_path: pathlib.Path, json_meta: Optional[dict] = None) -> Tuple[int, int]:
    """Determine (n_layer, n_embd) from JSON metadata or file byte size."""
    if json_meta and "shape" in json_meta:
        shape = json_meta["shape"]
        if len(shape) == 2:
            return int(shape[0]), int(shape[1])

    file_bytes = f32_path.stat().st_size
    float_count = file_bytes // 4
    if file_bytes % 4 != 0:
        raise ValueError(f"{f32_path}: file size {file_bytes} is not a multiple of 4 bytes")

    # Check known shapes first
    for (l, n) in KNOWN_PROFILES:
        if l * n == float_count:
            return l, n

    # Default assumption for Qwen3.8 Flash Next embedding width
    if float_count % 2560 == 0:
        return float_count // 2560, 2560
    if float_count % 4096 == 0:
        return float_count // 4096, 4096

    raise ValueError(
        f"{f32_path}: cannot automatically infer shape for {float_count} floats ({file_bytes} bytes). "
        "Specify --shape L,N or provide a companion .json file."
    )


def ds4_to_gguf(
    f32_path: pathlib.Path,
    gguf_path: pathlib.Path,
    shape: Optional[Tuple[int, int]] = None,
    model_hint: str = "qwen4exp",
    json_path: Optional[pathlib.Path] = None,
) -> None:
    """Convert DS4 raw .f32 steering tensor to GGUF controlvector."""
    meta = {}
    json_candidate = json_path or f32_path.with_suffix(".json")
    if json_candidate.is_file():
        try:
            meta = json.loads(json_candidate.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: could not parse {json_candidate}: {e}", file=sys.stderr)

    if shape is None:
        shape = infer_shape(f32_path, meta)

    n_layer, n_embd = shape
    expected_bytes = n_layer * n_embd * 4
    actual_bytes = f32_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{f32_path}: size mismatch. Expected {n_layer}x{n_embd}x4 = {expected_bytes} bytes, got {actual_bytes}"
        )

    print(f"Reading DS4 file {f32_path} (shape: {n_layer} layers, {n_embd} embd)...")
    data = array.array("f")
    with f32_path.open("rb") as f:
        data.fromfile(f, n_layer * n_embd)

    import numpy as np
    arr = np.frombuffer(data, dtype=np.float32).reshape((n_layer, n_embd))

    writer = GGUFWriter()
    writer.add("general.architecture", "controlvector")
    writer.add("controlvector.model_hint", model_hint)
    writer.add("controlvector.layer_count", n_layer - 1 if n_layer > 1 else 1)

    tensors_written = 0
    # In GGUF controlvector, directions are conventionally named direction.1 .. direction.<L-1>
    # If layer 0 has non-zero data, we also write direction.0
    start_layer = 0 if np.any(arr[0] != 0.0) else 1
    for layer in range(start_layer, n_layer):
        writer.add_f32(f"direction.{layer}", arr[layer])
        tensors_written += 1

    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    writer.write(gguf_path)
    print(f"Successfully wrote {gguf_path} ({tensors_written} direction tensors, hint={model_hint}).")


def gguf_to_ds4(
    gguf_path: pathlib.Path,
    f32_path: pathlib.Path,
    json_path: Optional[pathlib.Path] = None,
    total_layers: Optional[int] = None,
) -> None:
    """Convert GGUF controlvector to DS4 .f32 file and JSON metadata."""
    print(f"Reading GGUF file {gguf_path}...")
    gf = GGUFFile(gguf_path)
    arch = gf.metadata.get("general.architecture")
    if arch != "controlvector":
        print(f"Warning: general.architecture is '{arch}', expected 'controlvector'", file=sys.stderr)

    import numpy as np

    # Collect direction tensors
    dir_tensors = {}
    n_embd = None
    for t in gf.tensors:
        if not t.name.startswith("direction."):
            continue
        try:
            layer_idx = int(t.name.split(".", 1)[1])
        except ValueError:
            continue
        if n_embd is None:
            n_embd = t.elements
        elif n_embd != t.elements:
            raise ValueError(f"Inconsistent embedding dimension in tensor {t.name}: {t.elements} vs {n_embd}")
        dir_tensors[layer_idx] = t

    if not dir_tensors:
        raise ValueError(f"{gguf_path}: no 'direction.<layer>' tensors found")

    max_layer = max(dir_tensors.keys())
    # Determine n_layer: if total_layers given use that, else max_layer + 1
    n_layer = total_layers if total_layers is not None else (max_layer + 1)
    if n_layer <= max_layer:
        n_layer = max_layer + 1

    # Default Qwen3.8 Flash Next check: 48 layers if max_layer == 47
    if n_layer == 48 and n_embd == 2560:
        pass

    out_arr = np.zeros((n_layer, n_embd), dtype=np.float32)

    with gguf_path.open("rb") as f:
        for l_idx, t in dir_tensors.items():
            if l_idx >= n_layer:
                continue
            f.seek(gf.data_start + t.offset)
            t_data = np.fromfile(f, dtype="<f4", count=n_embd)
            out_arr[l_idx] = t_data

    f32_path.parent.mkdir(parents=True, exist_ok=True)
    with f32_path.open("wb") as f:
        out_arr.tofile(f)

    # Write companion JSON
    json_out = json_path or f32_path.with_suffix(".json")
    meta = {
        "format": "ds4-directional-steering-v1",
        "shape": [n_layer, n_embd],
        "component": "ffn_out",
        "source_gguf": str(gguf_path.name),
        "model_hint": str(gf.metadata.get("controlvector.model_hint", "qwen4exp")),
        "note": "converted from GGUF controlvector format",
    }
    json_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Successfully wrote {f32_path} ({n_layer}x{n_embd} floats, {f32_path.stat().st_size} bytes)")
    print(f"Wrote metadata to {json_out}")


def check_roundtrip(src_gguf: pathlib.Path) -> bool:
    """Verify bitwise exact round-trip conversion: GGUF -> DS4 -> GGUF."""
    import tempfile
    import numpy as np

    print(f"--- Round-trip Parity Check for {src_gguf} ---")
    with tempfile.TemporaryDirectory(prefix="cvec-roundtrip-") as td:
        tmp_dir = pathlib.Path(td)
        tmp_f32 = tmp_dir / "converted.f32"
        tmp_gguf = tmp_dir / "roundtrip.gguf"

        gguf_to_ds4(src_gguf, tmp_f32)
        ds4_to_gguf(tmp_f32, tmp_gguf)

        orig_gf = GGUFFile(src_gguf)
        new_gf = GGUFFile(tmp_gguf)

        orig_tensors = {t.name: t for t in orig_gf.tensors if t.name.startswith("direction.")}
        new_tensors = {t.name: t for t in new_gf.tensors if t.name.startswith("direction.")}

        if set(orig_tensors.keys()) != set(new_tensors.keys()):
            print(f"FAIL: Tensor sets differ! Orig: {len(orig_tensors)}, New: {len(new_tensors)}")
            return False

        max_err = 0.0
        with src_gguf.open("rb") as f_orig, tmp_gguf.open("rb") as f_new:
            for name, t_orig in orig_tensors.items():
                t_new = new_tensors[name]
                f_orig.seek(orig_gf.data_start + t_orig.offset)
                f_new.seek(new_gf.data_start + t_new.offset)
                d_orig = np.fromfile(f_orig, dtype="<f4", count=t_orig.elements)
                d_new = np.fromfile(f_new, dtype="<f4", count=t_new.elements)
                err = float(np.max(np.abs(d_orig - d_new)))
                if err > max_err:
                    max_err = err

        print(f"Maximum absolute float difference across all tensors: {max_err:.10e}")
        if max_err == 0.0:
            print("PASS: Exact bitwise match (max_err = 0.0)!")
            return True
        elif max_err < 1e-7:
            print("PASS: Match within float32 precision limits.")
            return True
        else:
            print(f"FAIL: Error exceeds threshold: {max_err}")
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert between DS4 .f32 and GGUF control vectors.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-ds4", type=pathlib.Path, help="Input DS4 .f32 file")
    group.add_argument("--from-gguf", type=pathlib.Path, help="Input GGUF controlvector file")
    group.add_argument("--check-roundtrip", type=pathlib.Path, help="Check roundtrip fidelity on a GGUF file")

    parser.add_argument("--to-gguf", type=pathlib.Path, help="Output GGUF file")
    parser.add_argument("--to-ds4", type=pathlib.Path, help="Output DS4 .f32 file")
    parser.add_argument("--shape", type=str, help="Shape as L,N (e.g. 48,2560)")
    parser.add_argument("--model-hint", default="qwen4exp", help="GGUF model hint (default: qwen4exp)")
    parser.add_argument("--total-layers", type=int, help="Total layers for output DS4 tensor")

    args = parser.parse_args()

    if args.check_roundtrip:
        ok = check_roundtrip(args.check_roundtrip)
        sys.exit(0 if ok else 1)

    if args.from_ds4:
        if not args.to_gguf:
            parser.error("--from-ds4 requires --to-gguf")
        shape = None
        if args.shape:
            parts = [int(x.strip()) for x in args.shape.split(",")]
            if len(parts) != 2:
                parser.error("--shape must be L,N")
            shape = (parts[0], parts[1])
        ds4_to_gguf(args.from_ds4, args.to_gguf, shape=shape, model_hint=args.model_hint)

    elif args.from_gguf:
        if not args.to_ds4:
            parser.error("--from-gguf requires --to-ds4")
        gguf_to_ds4(args.from_gguf, args.to_ds4, total_layers=args.total_layers)


if __name__ == "__main__":
    main()

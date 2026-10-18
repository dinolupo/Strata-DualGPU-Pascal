#!/usr/bin/env python3
"""Build a directional-steering vector from paired contrast prompt sets.

Implements the DS4 directional steering vector construction methodology:
Given pairs of target ('good') and control ('bad') prompts (e.g. concise vs verbose,
creative vs analytical, etc.):
1. Extracts per-layer activation vectors for the last prompt token position.
2. Computes the contrast direction: mean(good) - mean(bad) or pair-normalized difference.
3. Orthogonalizes the direction against the baseline control mean (Gram-Schmidt) to isolate
   the pure directional attribute without shifting the unconditioned latent center.
4. Normalizes each layer's direction vector to unit norm.
5. Saves in DS4 flat .f32 binary format and/or standard GGUF controlvector format.

Mathematical operation applied at runtime (DS4 & Strata project mode):
    h' = h - scale * direction[layer] * (h . direction[layer])
    (positive scale suppresses the direction; negative scale amplifies it).
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "serve"))
from ds4_to_cvec import ds4_to_gguf  # noqa: E402
from frontend import ChatTemplate  # noqa: E402 - the exact server chat-template renderer
import strata_tokenizer as ST  # noqa: E402

MODEL_PROFILES = {
    "qwen3.8-flash-next": (48, 2560, 4),      # layers, n_embd, hc
    "deepseek-v4-flash": (43, 4096, 1),
    "glm-5.3-flash": (45, 4096, 1),
}


def read_prompt_file(path: pathlib.Path, max_prompts: Optional[int] = None) -> List[str]:
    """Read prompts from .parquet, .jsonl, or plain .txt files."""
    prompts = []
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(str(path))
            target_col = None
            for candidate in ["text", "prompt", "instruction", "input"]:
                if candidate in tbl.column_names:
                    target_col = candidate
                    break
            if not target_col:
                target_col = tbl.column_names[0]
            col_data = tbl[target_col]
            limit = len(col_data) if max_prompts is None else min(len(col_data), max_prompts)
            for i in range(limit):
                val = col_data[i].as_py()
                if val is not None and str(val).strip():
                    prompts.append(str(val).strip())
            return prompts
        except Exception as e:
            raise ValueError(f"Failed to read parquet file {path}: {e}")

    elif path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txt = None
                for candidate in ["text", "prompt", "instruction", "input"]:
                    if candidate in obj:
                        txt = str(obj[candidate]).strip()
                        break
                if txt:
                    prompts.append(txt)
            except Exception:
                continue
            if max_prompts and len(prompts) >= max_prompts:
                break
        return prompts

    # Default: plain text, one prompt per non-empty line
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
        if max_prompts and len(prompts) >= max_prompts:
            break
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_git_commit() -> Optional[str]:
    """HEAD commit of the repository that contains this script (records the engine/build source state)."""
    repo = pathlib.Path(__file__).resolve().parent.parent
    res = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0:
        return res.stdout.decode().strip()
    return None


def load_pack_tokenizer(pack_dir: pathlib.Path):
    """The pack tokenizer (tools/strata_tokenizer), or None when the pack has no vocab.json."""
    tpath = pack_dir / "tokenizer"
    if not (tpath / "vocab.json").is_file():
        return None
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def build_prompt_renderer(pack_dir: pathlib.Path, system: str, thinking: str, raw_prompt: bool) -> Tuple[Callable[[str], str], dict]:
    """Returns (render(text) -> str, metadata) for the prompt fed to the engine.

    Template mode renders the pack's own chat_template.jinja through serve.frontend.ChatTemplate
    (the exact server renderer): system message, user turn, assistant-turn opening. The captured position
    is the last token of the rendered prompt, and the metadata records the last six tokens of that prompt
    as computed at runtime (ids and decoded text), so the position is stated by measurement, not by name.
    --raw-prompt keeps the legacy behavior (raw text tokenization, no template).
    """
    if raw_prompt:
        def render(text: str) -> str:
            return text
        meta = {
            "raw_prompt": True,
            "chat_template": None,
            "system_prompt": None,
            "thinking": None,
            "capture_position": "last token of the raw prompt text (legacy mode)",
        }
        return render, meta

    template_path = pack_dir / "tokenizer" / "chat_template.jinja"
    if not template_path.is_file():
        raise ValueError(f"pack {pack_dir} has no tokenizer/chat_template.jinja; pass --raw-prompt for the legacy mode")
    ct = ChatTemplate(template_path)
    kwargs = {"enable_thinking": False} if thinking == "none" else {"reasoning_effort": thinking}

    def render(text: str) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        return ct.render(messages=messages, tools=None, add_generation_prompt=True, **kwargs)

    tok = load_pack_tokenizer(pack_dir)
    tail_ids: List[int] = []
    tail_text = ""
    tail_note = "pack tokenizer missing; the tail tokens of the rendered prompt were not recorded"
    if tok is not None:
        tail_ids = tok.encode(render("probe"), parse_special=True)[-6:]
        tail_text, tail_note = tok.decode(tail_ids), None
    parity = ("parity with ds4 default: capture at start of direct answer (thinking disabled)"
              if thinking == "none" else "capture at start of reasoning, differs from ds4 default")

    meta = {
        "raw_prompt": False,
        "chat_template_sha256": sha256_file(template_path),
        "chat_template_path": str(template_path),
        "system_prompt": system,
        "thinking": thinking,
        "template_kwargs": kwargs,
        "capture_position": f"last token of the rendered prompt; {parity}",
        "capture_tail_ids": tail_ids,
        "capture_tail_text": tail_text,
        "capture_tail_note": tail_note,
    }
    return render, meta


def normalize(v: List[float]) -> List[float]:
    n2 = sum(x * x for x in v)
    if n2 <= 0.0:
        return v
    inv = 1.0 / math.sqrt(n2)
    return [x * inv for x in v]


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def extract_strata_activations(
    engine_bin: pathlib.Path,
    pack_dir: pathlib.Path,
    ple_gguf: Optional[pathlib.Path],
    prompt: str,
    n_layer: int,
    n_embd: int,
    hc: int,
    work_dir: pathlib.Path,
    render: Callable[[str], str],
) -> List[List[float]]:
    """Runs strata generate with --dump-layers to capture activations for the prompt's final token.

    `render` turns the raw contrast prompt into the text fed to the engine (server chat template by
    default, raw text with --raw-prompt). The captured position is the last token of that text.
    """
    prompt = render(prompt)
    dump_file = work_dir / "layers.bin"
    tokens_file = work_dir / "tokens.txt"

    # Tokenize prompt using strata_tokenizer if pack has vocab.json
    tpath = pack_dir / "tokenizer"
    if (tpath / "vocab.json").is_file():
        import strata_tokenizer as ST
        vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tpath / "token_type.json").read_text())
        tok = ST.Tokenizer(tokens, merges, types)
        encoded = tok.encode(prompt, parse_special=True)
        token_ids = ",".join(str(x) for x in encoded)
    else:
        # Fallback dummy tokens
        token_ids = "1, 2"

    tokens_file.write_text(token_ids, encoding="utf-8")
    num_tokens = len(token_ids.split(","))

    cmd = [
        str(engine_bin),
        "--pack", str(pack_dir),
        "--tokens-file", str(tokens_file),
        "--max-new", "1",
        "--dump-layers", str(dump_file),
    ]
    if ple_gguf and ple_gguf.is_file():
        cmd += ["--ple-gguf", str(ple_gguf)]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"strata execution failed:\n{err}")

    # Read dumped activations
    # --dump-layers writes (n_layers + 1) * hc * n_embd floats per position
    record_floats = (n_layer + 1) * hc * n_embd
    record_bytes = record_floats * 4
    file_bytes = dump_file.stat().st_size
    n_pos = file_bytes // record_bytes
    if n_pos < 1:
        raise RuntimeError(f"dump file {dump_file} has insufficient data: {file_bytes} bytes")

    # Read the prompt's last token position
    last_pos_offset = (num_tokens - 1) * record_bytes if num_tokens <= n_pos else (n_pos - 1) * record_bytes
    data = array.array("f")
    with dump_file.open("rb") as f:
        f.seek(last_pos_offset)
        data.fromfile(f, record_floats)

    # Extract layer rows: layer 0..n_layer-1 (layer 0 in dump is input embedding, layers 1..n_layer are transformer layers)
    rows: List[List[float]] = []
    for l in range(n_layer):
        # Layer l in model corresponds to dump layer l + 1
        dump_layer_idx = l + 1
        layer_start = dump_layer_idx * (hc * n_embd)
        # Average across hyperconnection streams or use stream 0
        avg_stream = [0.0] * n_embd
        for s in range(hc):
            s_offset = layer_start + s * n_embd
            for j in range(n_embd):
                avg_stream[j] += data[s_offset + j]
        inv_hc = 1.0 / float(hc)
        rows.append([x * inv_hc for x in avg_stream])

    return rows


def compute_steering_directions(
    good_activations: List[List[List[float]]],
    bad_activations: List[List[List[float]]],
    n_layer: int,
    n_embd: int,
    pair_normalize: bool = False,
    orthogonalize: bool = True,
) -> List[List[float]]:
    """Compute per-layer unit direction vectors using the contrast pairs."""
    n = min(len(good_activations), len(bad_activations))
    if n == 0:
        raise ValueError("No activation pairs provided")

    good_sum = [[0.0] * n_embd for _ in range(n_layer)]
    bad_sum = [[0.0] * n_embd for _ in range(n_layer)]
    pair_sum = [[0.0] * n_embd for _ in range(n_layer)]

    for i in range(n):
        good_rows = good_activations[i]
        bad_rows = bad_activations[i]
        for l in range(n_layer):
            for j in range(n_embd):
                good_sum[l][j] += good_rows[l][j]
                bad_sum[l][j] += bad_rows[l][j]
            if pair_normalize:
                diff = normalize([good_rows[l][j] - bad_rows[l][j] for j in range(n_embd)])
                for j in range(n_embd):
                    pair_sum[l][j] += diff[j]

    directions: List[List[float]] = []
    for l in range(n_layer):
        good_mean = [x / n for x in good_sum[l]]
        bad_mean = [x / n for x in bad_sum[l]]

        if pair_normalize:
            direction = normalize([x / n for x in pair_sum[l]])
        else:
            direction = normalize([good_mean[j] - bad_mean[j] for j in range(n_embd)])

        if orthogonalize:
            base = normalize(bad_mean)
            proj = dot(direction, base)
            direction = normalize([direction[j] - proj * base[j] for j in range(n_embd)])

        directions.append(direction)

    return directions


def save_steering_files(
    directions: List[List[float]],
    out_prefix: pathlib.Path,
    profile: str,
    component: str,
    good_file: str,
    bad_file: str,
    pair_normalize: bool,
    orthogonalize: bool,
    write_gguf: bool = True,
    extra_meta: Optional[dict] = None,
) -> Tuple[pathlib.Path, Optional[pathlib.Path]]:
    """Save .f32, .json, and optionally .gguf files."""
    n_layer = len(directions)
    n_embd = len(directions[0]) if n_layer > 0 else 0

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    f32_path = out_prefix.with_suffix(".f32")
    json_path = out_prefix.with_suffix(".json")
    gguf_path = out_prefix.with_suffix(".gguf") if write_gguf else None

    # Write .f32
    flat = array.array("f")
    for row in directions:
        flat.extend(row)
    with f32_path.open("wb") as f:
        flat.tofile(f)

    # Write .json
    meta = {
        "format": "ds4-directional-steering-v1",
        "shape": [n_layer, n_embd],
        "profile": profile,
        "component": component,
        "pair_normalize": pair_normalize,
        "orthogonalize_control_mean": orthogonalize,
        "good_file": good_file,
        "bad_file": bad_file,
        "note": "positive scale suppresses this direction; negative scale amplifies it",
    }
    if extra_meta:
        meta.update(extra_meta)
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {f32_path} ({n_layer}x{n_embd} f32)")
    print(f"Wrote {json_path}")

    # Write .gguf
    if gguf_path:
        ds4_to_gguf(f32_path, gguf_path, shape=(n_layer, n_embd))
        print(f"Wrote {gguf_path}")

    return f32_path, gguf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build directional steering vectors from contrast prompt pairs.")
    parser.add_argument("--good-file", required=True, help="Path to file containing desired/target prompts")
    parser.add_argument("--bad-file", required=True, help="Path to file containing control/contrast prompts")
    parser.add_argument("--out", default="out/steering", help="Output path prefix (without extension)")
    parser.add_argument("--profile", choices=MODEL_PROFILES.keys(), default="qwen3.8-flash-next")
    parser.add_argument("--component", choices=["ffn_out", "attn_out"], default="ffn_out")
    parser.add_argument("--pair-normalize", action="store_true", help="Average normalized per-pair differences")
    parser.add_argument("--no-orthogonalize", action="store_true", help="Skip Gram-Schmidt projection against control mean")
    parser.add_argument("--no-gguf", action="store_true", help="Do not write GGUF format (only .f32 + .json)")
    parser.add_argument("--max-prompts", type=int, default=None, help="Maximum number of contrast prompt pairs to evaluate")

    # Prompt rendering (server chat template by default; ds4 renders the template too)
    parser.add_argument("--raw-prompt", action="store_true", help="legacy mode: tokenize the raw prompt text, no chat template")
    parser.add_argument("--system", default="You are a helpful assistant.", help="system message rendered into the chat template")
    parser.add_argument("--thinking", choices=["xhigh", "medium", "low", "none"], default="none",
                        help="template thinking kwargs: reasoning_effort level (server convention) or "
                             "enable_thinking=false ('none', the ds4 default: capture at the start of the direct answer)")

    # Strata engine args (if extracting directly via Strata)
    parser.add_argument("--engine", type=pathlib.Path, help="Path to Strata engine executable")
    parser.add_argument("--pack", type=pathlib.Path, help="Path to Strata pack directory")
    parser.add_argument("--ple-gguf", type=pathlib.Path, help="Path to PLE GGUF shard")

    args = parser.parse_args()

    good_prompts = read_prompt_file(pathlib.Path(args.good_file), max_prompts=args.max_prompts)
    bad_prompts = read_prompt_file(pathlib.Path(args.bad_file), max_prompts=args.max_prompts)
    n = min(len(good_prompts), len(bad_prompts))
    if args.max_prompts and n > args.max_prompts:
        n = args.max_prompts
    print(f"Loaded {n} contrast prompt pairs.")

    n_layer, n_embd, hc = MODEL_PROFILES[args.profile]

    if not args.engine or not args.pack:
        print("Note: --engine and --pack not provided. Validating prompt pairs and algorithm.")
        print("To extract activations from Strata, supply --engine and --pack.")
        return

    render, render_meta = build_prompt_renderer(args.pack, args.system, args.thinking, args.raw_prompt)
    if args.raw_prompt:
        print("Mode: --raw-prompt (legacy raw-text tokenization, no chat template).")
    else:
        print(f"Mode: server chat template ({render_meta['chat_template_sha256'][:16]}...), thinking={args.thinking}.")

    good_acts = []
    bad_acts = []
    with tempfile.TemporaryDirectory(prefix="strata-steer-") as td:
        tmp_dir = pathlib.Path(td)
        for i, (g_p, b_p) in enumerate(zip(good_prompts[:n], bad_prompts[:n]), 1):
            print(f"Capturing pair {i}/{n}...")
            g_act = extract_strata_activations(args.engine, args.pack, args.ple_gguf, g_p, n_layer, n_embd, hc, tmp_dir, render)
            b_act = extract_strata_activations(args.engine, args.pack, args.ple_gguf, b_p, n_layer, n_embd, hc, tmp_dir, render)
            good_acts.append(g_act)
            bad_acts.append(b_act)

    directions = compute_steering_directions(
        good_acts,
        bad_acts,
        n_layer,
        n_embd,
        pair_normalize=args.pair_normalize,
        orthogonalize=not args.no_orthogonalize,
    )

    save_steering_files(
        directions,
        pathlib.Path(args.out),
        profile=args.profile,
        component=args.component,
        good_file=args.good_file,
        bad_file=args.bad_file,
        pair_normalize=args.pair_normalize,
        orthogonalize=not args.no_orthogonalize,
        write_gguf=not args.no_gguf,
        extra_meta={
            **render_meta,
            "num_pairs": n,
            "git_commit": repo_git_commit(),
            "engine_binary": str(args.engine),
            "engine_binary_sha256": sha256_file(args.engine),
            "pack_dir": str(args.pack),
            "good_file_sha256": sha256_file(pathlib.Path(args.good_file)),
            "bad_file_sha256": sha256_file(pathlib.Path(args.bad_file)),
            "tokenizer_sha256": {name: sha256_file(args.pack / "tokenizer" / name)
                                 for name in ("vocab.json", "merges.txt", "token_type.json")
                                 if (args.pack / "tokenizer" / name).is_file()},
        },
    )


if __name__ == "__main__":
    main()

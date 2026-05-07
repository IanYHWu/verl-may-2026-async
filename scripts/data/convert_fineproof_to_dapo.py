#!/usr/bin/env python3
"""Convert fineproof RL parquets into DAPO chat format and (optionally) push to HF.

Source schema (per row):
    problem: str
    data_source: str ∈ {aops, olympiads}
    reward_model: {ground_truth: str (rubric)}

Target schema (DAPO chat format, matches dapo_math_17k):
    prompt: list[{role: str, content: str}]
    data_source: str (preserved: aops / olympiads)
    ability: str ("math_proof")
    reward_model: {ground_truth: str (rubric), style: "rubric"}
    extra_info: {index: str}

The `data_source` is preserved so that downstream telemetry can split metrics
by source. The `style="rubric"` field marks the row as rubric-graded; the
`LLMJudgeRewardManager` does not dispatch on style, but a future hybrid
router could.

Usage:
    # Local conversion only:
    python scripts/data/convert_fineproof_to_dapo.py \
        --src-train /path/to/fineproof_rl_cleaned_train.parquet \
        --src-test  /path/to/fineproof_rl_cleaned_test.parquet \
        --out-dir   /path/to/output/fineproof_dapo

    # Local conversion + HF upload:
    python scripts/data/convert_fineproof_to_dapo.py \
        --src-train ... --src-test ... --out-dir ... \
        --hf-repo HerrHruby/fineproofs --push
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INSTRUCTION = (
    "Solve the following olympiad-style mathematics problem. "
    "Present a complete and rigorous proof, showing all key steps and reasoning."
)


def _row_index(problem: str, data_source: str) -> str:
    """Stable per-row id derived from problem text + source."""
    h = hashlib.sha1(f"{data_source}\x00{problem}".encode("utf-8")).hexdigest()
    return h[:16]


def convert_table(t: pa.Table) -> pa.Table:
    df = t.to_pandas()
    out = []
    for _, row in df.iterrows():
        problem = str(row["problem"])
        data_source = str(row["data_source"])
        rubric = str(row["reward_model"]["ground_truth"])
        out.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": f"{INSTRUCTION}\n\n{problem}",
                    }
                ],
                "data_source": data_source,
                "ability": "math_proof",
                "reward_model": {
                    "ground_truth": rubric,
                    "style": "rubric",
                },
                "extra_info": {"index": _row_index(problem, data_source)},
            }
        )
    return pa.Table.from_pylist(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-train", required=True, type=Path)
    p.add_argument("--src-test", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--hf-repo",
        default=None,
        help="HF dataset repo id, e.g. HerrHruby/fineproofs",
    )
    p.add_argument("--push", action="store_true", help="Push converted parquets to HF")
    p.add_argument("--private", action="store_true", help="Make HF repo private")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_train = args.out_dir / "train.parquet"
    out_test = args.out_dir / "test.parquet"

    for src, dst in [(args.src_train, out_train), (args.src_test, out_test)]:
        if not src.is_file():
            print(f"ERROR: source not found: {src}", file=sys.stderr)
            return 1
        t = pq.read_table(str(src))
        new_t = convert_table(t)
        pq.write_table(new_t, str(dst))
        print(f"Wrote {dst}  rows={new_t.num_rows}")

    if not args.push:
        print("Done (no upload). Pass --push to upload.")
        return 0

    if not args.hf_repo:
        print("ERROR: --hf-repo is required with --push", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True, private=args.private)
    for split, path in [("train", out_train), ("test", out_test)]:
        # HF datasets uses path-in-repo "data/{split}-00000-of-00001.parquet" by convention.
        dst_in_repo = f"data/{split}-00000-of-00001.parquet"
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dst_in_repo,
            repo_id=args.hf_repo,
            repo_type="dataset",
            commit_message=f"Upload {split} split (DAPO chat format)",
        )
        print(f"Uploaded {path} -> {args.hf_repo}:{dst_in_repo}")

    print(f"Done. https://huggingface.co/datasets/{args.hf_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

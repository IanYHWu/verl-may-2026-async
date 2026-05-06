# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset format checker for the LLM judge reward path.

Validates that a parquet file conforms to the DAPO chat schema expected by
``LLMJudgeRewardManager``:

  - ``prompt`` : list of ``{"role": str, "content": str}``, non-empty.
  - ``data_source`` : str.
  - ``reward_model`` : dict with at least ``ground_truth`` (str, non-empty).
    For the LLM judge, ``ground_truth`` is the rubric.
  - ``ability`` : str (optional but recommended).
  - ``extra_info`` : dict (optional).

If ``extra_fields`` is supplied (matching the manager's
``reward.reward_kwargs.judge.extra_fields`` config), the checker also
verifies that each dotted path resolves on every sampled row.

Use programmatically::

    from verl.utils.judge.check_dataset import check_dataset
    report = check_dataset("/path/to/data.parquet",
                           extra_fields={"reference_solution": "reward_model.reference_solution"})
    print(report)
    assert report.ok

Or from the command line::

    python -m verl.utils.judge.check_dataset \\
        --parquet /path/to/data.parquet \\
        --extra-field reference_solution=reward_model.reference_solution
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .paths import resolve_dotted

REQUIRED_COLUMNS = ("prompt", "data_source", "reward_model")
RECOMMENDED_COLUMNS = ("ability", "extra_info")
PROMPT_MSG_FIELDS = ("role", "content")


@dataclass
class CheckReport:
    path: str
    num_rows: int
    schema: str
    columns: list[str]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:  # pragma: no cover (cosmetic)
        lines: list[str] = []
        lines.append(f"Dataset: {self.path}")
        lines.append(f"Rows: {self.num_rows}")
        lines.append(f"Columns: {', '.join(self.columns)}")
        lines.append("")
        lines.append("Schema:")
        for line in self.schema.splitlines():
            lines.append(f"  {line}")
        lines.append("")
        for tag, items in (("ERROR", self.errors), ("WARN ", self.warnings), ("INFO ", self.info)):
            for item in items:
                lines.append(f"[{tag}] {item}")
        lines.append("")
        lines.append(f"errors={len(self.errors)} warnings={len(self.warnings)} info={len(self.info)}")
        lines.append("OK" if self.ok else "FAILED")
        return "\n".join(lines)


def check_dataset(
    path: str | Path,
    *,
    extra_fields: Optional[dict[str, str]] = None,
    sample_size: Optional[int] = None,
) -> CheckReport:
    """Check that a parquet file conforms to the LLM judge dataset schema.

    Args:
        path: parquet file path.
        extra_fields: optional mapping ``var -> dotted.path``; each path must
            resolve on every sampled row.
        sample_size: if set, validate this many randomly sampled rows
            (without replacement). ``None`` validates every row.

    Returns:
        ``CheckReport`` capturing errors / warnings / info.
    """
    import pyarrow.parquet as pq

    path = str(path)
    table = pq.read_table(path)
    df = table.to_pandas()
    columns = list(df.columns)

    report = CheckReport(
        path=path,
        num_rows=len(df),
        schema=str(table.schema),
        columns=columns,
    )

    # 1. Column presence.
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        report.errors.append(f"Missing required columns: {missing}")
        # Without these we can't meaningfully row-check; bail early.
        return report

    for col in RECOMMENDED_COLUMNS:
        if col not in columns:
            report.warnings.append(
                f"Recommended column {col!r} missing — dataset will work but downstream telemetry may degrade"
            )

    # 2. Per-row schema (optionally sampled).
    rows_iter: Iterable[tuple[int, Any]]
    if sample_size is not None and len(df) > sample_size:
        sampled = df.sample(n=sample_size, random_state=0)
        rows_iter = sampled.iterrows()
        report.info.append(f"Sampled {sample_size} rows out of {len(df)}")
    else:
        rows_iter = df.iterrows()

    extra_fields = dict(extra_fields or {})

    n_errors_before = len(report.errors)
    for idx, row in rows_iter:
        _check_row(idx, row, extra_fields, report)
        # Cap errors to keep output usable; fail fast on bad datasets.
        if len(report.errors) - n_errors_before >= 25:
            report.errors.append("...stopping after 25 row errors")
            break

    if extra_fields and report.ok:
        report.info.append(
            f"All extra_fields paths resolved on every checked row: {sorted(extra_fields.keys())}"
        )

    return report


def _check_row(
    idx: Any,
    row: Any,
    extra_fields: dict[str, str],
    report: CheckReport,
) -> None:
    # prompt
    prompt = row.get("prompt") if hasattr(row, "get") else row["prompt"]
    if prompt is None:
        report.errors.append(f"row {idx}: prompt is null")
        return
    try:
        msgs = list(prompt)
    except TypeError:
        report.errors.append(f"row {idx}: prompt is not iterable")
        return
    if len(msgs) == 0:
        report.errors.append(f"row {idx}: prompt is empty")
    has_user = False
    for j, msg in enumerate(msgs):
        if not _is_dict_like(msg):
            report.errors.append(f"row {idx}, prompt[{j}]: not a dict (got {type(msg).__name__})")
            continue
        for fld in PROMPT_MSG_FIELDS:
            if fld not in msg:
                report.errors.append(f"row {idx}, prompt[{j}]: missing {fld!r}")
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str):
            report.errors.append(f"row {idx}, prompt[{j}]: role is not str")
        if not isinstance(content, str):
            report.errors.append(f"row {idx}, prompt[{j}]: content is not str")
        if role == "user":
            has_user = True
    if msgs and not has_user:
        report.warnings.append(
            f"row {idx}: prompt has no 'user' role message — judge will fall back to decoded prompt tensor"
        )

    # data_source
    data_source = row["data_source"]
    if not isinstance(data_source, str) or not data_source:
        report.errors.append(f"row {idx}: data_source must be a non-empty str (got {data_source!r})")

    # reward_model
    rm = row["reward_model"]
    if not _is_dict_like(rm):
        report.errors.append(f"row {idx}: reward_model is not a dict (got {type(rm).__name__})")
    else:
        if "ground_truth" not in rm:
            report.errors.append(f"row {idx}: reward_model.ground_truth missing")
        else:
            gt = rm["ground_truth"]
            if not isinstance(gt, str):
                report.errors.append(f"row {idx}: reward_model.ground_truth is not str")
            elif not gt.strip():
                report.errors.append(f"row {idx}: reward_model.ground_truth is empty")

    # extra_fields
    if extra_fields:
        # Build a dict shaped like data_item.non_tensor_batch.
        non_tensor_like: dict[str, Any] = {}
        for col in row.index:
            non_tensor_like[col] = row[col]
        for var_name, path in extra_fields.items():
            try:
                val = resolve_dotted(non_tensor_like, path)
            except (KeyError, TypeError, IndexError) as exc:
                report.errors.append(
                    f"row {idx}: extra_field {var_name!r} path {path!r} unresolvable: {exc}"
                )
                continue
            if val is None or (isinstance(val, str) and not val):
                report.warnings.append(
                    f"row {idx}: extra_field {var_name!r} resolves but is empty/null"
                )


def _is_dict_like(x: Any) -> bool:
    if isinstance(x, dict):
        return True
    # numpy / pyarrow may surface struct fields as objects with item access.
    try:
        _ = x["role"]  # type: ignore[index]
        return True
    except Exception:  # noqa: BLE001
        return False


# ---- CLI -------------------------------------------------------------------


def _parse_extra_field(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=DOTTED.PATH, got {spec!r}"
        )
    name, path = spec.split("=", 1)
    name, path = name.strip(), path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Bad NAME=PATH spec: {spec!r}")
    return name, path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True, type=Path, help="Parquet file to check")
    p.add_argument(
        "--extra-field",
        action="append",
        type=_parse_extra_field,
        default=[],
        metavar="NAME=DOTTED.PATH",
        help="Extra field path to validate (may be repeated). "
        "Mirrors reward.reward_kwargs.judge.extra_fields.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Validate this many randomly sampled rows (default: all rows).",
    )
    args = p.parse_args()

    extra_fields = dict(args.extra_field) if args.extra_field else None
    report = check_dataset(args.parquet, extra_fields=extra_fields, sample_size=args.sample_size)
    print(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

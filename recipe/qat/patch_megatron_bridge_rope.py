"""Patch installed megatron-bridge bridges for transformers 5.x rope_theta.

The pinned megatron-bridge (rc0, e940d99...) reads `hf_config.rope_theta`
directly. transformers 5.x moved that attribute into the
`hf_config.rope_parameters["rope_theta"]` dict, raising AttributeError.

This script rewrites every bridge file in the installed megatron-bridge
package to fall back to `rope_parameters["rope_theta"]` when the direct
attribute is missing.

Idempotent: a marker comment is inserted so re-running is a no-op.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER_SENTINEL = "rope_parameters['rope_theta']"


def patch_file(path: Path) -> bool:
    src = path.read_text()
    if MARKER_SENTINEL in src:
        return False

    # Match `<obj>.rope_theta` where <obj> is hf_config / text_config / self.hf_config / etc.
    # but not method-style names like `_validate_default_rope_parameters`.
    pat = re.compile(r"(\b[\w.]+?)\.rope_theta\b")

    def repl(match: re.Match) -> str:
        obj = match.group(1)
        return (
            f"({obj}.rope_parameters['rope_theta'] "
            f"if hasattr({obj}, 'rope_parameters') and {obj}.rope_parameters "
            f"else {obj}.rope_theta)"
        )

    new_src, n = pat.subn(repl, src)
    if n == 0:
        return False
    path.write_text(new_src)
    print(f"patched {path} ({n} occurrence(s))")
    return True


def main() -> None:
    try:
        import megatron.bridge  # noqa: F401
    except ImportError:
        print("megatron.bridge not installed", file=sys.stderr)
        sys.exit(1)

    pkg_root = Path(megatron.bridge.__file__).parent / "models"
    if not pkg_root.exists():
        print(f"models dir not found: {pkg_root}", file=sys.stderr)
        sys.exit(1)

    patched = 0
    for py in pkg_root.rglob("*_bridge.py"):
        if patch_file(py):
            patched += 1
    # common.py also uses hf_config.rope_theta
    common = pkg_root / "deepseek" / "common.py"
    if common.exists() and patch_file(common):
        patched += 1
    print(f"done: {patched} file(s) patched.")


if __name__ == "__main__":
    main()

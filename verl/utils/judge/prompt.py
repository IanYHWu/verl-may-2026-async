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

"""Tag-based judge prompt template renderer.

We use a sentinel form ``<<name>>`` rather than ``{name}`` (Python format) or
``{{name}}`` (Jinja) because problem statements and rubrics are typically
LaTeX-heavy and frequently contain bare ``{`` / ``}``. The sentinel is unlikely
to appear in math text, and substitution is a plain string replace — no
external template-engine dependency.
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_template(name_or_path: str) -> str:
    """Load template text by short name (looked up in ``templates/``) or by path."""
    p = Path(name_or_path)
    if p.is_file():
        return p.read_text()
    candidate = _TEMPLATE_DIR / f"{name_or_path}.txt"
    if candidate.is_file():
        return candidate.read_text()
    raise FileNotFoundError(
        f"Judge template not found: {name_or_path!r} "
        f"(looked for an absolute path and {candidate})"
    )


def render_template(name_or_path: str, **fields: object) -> str:
    """Render a template, substituting ``<<name>>`` sentinels with str(value)."""
    text = load_template(name_or_path)
    for key, value in fields.items():
        text = text.replace(f"<<{key}>>", str(value))
    return text

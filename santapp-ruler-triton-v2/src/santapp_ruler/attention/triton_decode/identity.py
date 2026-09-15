"""Source-bound gate identity: an old image's success cannot authorize new code."""
from __future__ import annotations

import hashlib
from pathlib import Path


def source_digest() -> str:
    root=Path(__file__).resolve().parents[2]
    h=hashlib.sha256()
    for file in sorted(root.rglob('*.py')):
        h.update(str(file.relative_to(root)).replace('\\','/').encode())
        h.update(b'\0')
        h.update(file.read_bytes())
    return h.hexdigest()

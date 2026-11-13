# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Hash of a tutorial source without its SPDX header, so recordings survive header edits."""

import hashlib
import re
from pathlib import Path

_SPDX_HEADER = re.compile(rb"\A(#![^\n]*\n)?(?:# SPDX-[^\n]*\n)+\n?")


def source_bytes(path) -> bytes:
    """The bytes without a leading SPDX comment block (after any shebang) and one blank line."""
    return _SPDX_HEADER.sub(rb"\1", Path(path).read_bytes())


def source_sha256(path) -> str:
    return hashlib.sha256(source_bytes(path)).hexdigest()

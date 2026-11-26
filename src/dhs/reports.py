"""README findings, one function per topic: `dhs report <topic>` writes charts to out/."""

from collections.abc import Callable
from pathlib import Path

# Topic name -> function(out_dir) returning {label: value}; filled in as each study is completed.
REPORTS: dict[str, Callable[[Path], dict]] = {}

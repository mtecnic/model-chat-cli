"""Run untrusted Python (HumanEval completions) in a limited subprocess.

This is NOT a security sandbox: it isolates the interpreter (-I), uses an
empty temp working directory and environment, and on Linux caps CPU time,
memory and file size. Model code can still touch the network and files the
user can access, which is why the UI asks before running it.
"""
import os
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

TIMEOUT_S = 10
MEM_LIMIT_BYTES = 1024 * 1024 * 1024  # 1 GiB address space
FSIZE_LIMIT_BYTES = 10 * 1024 * 1024


def _limits():  # runs in the child before exec
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S + 1))
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_LIMIT_BYTES, FSIZE_LIMIT_BYTES))
        os.setsid()  # own process group, so Ctrl+C in the TUI doesn't hit it
    except Exception:
        pass


def run_python(source: str, timeout: float = TIMEOUT_S) -> Tuple[bool, Optional[str]]:
    """Run `source`; return (passed, short failure reason)."""
    with tempfile.TemporaryDirectory(prefix="bench_") as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path], cwd=tmp, env={"PATH": "/usr/bin:/bin"},
                stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                preexec_fn=_limits if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode == 0:
        return True, None
    err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
    return False, (err[-1][:120] if err else f"exit {proc.returncode}")

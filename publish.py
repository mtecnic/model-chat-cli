"""GitHub publishing via a local repository.

Copies result files into a local clone, then runs git add/commit/push through
the user's own git installation (auth comes from their existing credentials —
no tokens stored by the app).
"""
import asyncio
import datetime
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


class PublishError(Exception):
    """Raised when a publish step fails."""


@dataclass
class PublishResult:
    ok: bool = True
    dry_run: bool = False
    copied: List[str] = field(default_factory=list)   # rel paths inside the repo
    committed: bool = False
    pushed: bool = False
    message: str = ""
    output: List[str] = field(default_factory=list)   # git stdout/stderr lines

    def summary(self) -> str:
        if not self.ok:
            return self.message or "publish failed"
        if self.dry_run:
            return f"dry-run: would copy {len(self.copied)} file(s) and commit"
        parts = [f"copied {len(self.copied)} file(s)"]
        parts.append("committed" if self.committed else "nothing to commit")
        if self.pushed:
            parts.append("pushed")
        return " · ".join(parts)


class Publisher:
    """Publish result files to a local git repository."""

    def __init__(
        self,
        repo_path: str,
        branch: str = "main",
        prefix: str = "results",
        push: bool = True,
        dry_run: bool = False,
    ):
        self.repo = Path(repo_path).expanduser() if repo_path else None
        self.branch = branch or "main"
        self.prefix = (prefix or "results").strip("/")
        self.push = push
        self.dry_run = dry_run

    # ── Validation ─────────────────────────────────────────────────────────

    def validate(self) -> tuple:
        """Return (ok, description)."""
        if not self.repo:
            return False, "no repository configured"
        if not self.repo.is_dir():
            return False, f"directory not found: {self.repo}"
        try:
            rc, out, err = self._git_sync("rev-parse", "--is-inside-work-tree")
            if rc != 0 or out.strip() != "true":
                return False, f"not a git repository: {self.repo}"
        except FileNotFoundError:
            return False, "git executable not found"
        try:
            rc, out, err = self._git_sync("branch", "--show-current")
            current_branch = out.strip()
        except FileNotFoundError:
            return False, "git executable not found"
        return True, f"ok ({self.repo}) on branch '{current_branch}'"

    # ── Git plumbing ───────────────────────────────────────────────────────

    def _git_sync(self, *args: str) -> tuple:
        """Run a git command synchronously (validation only)."""
        proc = self._run_sync("git", *args)
        return proc.returncode, proc.stdout, proc.stderr

    @staticmethod
    def _run_sync(*args: str):
        import subprocess
        return subprocess.run(args, capture_output=True, text=True, timeout=30)

    async def _git(self, *args: str, timeout: float = 300.0) -> tuple:
        """Run a git command asynchronously, returning (rc, stdout, stderr)."""
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(self.repo),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return 124, "", f"git {' '.join(args)} timed out after {timeout}s"
        return (
            proc.returncode,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )

    async def list_published(self) -> List[str]:
        """Repo-relative paths currently tracked under the results prefix."""
        if not self.repo or not self.repo.is_dir():
            return []
        try:
            rc, out, err = await self._git("ls-files")
        except Exception:
            return []
        if rc != 0:
            return []
        base = f"{self.prefix}/"
        return [line for line in out.splitlines() if line.startswith(base)]

    # ── Publish ────────────────────────────────────────────────────────────

    async def publish(
        self,
        files: List[Path],
        rel_paths: Optional[List[str]] = None,
        commit_message: str = "",
    ) -> PublishResult:
        """Copy files into the repo and commit (+push) them.

        files: absolute source paths. rel_paths: optional target paths inside
        the repo (must align with files). Defaults to <prefix>/<basename>.
        """
        result = PublishResult(dry_run=self.dry_run)
        if not self.repo or not self.repo.is_dir():
            result.ok = False
            result.message = f"repository not found: {self.repo}"
            return result

        rel_paths = rel_paths or [f"{self.prefix}/{p.name}" for p in files]
        if len(files) != len(rel_paths):
            result.ok = False
            result.message = "files and rel_paths must have the same length"
            return result

        if not commit_message:
            commit_message = self._default_message(len(files))

        # Copy
        for src, rel in zip(files, rel_paths):
            src = Path(src).expanduser()
            if not src.exists():
                result.ok = False
                result.message = f"source file missing: {src}"
                return result
            target = self.repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not self.dry_run:
                shutil.copyfile(src, target)
            result.copied.append(rel)

        if self.dry_run:
            result.message = (
                f"[dry-run] git add {len(result.copied)} file(s)\n"
                f"[dry-run] git commit -m \"{commit_message}\"\n"
                + (f"[dry-run] git push origin {self.branch}\n" if self.push else "")
            )
            return result

        # Stage
        rc, out, err = await self._git("add", "--", *rel_paths)
        self._append(result, out, err)
        if rc != 0:
            result.ok = False
            result.message = f"git add failed: {err.strip()}"
            return result

        # Commit
        rc, out, err = await self._git("commit", "-m", commit_message)
        self._append(result, out, err)
        if rc != 0:
            if "nothing to commit" in (out + err).lower():
                result.message = "no changes to commit"
                return result
            result.ok = False
            result.message = f"git commit failed: {err.strip()}"
            return result
        result.committed = True

        # Push
        if self.push:
            rc, out, err = await self._git(
                "push", "origin", self.branch,
                timeout=max(300.0, len(files) * 60),
            )
            self._append(result, out, err)
            if rc != 0:
                result.pushed = False
                result.message = f"commit ok, push failed: {err.strip()}"
                return result
            result.pushed = True

        result.message = "published"
        return result

    @staticmethod
    def _default_message(count: int) -> str:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"results: add {count} file(s) ({ts})"

    @staticmethod
    def _append(result: PublishResult, out: str, err: str) -> None:
        for line in (out or "").splitlines():
            if line.strip():
                result.output.append(line)
        for line in (err or "").splitlines():
            if line.strip():
                result.output.append(f"  {line}")

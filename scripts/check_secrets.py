"""Fail on obvious credential material in tracked text, without printing values.

This lightweight preflight is not a comprehensive secret scanner. It only reads
git-tracked project files, never environment variables or credential stores.
"""
from pathlib import Path
import re
import subprocess

PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{32,}"),
    re.compile(rb"(?i)(?:api_key|api_secret|password|access_token)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"),
]
FORBIDDEN_NAMES = {".env", "credentials", "credentials.json", "token.json", "secrets.toml"}


def main() -> int:
    files = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    rejected = []
    checked = 0
    for name in filter(None, files):
        path = Path(name)
        if not path.is_file():
            continue
        checked += 1
        forbidden = path.name in FORBIDDEN_NAMES or path.suffix in {".pem", ".key", ".p12"}
        forbidden |= bool(set(path.parts) & {".aws", ".ssh", ".venv", ".codex"})
        # Never echo matching values; report relative filenames only.
        if forbidden or any(pattern.search(path.read_bytes()) for pattern in PATTERNS):
            rejected.append(name)
    if rejected:
        print("Credential preflight rejected these paths:")
        for name in rejected:
            print(f"  {name}")
        return 1
    print(f"Credential preflight passed for {checked} tracked files (heuristic check).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

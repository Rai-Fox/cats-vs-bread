import subprocess


def git_commit_id() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

import subprocess

from cats_vs_bread.utils.logging_utils import get_logger

logger = get_logger(__name__)


def git_commit_id() -> str:
    logger.info("Retrieving current Git commit ID (git rev-parse HEAD).")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

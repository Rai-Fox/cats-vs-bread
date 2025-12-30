import subprocess
from pathlib import Path

from cats_vs_bread.configs import DataConfig
from cats_vs_bread.utils.logging_utils import get_logger

logger = get_logger(__name__)


def dvc_pull(targets: list[Path] | None = None, remote: str | None = None) -> None:
    cmd = ["dvc", "pull"]
    if remote:
        cmd.extend(["-r", remote])
    if targets:
        cmd.extend([f"{target}.dvc" for target in targets])
    logger.info(f"Running DVC pull command: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def unpack_tar_archive(archive_path: Path, target_dir: Path, force: bool = False) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if force:
        logger.info(f"Removing existing data in {target_dir} due to force_extract=True.")
        subprocess.check_call(["rm", "-rf", str(target_dir)])
    is_empty = not list(target_dir.rglob("*.jpeg"))
    if is_empty:
        logger.info(f"Unpacking archive {archive_path} to {target_dir.parent}.")
        subprocess.check_call(["tar", "-xzf", str(archive_path), "-C", str(target_dir.parent)])


def pull_and_unpack_data(data_config: DataConfig) -> None:
    if not data_config.force_extract and data_config.train_archive.exists() and data_config.val_archive.exists():
        logger.info("Data archives already exist and force_extract is False. Skipping DVC pull and unpacking.")
    else:
        dvc_pull(
            targets=[data_config.train_archive, data_config.val_archive],
            remote=data_config.dvc_remote,
        )

    if not data_config.force_extract and data_config.train_dir.exists() and data_config.val_dir.exists():
        logger.info("Data directories already exist and force_extract is False. Skipping unpacking.")
    else:
        unpack_tar_archive(data_config.train_archive, data_config.train_dir, force=data_config.force_extract)
        unpack_tar_archive(data_config.val_archive, data_config.val_dir, force=data_config.force_extract)

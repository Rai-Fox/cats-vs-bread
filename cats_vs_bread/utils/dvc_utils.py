import subprocess
from pathlib import Path

from cats_vs_bread.configs import DataConfig


def dvc_pull(targets: list[Path] | None = None, remote: str | None = None) -> None:
    cmd = ["dvc", "pull"]
    if remote:
        cmd.extend(["-r", remote])
    if targets:
        cmd.extend([str(t) for t in targets])
    subprocess.check_call(cmd)


def unpack_tar_archive(archive_path: Path, target_dir: Path, force: bool = False) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if force:
        subprocess.check_call(["rm", "-rf", str(target_dir)])
    is_empty = not list(target_dir.rglob("*.jpeg"))
    if is_empty:
        subprocess.check_call(["tar", "-xzf", str(archive_path), "-C", str(target_dir)])


def pull_and_unpack_data(data_config: DataConfig) -> None:
    dvc_pull(
        targets=[data_config.train_archive, data_config.val_archive],
        remote=data_config.dvc_remote,
    )
    unpack_tar_archive(data_config.train_archive, data_config.train_dir, force=data_config.force_extract)
    unpack_tar_archive(data_config.val_archive, data_config.val_dir, force=data_config.force_extract)

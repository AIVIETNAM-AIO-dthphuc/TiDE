"""Sao luu / khoi phuc checkpoint TiDE giua Kaggle va Google Drive bang rclone.

Yeu cau:
    - rclone da cai dat trong moi truong (Kaggle Notebook).
    - rclone.conf da duoc khoi phuc tu Kaggle Secrets (KHONG luu token vao file nay).
    - Remote mac dinh ten 'gdrive'.

Duong dan mac dinh tren Drive:
    [54] IS403 - BA/[1] PROJECT/[IS403]_Experiment/
        {dataset}/seed_{seed}/H{horizon}/checkpoints/snapshots/{snapshot_id}/

Moi snapshot la mot thu muc bat bien va chi duoc coi la hoan tat neu co
file danh dau '_BACKUP_COMPLETE.json' (duoc upload SAU cung). Do do, neu
Kaggle bi ngat giua luc upload, restore se bo qua snapshot dang do.

Khong goi ham backup khi framework CHUA luu xong checkpoint vao o dia.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union


DEFAULT_REMOTE_BASE = (
    "gdrive:[54] IS403 - BA/[1] PROJECT/[IS403]_Experiment"
)
DEFAULT_LOCAL_BASE = Path("/kaggle/working/IS403_Experiment")
DATASETS = frozenset({
    "electricity", "traffic", "weather", "ettm1", "ettm2", "etth1", "etth2"
})
HORIZONS = frozenset({96, 192, 336, 720})
COMPLETE_MARKER = "_BACKUP_COMPLETE.json"
SNAPSHOT_RE = re.compile(r"^epoch_(\d{6,})_(\d{8}T\d{6}Z)_([a-f0-9]{8})$")
PathLike = Union[str, Path]


class RcloneBackupError(RuntimeError):
    """Loi chay rclone hoac loi kiem tra checkpoint."""


class DriveBackup:
    """Quan ly checkpoint va ket qua theo (dataset, seed, horizon).

    Moi doi tuong dai dien cho mot to hop thuc nghiem, vi du:
        DriveBackup("electricity", seed=1, horizon=336)

    Neu muon chay doc lap cung dataset/seed/horizon tu dau, truyen run_id
    khac nhau de tach rieng cac snapshot (neu bo trong thi coi la resume).
    """

    def __init__(
        self,
        dataset: str,
        seed: int,
        horizon: int,
        *,
        local_base: PathLike = DEFAULT_LOCAL_BASE,
        remote_base: str = DEFAULT_REMOTE_BASE,
        run_id: Optional[str] = None,
        rclone_executable: str = "rclone",
        verify: bool = True,
    ) -> None:
        dataset = dataset.lower()
        if dataset not in DATASETS:
            raise ValueError(f"Dataset khong hop le: {dataset}. Hop le: {sorted(DATASETS)}")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0:
            raise ValueError("seed phai la so nguyen duong.")
        if horizon not in HORIZONS:
            raise ValueError(f"horizon phai thuoc {sorted(HORIZONS)}")
        if run_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
            raise ValueError("run_id chi gom chu, so, _ hoac -, toi da 64 ky tu.")
        if ":" not in remote_base.split("/", 1)[0]:
            raise ValueError("remote_base phai bat dau bang remote rclone, vi du 'gdrive:...'.")
        if not shutil.which(rclone_executable):
            raise FileNotFoundError(
                f"Khong tim thay '{rclone_executable}'. Cai rclone truoc khi chay."
            )

        self.dataset = dataset
        self.seed = seed
        self.horizon = horizon
        self.run_id = run_id
        self.local_base = Path(local_base)
        self.remote_base = remote_base.rstrip("/")
        self.rclone_executable = rclone_executable
        self.verify = verify

        self.relative_run = f"{dataset}/seed_{seed}/H{horizon}"
        self.local_run = self.local_base / dataset / f"seed_{seed}" / f"H{horizon}"
        self.remote_run = f"{self.remote_base}/{self.relative_run}"
        self.local_checkpoints = self.local_run / "checkpoints"
        self.local_results = self.local_run / "results"
        self.remote_results = f"{self.remote_run}/results"

        # Mac dinh: checkpoints/snapshots/epoch_.../
        # Neu co run_id: checkpoints/runs/<run_id>/snapshots/epoch_.../
        relative_snapshot_root = "checkpoints"
        if run_id:
            relative_snapshot_root += f"/runs/{run_id}"
        self.remote_snapshots = f"{self.remote_run}/{relative_snapshot_root}/snapshots"

    def _rclone(self, *args: str) -> str:
        """Chay rclone an toan (khong shell=True), bao loi neu that bai."""
        cmd = [self.rclone_executable, *map(str, args)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            raise RcloneBackupError(
                f"rclone {args[0]} that bai (exit {result.returncode}): "
                f"{details[-2000:]}"
            )
        return result.stdout

    def check_connection(self) -> None:
        """Kiem tra remote va duong dan goc tren Drive co the doc duoc."""
        self._rclone("lsf", self.remote_base, "--max-depth", "1")

    @staticmethod
    def _has_files(folder: Path) -> bool:
        return folder.is_dir() and any(p.is_file() for p in folder.rglob("*"))

    def backup_checkpoint(self, epoch: int, checkpoint_dir: Optional[PathLike] = None) -> str:
        """Upload checkpoint da luu xong, tra ve duong dan snapshot tren Drive.

        Ghi nho: Phai goi SAU tf.train.CheckpointManager.save() (hoac save tuong duong).
        Snapshot chi duoc danh dau completed sau khi copy va verify thanh cong.
        """
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch phai la so nguyen khong am.")
        source = Path(checkpoint_dir) if checkpoint_dir else self.local_checkpoints
        if not self._has_files(source):
            raise FileNotFoundError(f"Khong co file checkpoint de upload: {source}")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_id = f"epoch_{epoch:06d}_{timestamp}_{uuid.uuid4().hex[:8]}"
        remote_snapshot = f"{self.remote_snapshots}/{snapshot_id}"

        # Chi tao 1 snapshot moi, khong ghi de snapshot cu va khong xoa file tren Drive.
        self._rclone("copy", str(source), remote_snapshot, "--transfers", "2")
        if self.verify:
            self._rclone("check", str(source), remote_snapshot)

        # Marker duoc upload cuoi cung, khong co marker thi restore se bo qua.
        metadata = {
            "format_version": 1,
            "dataset": self.dataset,
            "seed": self.seed,
            "horizon": self.horizon,
            "run_id": self.run_id,
            "epoch": epoch,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_id": snapshot_id,
        }
        with tempfile.TemporaryDirectory(prefix="tide_backup_marker_") as td:
            marker = Path(td) / COMPLETE_MARKER
            marker.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self._rclone("copyto", str(marker), f"{remote_snapshot}/{COMPLETE_MARKER}")

        print(f"[drive_backup] Backup OK (epoch {epoch}): {remote_snapshot}")
        return remote_snapshot

    @staticmethod
    def _is_missing_folder_error(error: RcloneBackupError) -> bool:
        message = str(error).lower()
        return any(phrase in message for phrase in (
            "directory not found", "directory doesn't exist", "directory does not exist",
            "couldn't find root directory", "error 404: not found",
            "path not found", "not a directory", "file not found",
        ))

    def list_completed_snapshots(self) -> list[str]:
        """Danh sach snapshot da hoan tat, tu cu den moi theo epoch va timestamp."""
        try:
            raw = self._rclone("lsf", self.remote_snapshots, "--dirs-only")
        except RcloneBackupError as error:
            if self._is_missing_folder_error(error):
                return []
            raise

        candidates = []
        for line in raw.splitlines():
            name = line.strip().rstrip("/")
            if SNAPSHOT_RE.fullmatch(name):
                candidates.append(name)

        completed = []
        # Quet tu moi den cu, nhung chi nhan snapshot co marker hoan tat.
        for name in sorted(candidates, reverse=True):
            path = f"{self.remote_snapshots}/{name}"
            listed = self._rclone("lsf", path, "--files-only")
            if COMPLETE_MARKER in listed.splitlines():
                completed.append(name)
        return list(reversed(completed))

    def restore_latest_checkpoint(
        self, checkpoint_dir: Optional[PathLike] = None
    ) -> Optional[dict]:
        """Khoi phuc snapshot moi nhat vao mot thu muc rong.

        Tra ve metadata dict neu restore, hoac None neu chua co checkpoint.
        Khong ghi de thu muc co du lieu de tranh mat/tron lan checkpoint.
        """
        target = Path(checkpoint_dir) if checkpoint_dir else self.local_checkpoints
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError(
                f"Thu muc dich khong rong: {target}. "
                "Chi restore vao thu muc moi/rong de tranh tron checkpoint."
            )

        completed = self.list_completed_snapshots()
        if not completed:
            print("[drive_backup] Chua co checkpoint hoan tat tren Google Drive.")
            return None

        latest = completed[-1]
        remote_snapshot = f"{self.remote_snapshots}/{latest}"
        metadata_str = self._rclone("cat", f"{remote_snapshot}/{COMPLETE_MARKER}")
        metadata = json.loads(metadata_str)
        if (
            metadata.get("snapshot_id") != latest
            or metadata.get("dataset") != self.dataset
            or metadata.get("seed") != self.seed
            or metadata.get("horizon") != self.horizon
            or metadata.get("run_id") != self.run_id
        ):
            raise RcloneBackupError("Metadata snapshot khong khop voi cau hinh run.")

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="tide_restore_", dir=target.parent))
        try:
            self._rclone(
                "copy", remote_snapshot, str(staging),
                "--exclude", COMPLETE_MARKER,
                "--transfers", "2",
            )
            if self.verify:
                self._rclone(
                    "check", remote_snapshot, str(staging),
                    "--exclude", COMPLETE_MARKER,
                )
            if not self._has_files(staging):
                raise RcloneBackupError("Snapshot restore khong co file checkpoint.")

            # Chuyen vao vi tri dich SAU khi download/verify thanh cong.
            if target.exists():
                target.rmdir()  # Chi thu muc rong moi duoc phep xoa.
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        print(f"[drive_backup] Restore OK: {latest} -> {target}")
        return metadata

    def upload_results(self, results_dir: Optional[PathLike] = None) -> None:
        """Upload ket qua ra results/ (khong xoa file cu tren Drive)."""
        source = Path(results_dir) if results_dir else self.local_results
        if not self._has_files(source):
            raise FileNotFoundError(f"Khong co file results de upload: {source}")
        self._rclone("copy", str(source), self.remote_results)
        if self.verify:
            # --one-way: neu Drive da co results cu thi khong xem la loi.
            self._rclone("check", str(source), self.remote_results, "--one-way")
        print(f"[drive_backup] Results upload OK: {self.remote_results}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sao luu / khoi phuc TiDE qua rclone")
    parser.add_argument("action", choices=["check", "backup", "restore", "results"])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--horizon", required=True, type=int, choices=sorted(HORIZONS))
    parser.add_argument("--epoch", type=int, help="Bat buoc voi action=backup")
    parser.add_argument("--run-id", help="Dinh danh lan chay doc lap")
    parser.add_argument("--checkpoint-dir", help="Thu muc chua checkpoint local")
    parser.add_argument("--results-dir", help="Thu muc chua results local")
    parser.add_argument("--local-base", default=str(DEFAULT_LOCAL_BASE))
    parser.add_argument("--remote-base", default=DEFAULT_REMOTE_BASE)
    parser.add_argument("--no-verify", action="store_true", help="Bo qua rclone check")
    args = parser.parse_args()

    backup = DriveBackup(
        args.dataset, args.seed, args.horizon,
        run_id=args.run_id, local_base=args.local_base,
        remote_base=args.remote_base, verify=not args.no_verify,
    )
    if args.action == "check":
        backup.check_connection()
        print(f"[drive_backup] Remote OK: {backup.remote_base}")
    elif args.action == "backup":
        if args.epoch is None:
            parser.error("--epoch la tham so bat buoc khi action=backup")
        backup.backup_checkpoint(args.epoch, args.checkpoint_dir)
    elif args.action == "restore":
        info = backup.restore_latest_checkpoint(args.checkpoint_dir)
        if info is not None:
            print(f"[drive_backup] Resume sau epoch {info['epoch']}")
    elif args.action == "results":
        backup.upload_results(args.results_dir)


if __name__ == "__main__":
    main()

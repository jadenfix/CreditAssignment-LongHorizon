"""Upload a local directory to GCS. Used at the end of a training run.

Wired into ``cts.train.main`` via the ``--gcs-bucket`` flag and into
``deploy/launch_tpu.sh`` for post-run sync. Falls back to ``gsutil -m rsync``
if ``google-cloud-storage`` isn't installed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def upload(local_dir: str, gcs_uri: str) -> None:
    """Recursively upload ``local_dir`` → ``gcs_uri`` (must start with ``gs://``)."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"gcs_uri must start with gs://, got {gcs_uri!r}")
    src = Path(local_dir)
    if not src.exists():
        raise FileNotFoundError(src)

    try:
        from google.cloud import storage  # type: ignore[import-not-found]
    except ImportError:
        if shutil.which("gsutil") is None:
            raise SystemExit(
                "Need google-cloud-storage (`uv pip install .[gcp]`) or gsutil on PATH."
            ) from None
        subprocess.run(["gsutil", "-m", "rsync", "-r", str(src), gcs_uri], check=True)
        return

    bucket_name, _, prefix = gcs_uri[len("gs://") :].partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    n = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        blob = bucket.blob(f"{prefix.rstrip('/')}/{rel}" if prefix else rel)
        blob.upload_from_filename(str(path))
        n += 1
    print(f"[gcs_sync] uploaded {n} files to {gcs_uri}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("local_dir")
    ap.add_argument("gcs_uri")
    args = ap.parse_args()
    upload(args.local_dir, args.gcs_uri)


if __name__ == "__main__":
    main()

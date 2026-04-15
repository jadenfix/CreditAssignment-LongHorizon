"""Submit the paper matrix as a Vertex AI CustomJob with a TPU worker pool.

Use when you want managed retries, Tensorboard, and structured logs instead of
the bare ``gcloud ssh`` path in ``launch_tpu.sh``.

Usage::

    python deploy/launch_vertex.py \\
        --project my-proj --region us-central1 \\
        --image us-central1-docker.pkg.dev/my-proj/cts/tpu:latest \\
        --bucket gs://my-cts-runs --tpu-type v5litepod-8 \\
        -- --tasks both --seeds 5 --max-steps 2000

Args after ``--`` are forwarded to ``scripts/run_paper_matrix.py``.
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--image", required=True)
    ap.add_argument("--bucket", required=True, help="gs:// prefix for results + staging")
    ap.add_argument("--tpu-type", default="v5litepod-8")
    ap.add_argument("--display-name", default=f"cts-matrix-{int(time.time())}")
    args, matrix_args = ap.parse_known_args()
    if matrix_args and matrix_args[0] == "--":
        matrix_args = matrix_args[1:]

    try:
        from google.cloud import aiplatform
    except ImportError:
        sys.exit("Install with `uv pip install -e .[gcp]` first.")

    aiplatform.init(project=args.project, location=args.region, staging_bucket=args.bucket)

    cmd = [
        "python",
        "scripts/run_paper_matrix.py",
        "--backend",
        "tunix",
        "--out",
        f"{args.bucket}/runs/{args.display_name}/",
        *matrix_args,
    ]

    job = aiplatform.CustomJob(
        display_name=args.display_name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "cloud-tpu",
                    "accelerator_type": args.tpu_type,
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {"image_uri": args.image, "command": cmd},
            }
        ],
        base_output_dir=f"{args.bucket}/vertex/{args.display_name}",
    )
    job.run(sync=False)
    print(f"Submitted: {job.resource_name}")
    print(
        f"Console: https://console.cloud.google.com/vertex-ai/locations/"
        f"{args.region}/training/{job.name}?project={args.project}"
    )


if __name__ == "__main__":
    main()

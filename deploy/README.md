# Deploying CTS to GCP

Five commands. Nothing here runs automatically — invoke when you're ready.

## 0. One-time setup

```bash
gcloud auth login
gcloud config set project $PROJECT
gcloud artifacts repositories create cts \
  --repository-format=docker --location=us-central1
gsutil mb gs://my-cts-runs
```

## 1. Build & push images

```bash
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions=_TAG=$(git rev-parse --short HEAD) .
```

Pushes both `…/cts/cpu:<sha>` and `…/cts/tpu:<sha>`.

## 2. Launch the paper matrix on a TPU

```bash
PROJECT=my-proj ZONE=us-east5-a TPU_NAME=cts-run-1 \
IMAGE_TAG=$(git rev-parse --short HEAD) GCS_BUCKET=gs://my-cts-runs \
TPU_TYPE=v5p-8 bash deploy/launch_tpu.sh
```

Use `DRY_RUN=1` to print the `gcloud` invocations without executing.

Alternative (Vertex AI, with managed retries + Tensorboard):

```bash
uv run python deploy/launch_vertex.py \
  --project $PROJECT --region us-central1 \
  --image us-central1-docker.pkg.dev/$PROJECT/cts/tpu:latest \
  --bucket gs://my-cts-runs --tpu-type v5litepod-8 \
  -- --tasks both --seeds 5 --max-steps 2000
```

## 3. Pull & aggregate results

```bash
gsutil -m rsync -r gs://my-cts-runs/runs/cts-run-1/ artifacts/results/cts-run-1/
uv run python scripts/aggregate_results.py artifacts/results/cts-run-1/ \
  --base B3 --out artifacts/results/cts-run-1/table.md
```

## 4. Tear down

```bash
gcloud alpha compute tpus queued-resources delete cts-run-1 \
  --project=$PROJECT --zone=$ZONE --force
```

## Notes

- No Terraform / GKE / secrets manager intentionally. A single SA + manual
  `gcloud auth` is enough for this research repo.
- The TPU image base pin (`jax0.4.34-rev1`) must match the Tunix `[scale]`
  extra. Bump them together.
- Algorithm code never lives under `deploy/`; this dir is launchers + Dockerfiles
  only.

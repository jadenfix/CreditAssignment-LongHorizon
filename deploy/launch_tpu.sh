#!/usr/bin/env bash
# Reserve a Cloud TPU queued resource and run the paper matrix on it.
# Required env: PROJECT, ZONE, TPU_NAME, IMAGE_TAG, GCS_BUCKET.
# Optional: TPU_TYPE (default v5p-8), RUNTIME (default v2-alpha-tpuv5),
#           MATRIX_ARGS (default --tasks both --seeds 5 --max-steps 2000).
set -euo pipefail

: "${PROJECT:?set PROJECT}"
: "${ZONE:?set ZONE, e.g. us-east5-a}"
: "${TPU_NAME:?set TPU_NAME}"
: "${IMAGE_TAG:?set IMAGE_TAG (the SHORT_SHA pushed by cloudbuild)}"
: "${GCS_BUCKET:?set GCS_BUCKET, e.g. gs://my-cts-runs}"

TPU_TYPE=${TPU_TYPE:-v5p-8}
RUNTIME=${RUNTIME:-v2-alpha-tpuv5}
REGION=${REGION:-us-central1}
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/cts/tpu:${IMAGE_TAG}"
MATRIX_ARGS=${MATRIX_ARGS:---backend tunix --tasks both --seeds 5 --max-steps 2000}
DRY=${DRY_RUN:-0}

run() {
  echo "+ $*"
  if [[ "$DRY" != "1" ]]; then "$@"; fi
}

run gcloud alpha compute tpus queued-resources create "${TPU_NAME}" \
  --node-id="${TPU_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --accelerator-type="${TPU_TYPE}" \
  --runtime-version="${RUNTIME}"

# Wait for ACTIVE.
if [[ "$DRY" != "1" ]]; then
  until gcloud alpha compute tpus queued-resources describe "${TPU_NAME}" \
        --project="${PROJECT}" --zone="${ZONE}" \
        --format='value(state.state)' | grep -q ACTIVE; do
    echo "waiting for TPU…"; sleep 20
  done
fi

REMOTE_CMD="docker run --rm --privileged --net=host \
  -e GCS_BUCKET=${GCS_BUCKET} \
  -e RUN_ID=${TPU_NAME} \
  ${IMAGE} \
  python scripts/run_paper_matrix.py ${MATRIX_ARGS} \
    --out /tmp/results && \
  python deploy/gcs_sync.py /tmp/results ${GCS_BUCKET}/runs/${TPU_NAME}/"

run gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" --command="${REMOTE_CMD}"

echo
echo "Results: ${GCS_BUCKET}/runs/${TPU_NAME}/"
echo "To delete the queued resource:"
echo "  gcloud alpha compute tpus queued-resources delete ${TPU_NAME} --project=${PROJECT} --zone=${ZONE} --force"

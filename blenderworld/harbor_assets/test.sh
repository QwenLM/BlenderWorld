#!/bin/bash
set -euo pipefail
exec python -m blenderworld.harbor_verify \
    --dataset "${BW_HARBOR_DATASET:-/tests/dataset}" \
    --submission "${BW_HARBOR_SUBMISSION:-/workspace/exports}" \
    --output "${BW_HARBOR_OUTPUT:-/logs/verifier}"

#!/usr/bin/env bash

set -Eeuo pipefail

image_name="bank-marketing-api:ci"
container_name="bank-marketing-api-smoke-${RANDOM}-$$"
port="${SMOKE_TEST_PORT:-8000}"

cleanup() {
  docker logs "${container_name}" 2>/dev/null || true
  docker rm --force "${container_name}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

docker build --tag "${image_name}" .
docker run \
  --detach \
  --name "${container_name}" \
  --publish "${port}:8000" \
  "${image_name}" >/dev/null

for attempt in {1..30}; do
  if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; then
    break
  fi

  if [[ "${attempt}" == "30" ]]; then
    echo "API did not become ready in time." >&2
    exit 1
  fi

  sleep 2
done

curl --fail --silent --show-error \
  --request POST \
  --header "Content-Type: application/json" \
  --data '{
    "age": 42,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1850,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 320,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }' \
  "http://127.0.0.1:${port}/predict"

echo
echo "Docker smoke test passed."

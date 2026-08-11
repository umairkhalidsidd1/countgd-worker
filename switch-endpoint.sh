#!/usr/bin/env bash
# Switch a RunPod serverless endpoint between the two templates and measure.
#
# Required env: RUNPOD_ENDPOINT_ID, RUNPOD_OLD_TEMPLATE, RUNPOD_NEW_TEMPLATE,
# RUNPOD_VOLUME_ID (rollback only). The API key is read from ~/.runpod-key.
#
#   ./switch-endpoint.sh baked     -> the new self-contained image, no volume
#   ./switch-endpoint.sh rollback  -> the original volume-based template
#   ./switch-endpoint.sh measure   -> time a genuinely cold scan
#
# The baked image is PRIVATE until the ghcr package is made public (or RunPod is
# given a read:packages credential). Switching before then leaves workers unable
# to pull, so `baked` refuses to run until the image is reachable anonymously.
set -euo pipefail
EP=${RUNPOD_ENDPOINT_ID:?set RUNPOD_ENDPOINT_ID}
OLD_TEMPLATE=${RUNPOD_OLD_TEMPLATE:?set RUNPOD_OLD_TEMPLATE}   # runpod/pytorch + network volume
NEW_TEMPLATE=${RUNPOD_NEW_TEMPLATE:?set RUNPOD_NEW_TEMPLATE}   # the baked image template
IMAGE=ghcr.io/umairkhalidsidd1/countgd:latest
KEY=$(tr -d '\n' < ~/.runpod-key)

api() { # api <METHOD> <path> [json] — fails loudly on a non-2xx
  local out code
  out=$(curl -sS -X "$1" "https://rest.runpod.io/v1$2" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -H 'User-Agent: curl/8.4.0' ${3:+-d "$3"} -w '\n%{http_code}')
  code=${out##*$'\n'}
  out=${out%$'\n'*}
  if [ "$code" -lt 200 ] || [ "$code" -ge 300 ]; then
    echo "API $1 $2 -> HTTP $code: ${out:0:300}" >&2
    return 1
  fi
  printf '%s' "$out"
}

pullable() { # is the image anonymously pullable?
  local tok
  tok=$(curl -sS "https://ghcr.io/token?scope=repository:umairkhalidsidd1/countgd:pull&service=ghcr.io" \
        | python3 -c 'import json,sys;print(json.load(sys.stdin).get("token",""))' 2>/dev/null || true)
  [ -n "$tok" ] || return 1
  curl -sS -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $tok" \
    -H 'Accept: application/vnd.oci.image.manifest.v1+json,application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.v2+json' \
    "https://ghcr.io/v2/umairkhalidsidd1/countgd/manifests/latest" | grep -q '^200$'
}

case "${1:-}" in
  baked)
    if ! pullable; then
      echo "REFUSING: $IMAGE is not anonymously pullable yet." >&2
      echo "Make the ghcr package public, or add a read:packages credential to RunPod." >&2
      exit 1
    fi
    # Separate calls on purpose: sending both keys with networkVolumeId null is
    # rejected (HTTP 400), and "" is the only value that actually detaches it.
    api PATCH "/endpoints/$EP" "{\"templateId\":\"$NEW_TEMPLATE\"}" >/dev/null
    api PATCH "/endpoints/$EP" '{"networkVolumeId":""}' >/dev/null
    echo "switched to the baked image (no network volume)"
    ;;
  rollback)
    api PATCH "/endpoints/$EP" "{\"templateId\":\"$OLD_TEMPLATE\",\"networkVolumeId\":\"${RUNPOD_VOLUME_ID:?set RUNPOD_VOLUME_ID}\"}" >/dev/null
    echo "rolled back to the volume-based template"
    ;;
  measure)
    api GET "/endpoints/$EP" | python3 -c '
import json,sys; d=json.load(sys.stdin)
print("  template:", d.get("templateId"), "volume:", d.get("networkVolumeId"),
      "idle:", d.get("idleTimeout"), "workers:", d.get("workersMin"), "-", d.get("workersMax"))'
    ;;
  *) echo "usage: $0 {baked|rollback|measure}" >&2; exit 2;;
esac

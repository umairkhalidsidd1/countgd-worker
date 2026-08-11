#!/usr/bin/env bash
# Switch RunPod endpoint oczn63esmtysqm between the two templates and measure.
#
#   ./switch-endpoint.sh baked     -> the new self-contained image, no volume
#   ./switch-endpoint.sh rollback  -> the original volume-based template
#   ./switch-endpoint.sh measure   -> time a genuinely cold scan
#
# The baked image is PRIVATE until the ghcr package is made public (or RunPod is
# given a read:packages credential). Switching before then leaves workers unable
# to pull, so `baked` refuses to run until the image is reachable anonymously.
set -euo pipefail
EP=oczn63esmtysqm
OLD_TEMPLATE=nakg84aofh      # runpod/pytorch + network volume 0we7ub4z01
NEW_TEMPLATE=kfpdzxd8r6      # ghcr.io/umairkhalidsidd1/countgd-worker:latest
IMAGE=ghcr.io/umairkhalidsidd1/countgd-worker:latest
KEY=$(tr -d '\n' < ~/.runpod-key)

api() { # api <METHOD> <path> [json]
  curl -sS -X "$1" "https://rest.runpod.io/v1$2" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -H 'User-Agent: curl/8.4.0' ${3:+-d "$3"}
}

pullable() { # is the image anonymously pullable?
  local tok
  tok=$(curl -sS "https://ghcr.io/token?scope=repository:umairkhalidsidd1/countgd-worker:pull&service=ghcr.io" \
        | python3 -c 'import json,sys;print(json.load(sys.stdin).get("token",""))' 2>/dev/null || true)
  [ -n "$tok" ] || return 1
  curl -sS -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $tok" \
    -H 'Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json' \
    "https://ghcr.io/v2/umairkhalidsidd1/countgd-worker/manifests/latest" | grep -q '^200$'
}

case "${1:-}" in
  baked)
    if ! pullable; then
      echo "REFUSING: $IMAGE is not anonymously pullable yet." >&2
      echo "Make the ghcr package public, or add a read:packages credential to RunPod." >&2
      exit 1
    fi
    api PATCH "/endpoints/$EP" "{\"templateId\":\"$NEW_TEMPLATE\",\"networkVolumeId\":null}" >/dev/null
    echo "switched to the baked image (no network volume)"
    ;;
  rollback)
    api PATCH "/endpoints/$EP" "{\"templateId\":\"$OLD_TEMPLATE\",\"networkVolumeId\":\"0we7ub4z01\"}" >/dev/null
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

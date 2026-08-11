# countgd-worker

Serverless worker image for the CountThis counting backend (a RunPod serverless endpoint).

## Why this repo exists

The endpoint used to run a generic `runpod/pytorch` image with the venv, the code
and 2.6GB of weights all living on a RunPod **network volume**. Two things
followed from that:

1. Every cold start imported torch and loaded the model **over network storage**.
2. The volume pinned the endpoint to a single datacenter (EU-RO-1), so workers sat
   `throttled` waiting for a GPU in that one region.

Measured on a real job before the change:

| | |
|---|---|
| `delayTime` (startup) | 248.9 s |
| `executionTime` (actual counting) | 6.0 s |

98% of the wait was startup. This image bakes the venv, the code and the weights
in, so all of it is on the worker's local NVMe and the endpoint no longer needs a
volume — which also lets it schedule in any region.

## Layout

- `Dockerfile` — clones the public HF space `nikigoli/countgd` for the weights,
  then applies `overlay/` on top and installs the pinned dependency set.
- `overlay/` — the locally modified sources, taken from the volume that has served
  every successful job to date. `app.py` carries a no-op `spaces = _S()` shim so
  the HF `@spaces.GPU` decorators are inert off-platform; `rp_handler.py` is the
  RunPod entry point.
- `freeze-reference.txt` — `pip freeze` of the proven venv. The Dockerfile's pins
  are transcribed from it.

## Pins that are not negotiable

- `torch==2.1.1+cu121` / `torchvision==0.16.1+cu121` come from the base image.
  Anything newer makes CountGD silently return **zero boxes**.
- `transformers==4.44.2` — newer raises `register_pytree_node` AttributeError.
- `numpy==1.26.4` — numpy 2 makes torch 2.1 report "Numpy is not available" at
  inference time.

Both the dependency set and `import app` are verified during the build, so a bad
pin fails the build rather than showing up as a 250-second timeout in the app.

## Contract

Input `{"input": {"image": "<bare base64, no data: prefix>", "prompt": "<text>",
"return_image": false}}` → output `{count, points[[x,y]…], width, height, prompt}`.

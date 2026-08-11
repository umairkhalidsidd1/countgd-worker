# CountGD serverless worker — everything baked in.
#
# Why this image exists: the endpoint used to run a generic PyTorch image and keep
# the venv, the code and 2.6GB of weights on a RunPod network volume. Every cold
# start therefore imported torch and loaded the model over network storage, and the
# volume pinned the endpoint to one datacenter so workers spent long stretches
# throttled waiting for a GPU there. Measured on a real job: delayTime 248.9s,
# executionTime 6.0s — 98% of the wait was startup, none of it counting.
#
# Baking the venv, the code and the weights into the image puts all of it on the
# worker's local NVMe and frees the endpoint from the volume, so it can schedule in
# any region.
#
# Base is deliberately the SAME image the endpoint already pulls today
# (runpod/pytorch 2.1.1 / py3.10 / cu121). RunPod hosts cache their own images, so
# only the layers added below have to travel on a cold pull. It also guarantees the
# torch build stays 2.1.1+cu121 — anything newer makes CountGD silently return
# zero boxes.
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # matplotlib writes a cache on first import; without a writable dir it warns
    # and stalls on a read-only filesystem.
    MPLCONFIGDIR=/tmp/.mplcache \
    # Keep every HF/torch cache inside the image so nothing is fetched at runtime.
    HF_HOME=/app/.hf \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The pristine space, weights included. checkpoint_best_regular.pth (1.25GB),
# checkpoints/groundingdino_swinb_cogcoor.pth (938MB) and a LOCAL
# checkpoints/bert-base-uncased (436MB) all arrive here. The local BERT is what
# lets TRANSFORMERS_OFFLINE=1 hold — the text encoder never calls HuggingFace at
# boot, which is worth several seconds of every cold start and removes a runtime
# dependency on huggingface.co being up.
RUN git clone --depth 1 https://huggingface.co/spaces/nikigoli/countgd /app \
    && git lfs pull \
    && rm -rf /app/.git

# The locally modified sources on top of the pristine checkout. app.py matters
# most: it carries a no-op `spaces = _S()` shim so the HF @spaces.GPU decorators
# are inert off-platform, plus rp_handler.py itself.
COPY overlay/ /app/

# Pins are transcribed from a pip freeze of the venv that has served every
# successful job to date. torch/torchvision come from the base image and must NOT
# be touched. transformers 4.44.2 avoids a register_pytree_node AttributeError and
# numpy must stay on 1.x or torch 2.1 reports "Numpy is not available" at
# inference — both were live failures, not theory.
#
# Every version below is transcribed from that freeze. Note what is deliberately
# ABSENT: seaborn, jsonlines, emoji, supervision and ipdb all appear in `import`
# statements inside the checkout but are not installed in the working venv — proof
# those modules are never reached from the handler, so adding them would be
# guesswork rather than reproduction.
#
# opencv is also absent, and that is not an oversight. The venv lists
# opencv-python==5.0.0.93, which requires numpy>=2 — impossible next to the
# numpy 1.26.4 that torch 2.1 demands, so `import cv2` cannot ever have succeeded
# there. It only appears in util/visualizer.py, util/vis_utils.py and the
# groundingdino visualiser/inference helpers, none of which app.py reaches. pip
# refuses to resolve the pair in one pass, which is what surfaced this.
#
# Three passes, not one, and the ORDER is the point. gradio 4.44.1 pins
# tomlkit==0.12.0 while runpod 1.11.0 asks for tomlkit>=0.15.1, so a single
# resolution is impossible — yet the working venv holds both, because they were
# installed at different times and ended on tomlkit 0.12.0. runpod runs fine
# against it; its declared floor is stricter than its real need. Installing runpod
# first and letting gradio pull tomlkit back down to 0.12.0 lands on exactly the
# freeze's final state.
RUN python -m pip install --upgrade pip \
    && python -m pip install "runpod==1.11.0" \
    && python -m pip install \
        "transformers==4.44.2" \
        "numpy==1.26.4" \
        "timm==1.0.28" \
        "addict==2.4.0" \
        "yapf==0.40.1" \
        "pycocotools==2.0.11" \
        "pillow==10.4.0" \
        "matplotlib==3.10.9" \
        "pyyaml==6.0.3" \
        "tqdm==4.70.0" \
        "scipy==1.15.3" \
        "pandas==2.3.3" \
        "termcolor==3.3.0" \
        "colorlog==6.12.0" \
        "gradio==4.44.1" \
        "gradio_client==1.3.0" \
    && python -m pip install \
        /app/gradio_image_prompter-0.1.0-py3-none-any.whl \
    # Fail the BUILD, not the first user request, if the pinned set is broken.
    && python -c "import torch, transformers, numpy, tomlkit, runpod, gradio; \
print('torch', torch.__version__, 'transformers', transformers.__version__, \
      'numpy', numpy.__version__, 'tomlkit', tomlkit.__version__); \
assert torch.__version__.startswith('2.1.1'), torch.__version__; \
assert numpy.__version__.startswith('1.'), numpy.__version__; \
assert tomlkit.__version__ == '0.12.0', tomlkit.__version__"

# The MultiScaleDeformableAttention CUDA extension.
#
# NOT the wheel that ships in the space. That wheel's .so is mangled with
# `St8optional` (std::optional), i.e. it was compiled against a torch new enough
# to have dropped c10::optional, and importing it under torch 2.1.1 dies with
# `undefined symbol: _ZN2at4_ops10zeros_like4call...`. Verified by symbol
# inspection: the wheel carries 496 St8optional symbols and zero c10 ones.
#
# This .so is the binary that has served every successful job — compiled from
# ops/src against torch 2.1.1, and carrying 3245 c10::optional symbols. Shipping
# it means no nvcc pass at build time and an artefact bit-identical to the one in
# production rather than a fresh guess at it.
#
# It lives in msda/ rather than under ops/build/ because the HF space's own
# .gitignore excludes `models/GroundingDINO/ops/build/` — which silently kept the
# file out of the first push and made this step fail with a missing source path.
COPY msda/MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so \
     /usr/local/lib/python3.10/dist-packages/
RUN python -c "import MultiScaleDeformableAttention; print('MSDA extension loads against torch', __import__('torch').__version__)"

# Import the handler's whole dependency graph at build time, so a missing module
# or a bad pin surfaces here rather than as a 250-second timeout in the app. The
# model itself needs CUDA to move to the GPU, so only the import is checked.
RUN python -c "import app; \
[getattr(app, n) for n in ('get_args_parser','get_device','build_model_and_transforms','predict','get_xy_from_boxes','generate_heatmap')]; \
print('app.py imports clean, all six handler entry points present')"

CMD ["python", "-u", "rp_handler.py"]

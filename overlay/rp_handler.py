import io, base64
import runpod
from PIL import Image
from app import CONF_THRESH, get_args_parser, get_device, build_model_and_transforms, predict, get_xy_from_boxes, generate_heatmap

parser = get_args_parser()
args = parser.parse_args([])
device = get_device()
model, transform = build_model_and_transforms(args)
model = model.to(device)

# CountGD was trained with up to three visual exemplars; extra boxes add cost
# without adding signal, so anything beyond three is ignored.
MAX_EXEMPLARS = 3


def parse_exemplars(raw, width, height):
    """[[x1,y1,x2,y2]...] in source-image pixels -> image-prompter box tuples.

    The gradio image-prompter encodes a box as [x1, y1, 2.0, x2, y2, 3.0] and
    get_box_inputs() in app.py filters on those two marker values, so this is
    the exact shape preprocess() expects. Boxes are clamped to the image and
    degenerate ones (under 4px a side) dropped rather than rejected — a partial
    exemplar list still improves the count, a hard error never does.
    """
    points = []
    if not isinstance(raw, list):
        return points
    for box in raw[:MAX_EXEMPLARS]:
        try:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
        y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            points.append([x1, y1, 2.0, x2, y2, 3.0])
    return points


def handler(event):
    inp = event.get("input") or {}
    if "image" not in inp:
        return {"error": "missing input.image (base64)"}
    img = Image.open(io.BytesIO(base64.b64decode(inp["image"]))).convert("RGB")

    text = (inp.get("prompt") or "").strip()
    exemplar_points = parse_exemplars(inp.get("exemplars"), img.width, img.height)
    # Text-only requests behave exactly as v1 (same "objects" fallback). With
    # exemplars present an empty text is legitimate — that is CountGD's
    # JUST_EXEMPLARS mode — so the fallback must not overwrite it.
    if not text and not exemplar_points:
        text = "objects"
    prompts = {"image": img, "points": exemplar_points} if exemplar_points else None

    try:
        conf_thresh = float(inp["threshold"])
    except (KeyError, TypeError, ValueError):
        conf_thresh = None  # predict() falls back to CONF_THRESH (0.23)
    if conf_thresh is not None:
        # Below 0.05 the tail is noise; above 0.95 nothing survives. Clamping
        # beats erroring for the same reason as parse_exemplars above.
        conf_thresh = min(max(conf_thresh, 0.05), 0.95)

    boxes, _ = predict(model, transform, img, text, prompts, device, conf_thresh)
    x, y = get_xy_from_boxes(img, boxes)
    points = [[round(float(px), 2), round(float(py), 2)] for px, py in zip(x, y)]
    out = {
        "count": int(len(boxes)),
        "points": points,
        "width": img.width,
        "height": img.height,
        "prompt": text,
        "exemplars_used": len(exemplar_points),
        "threshold": conf_thresh if conf_thresh is not None else CONF_THRESH,
    }
    if inp.get("return_image"):
        heat = generate_heatmap(img, boxes).convert("RGB")
        buf = io.BytesIO()
        heat.save(buf, format="WEBP", quality=80)
        out["marked_image_b64"] = base64.b64encode(buf.getvalue()).decode()
    return out


runpod.serverless.start({"handler": handler})

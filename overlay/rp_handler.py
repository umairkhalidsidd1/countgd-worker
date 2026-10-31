import io, base64, os, statistics
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

# Automatic examples (v3). Text alone misses objects that differ in size or
# look from the "typical" one: on a photo of 29 mixed pills (big ovals,
# capsules, small round tablets) "pills" found 6. Giving the model three of
# its own first-look boxes as visual exemplars - smallest, middle and largest
# of a consistent size - found all 29. Measured 2026-10-29 on the app's graded
# set (12 photos): mean error 10.2% -> 2.2%, no photo worse.
#   - Only with >= REFINE_MIN_COUNT first-look objects: small scenes are easy
#     and too few boxes give poor examples.
#   - Examples only from boxes within 4x of the median area: a mixed desk
#     photo's "coins" look included a phone and a pen, and examples taken from
#     those taught the model to count phones and pens too.
#   - The second count is used only when it finds REFINE_MIN_GAIN times more:
#     the examples must have recovered objects the words clearly missed, not
#     nudged an already-good count (boxes 43 -> 47, coins 8 -> 10 rejected).
# AUTO_REFINE=0 turns it off for every request; input.refine=false per request.
AUTO_REFINE = os.environ.get("AUTO_REFINE", "1") != "0"
REFINE_MIN_COUNT = 5
REFINE_MIN_GAIN = 1.3
REFINE_SIZE_SPREAD = 4.0


# Weak dots on busy photos (measured 2026-10-31: 18 fresh photos, mean error
# 18.6% -> 14.4%, none worse). With HARD_MIN_COUNT+ dots, drop those scoring under
# HARD_RAISED: labels, caps and the bark side of logs score low. It is the same
# model pass, so it adds no time.
# Guard: if that would remove more than 15% of the dots, the whole photo scores
# low and they are real objects. Carrom pieces went 29 -> 1 without it.
# Skipped when the v3 recount was used, for user examples, for a request with its
# own threshold, and for panoramas: objects there get little zoom and score low
# (deer 28 -> 24 of 36). HARD_RULES=0 turns it off; input.hard_rules=false per
# request.
HARD_RULES = os.environ.get("HARD_RULES", "1") != "0"
HARD_MIN_COUNT = 20
HARD_RAISED = 0.30
HARD_GUARD = 0.85
WIDE_RATIO = 2.5


def raised_boxes(boxes, logits):
    """Boxes scoring above HARD_RAISED, or None if the guard says keep them all."""
    keep = logits.max(axis=-1) > HARD_RAISED
    if keep.all() or keep.sum() < HARD_GUARD * len(boxes):
        return None
    return boxes[keep]


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


def pick_auto_exemplars(boxes, width, height):
    """Smallest, middle and largest first-look box of a consistent size.

    boxes are CountGD's normalised (cx, cy, w, h); returns image-prompter
    tuples like parse_exemplars().
    """
    sized = []
    for b in boxes:
        cx, cy, w, h = (float(v) for v in b[:4])
        sized.append((w * h, cx, cy, w, h))
    if not sized:
        return []
    median = statistics.median(a for a, *_ in sized)
    ok = sorted(s for s in sized
                if median / REFINE_SIZE_SPREAD <= s[0] <= median * REFINE_SIZE_SPREAD)
    if len(ok) > MAX_EXEMPLARS:
        ok = [ok[0], ok[len(ok) // 2], ok[-1]]
    out = []
    for _, cx, cy, w, h in ok:
        x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
        x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            out.append([x1, y1, 2.0, x2, y2, 3.0])
    return out


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

    boxes, logits = predict(model, transform, img, text, prompts, device, conf_thresh)
    words_count = int(len(boxes))
    auto_refined = False
    refine = inp.get("refine", AUTO_REFINE)
    if refine and not exemplar_points and words_count >= REFINE_MIN_COUNT:
        auto_points = pick_auto_exemplars(boxes, img.width, img.height)
        if auto_points:
            boxes2, _ = predict(model, transform, img, text,
                                {"image": img, "points": auto_points}, device, conf_thresh)
            if len(boxes2) >= REFINE_MIN_GAIN * words_count:
                boxes = boxes2
                auto_refined = True
    hard_route = None
    if (inp.get("hard_rules", HARD_RULES) and conf_thresh is None and not exemplar_points
            and not auto_refined and words_count >= HARD_MIN_COUNT
            and max(img.size) / min(img.size) < WIDE_RATIO):
        kept = raised_boxes(boxes, logits)
        if kept is not None:
            boxes = kept
            hard_route = "raised"
    x, y = get_xy_from_boxes(img, boxes)
    points = [[round(float(px), 2), round(float(py), 2)] for px, py in zip(x, y)]
    out = {
        "count": int(len(boxes)),
        "points": points,
        "width": img.width,
        "height": img.height,
        "prompt": text,
        # The user's own examples only: the app shows "refined with your
        # examples" for this, so automatic ones must not light it up.
        "exemplars_used": len(exemplar_points),
        "auto_refined": auto_refined,
        "words_count": words_count,
        "hard_route": hard_route,
        "threshold": conf_thresh if conf_thresh is not None else CONF_THRESH,
    }
    if inp.get("return_image"):
        heat = generate_heatmap(img, boxes).convert("RGB")
        buf = io.BytesIO()
        heat.save(buf, format="WEBP", quality=80)
        out["marked_image_b64"] = base64.b64encode(buf.getvalue()).decode()
    return out


runpod.serverless.start({"handler": handler})

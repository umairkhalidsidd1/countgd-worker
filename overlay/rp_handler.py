import io, base64
import runpod
from PIL import Image
from app import get_args_parser, get_device, build_model_and_transforms, predict, get_xy_from_boxes, generate_heatmap

parser = get_args_parser()
args = parser.parse_args([])
device = get_device()
model, transform = build_model_and_transforms(args)
model = model.to(device)


def handler(event):
    inp = event.get("input") or {}
    if "image" not in inp:
        return {"error": "missing input.image (base64)"}
    img = Image.open(io.BytesIO(base64.b64decode(inp["image"]))).convert("RGB")
    text = (inp.get("prompt") or "objects").strip() or "objects"
    boxes, _ = predict(model, transform, img, text, None, device)
    x, y = get_xy_from_boxes(img, boxes)
    points = [[round(float(px), 2), round(float(py), 2)] for px, py in zip(x, y)]
    out = {
        "count": int(len(boxes)),
        "points": points,
        "width": img.width,
        "height": img.height,
        "prompt": text,
    }
    if inp.get("return_image"):
        heat = generate_heatmap(img, boxes).convert("RGB")
        buf = io.BytesIO()
        heat.save(buf, format="WEBP", quality=80)
        out["marked_image_b64"] = base64.b64encode(buf.getvalue()).decode()
    return out


runpod.serverless.start({"handler": handler})

import io
from PIL import Image
from app import get_args_parser, get_device, build_model_and_transforms, predict

parser = get_args_parser()
args = parser.parse_args([])
device = get_device()
print("SMOKE device:", device)
model, transform = build_model_and_transforms(args)
model = model.to(device)
img = Image.open("strawberry.jpg").convert("RGB")
boxes, _ = predict(model, transform, img, "strawberry", None, device)
print("SMOKE_COUNT =", len(boxes))

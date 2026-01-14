import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.ops import nms
from PIL import Image, ImageDraw
import imagehash
from skimage.metrics import structural_similarity as ssim

print("--- PCB Differential Detection (EfficientNet-B0) ---")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 1. MODEL LOADING
# =========================
model_path = "best_efficientnet_pcb_defects_50epochs.pth"
checkpoint = torch.load(model_path, map_location=device)

class_names = checkpoint.get(
    "class_names",
    ['missing_hole', 'mouse_bite', 'open_circuit',
     'short', 'spur', 'spurious_copper']
)

if "normal" in class_names:
    class_names.remove("normal")

num_classes = len(class_names)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    num_classes
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

# =========================
# 2. CONFIGURATION
# =========================
WINDOW_SIZE = 64
STRIDE = 16
SIMILARITY_THRESHOLD = 0.92
CONF_THRESHOLD = 0.35
NMS_IOU = 0.05  # 🔥 aggressive merge for thin defects

golden_images_dir = r"C:\Project5\milestone3\PCB_DATASET\PCB_USED"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# 3. GOLDEN IMAGE DATABASE
# =========================
golden_db = []

for f in os.listdir(golden_images_dir):
    try:
        img = Image.open(os.path.join(golden_images_dir, f)).convert("RGB")
        golden_db.append({
            "img": img,
            "hash": imagehash.phash(img)
        })
    except:
        continue

def find_best_golden(img):
    h = imagehash.phash(img)
    return min(golden_db, key=lambda x: h - x["hash"])["img"]

# =========================
# 4. DEFECT DETECTION
# =========================
def detect_defects(input_img, golden_img):
    if input_img.size != golden_img.size:
        golden_img = golden_img.resize(input_img.size)

    w, h = input_img.size
    input_gray = np.array(input_img.convert("L"))
    golden_gray = np.array(golden_img.convert("L"))

    detections = []

    for y in range(0, h - WINDOW_SIZE + 1, STRIDE):
        for x in range(0, w - WINDOW_SIZE + 1, STRIDE):

            patch_in = input_gray[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
            patch_golden = golden_gray[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]

            score = ssim(patch_golden, patch_in)

            if score < SIMILARITY_THRESHOLD:
                patch = input_img.crop((x, y, x+WINDOW_SIZE, y+WINDOW_SIZE))
                tensor = transform(patch).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs = F.softmax(model(tensor), dim=1)
                    topk_conf, topk_idx = torch.topk(probs, k=2, dim=1)

                conf1 = topk_conf[0][0].item()
                idx1 = topk_idx[0][0].item()
                conf2 = topk_conf[0][1].item()
                idx2 = topk_idx[0][1].item()

                if conf1 > CONF_THRESHOLD:
                    detections.append({
                        "box": [x, y, x+WINDOW_SIZE, y+WINDOW_SIZE],
                        "label": class_names[idx1],
                        "confidence": conf1,
                        "alt_label": class_names[idx2],
                        "alt_confidence": conf2
                    })

    if not detections:
        return []

    boxes = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["confidence"] for d in detections])

    keep = nms(boxes, scores, iou_threshold=NMS_IOU)

    return [detections[i] for i in keep]

# =========================
# 5. OPEN CIRCUIT LABEL FIX
# =========================
def fix_opencircuit_label(detections):
    if not detections:
        return detections

    open_score = 0
    copper_score = 0

    for d in detections:
        if d["label"] == "open_circuit":
            open_score += d["confidence"]
        if d["label"] in ["spur", "spurious_copper"]:
            copper_score += d["confidence"]

        if d.get("alt_label") == "open_circuit":
            open_score += d.get("alt_confidence", 0) * 0.7
        if d.get("alt_label") in ["spur", "spurious_copper"]:
            copper_score += d.get("alt_confidence", 0) * 0.7

    # 🔥 FINAL DECISION
    if open_score >= copper_score:
        for d in detections:
            d["label"] = "open_circuit"

    return detections

# =========================
# 6. DRAW RESULTS
# =========================
def draw_boxes(img, detections):
    draw = ImageDraw.Draw(img)

    colors = {
        "missing_hole": "red",
        "mouse_bite": "orange",
        "open_circuit": "blue",
        "short": "purple",
        "spur": "yellow",
        "spurious_copper": "green"
    }

    for d in detections:
        box = d["box"]
        label = d["label"]
        color = colors.get(label, "red")

        draw.rectangle(box, outline=color, width=3)
        draw.text(
            (box[0], box[1] - 12),
            f"{label} {d['confidence']:.2f}",
            fill=color
        )

    return img

# =========================
# 7. STREAMLIT ENTRY POINT
# =========================
def run_inference_on_pil(input_image):
    input_image = input_image.convert("RGB")
    golden = find_best_golden(input_image)

    detections = detect_defects(input_image, golden)
    detections = fix_opencircuit_label(detections)

    result = draw_boxes(input_image.copy(), detections)
    return result, detections
# app.py
import os, json
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, url_for, send_from_directory

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ---------------- CONFIG ----------------
MODEL_FILE   = "repvgg_mem_pruned_finetuned.pth"
MAPPING_FILE = "class_mapping_merged.json"
UPLOAD_FOLDER = Path("uploads")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
IMG_SIZE = 224
OOD_THRESHOLD = 80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.85   # 85% strict threshold

# ----------------------------------------

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# ---------------- transforms ----------------
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------- proper inverse normalization (100% correct) ----------
inv_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
inv_std  = [1/0.229, 1/0.224, 1/0.225]

_inv_norm = transforms.Normalize(mean=inv_mean, std=inv_std)

def tensor_to_pil(tensor):
    """Convert normalized tensor -> clean preview PIL"""
    t = _inv_norm(tensor).clamp(0,1)
    t = (t.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
    return Image.fromarray(t)

# ---------------- MODEL ----------------
class MSCA(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        mid = max(8, in_ch//8)
        self.conv1 = nn.Conv2d(in_ch, mid, (1,3), padding=(0,1), bias=False)
        self.conv2 = nn.Conv2d(mid, in_ch, (3,1), padding=(1,0), bias=False)
        self.gn = nn.GroupNorm(1, in_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        y = self.conv1(x); y = self.conv2(y); y = self.gn(y)
        return self.act(y + x)

class ReparamBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.identity = nn.BatchNorm2d(in_ch) if (in_ch == out_ch and stride == 1) else None
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.identity is not None:
            out += self.identity(x)
        return self.act(out)

class RepVGG_MEM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        c1, c2, c3 = 64, 128, 512
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(ReparamBlock(c1,c1,1), ReparamBlock(c1,c1,1))
        self.stage2 = nn.Sequential(ReparamBlock(c1,c2,2), ReparamBlock(c2,c2,1))
        self.stage3 = nn.Sequential(ReparamBlock(c2,c3,2), ReparamBlock(c3,c3,1))
        self.msca = MSCA(c3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.msca(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# ---------------- load mapping ----------------
with open(MAPPING_FILE, "r") as f:
    mapping = json.load(f)

if isinstance(mapping, list):
    mapping = {str(i): name for i, name in enumerate(mapping)}
mapping = {str(k): v for k,v in mapping.items()}
idx_to_name = {int(k): v for k,v in mapping.items()}
num_classes = len(mapping)

# ---------------- load model ----------------
model = RepVGG_MEM(num_classes=num_classes).to(DEVICE)
ck = torch.load(MODEL_FILE, map_location=DEVICE)
if "state_dict" in ck:
    ck = ck["state_dict"]
sd_fixed = {k.replace("module.",""):v for k,v in ck.items()}
model.load_state_dict(sd_fixed, strict=False)
model.eval()

# ---------------- helper utils ----------------
def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT

def save_preview_and_return_urls(pil_img, save_basename):
    """Save processed preview correctly."""
    processed_tensor = val_transform(pil_img)
    preview_pil = tensor_to_pil(processed_tensor)

    preview_name = f"prep_{save_basename}"
    preview_path = UPLOAD_FOLDER / preview_name
    preview_pil.save(preview_path)

    return (
        url_for("uploaded_file", filename=save_basename),
        url_for("uploaded_file", filename=preview_name),
    )

def predict_pil(pil_img):
    x = val_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = int(probs.argmax())
    conf = float(probs[top_idx])
    top5 = [(int(i), float(probs[i]), idx_to_name[int(i)]) 
            for i in probs.argsort()[::-1][:5]]
    return top_idx, conf, top5

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/upload")
def upload_page():
    return render_template("predict.html", error=None, image_url=None)

@app.route("/predict", methods=["POST"])
def predict_route():

    # 🔴 MUST be images[]
    if "images[]" not in request.files:
        return render_template("predict.html", error="No file part")

    files = request.files.getlist("images[]")

    if not files or files[0].filename == "":
        return render_template("predict.html", error="No selected file")

    results = []

    for file in files:
        filename = secure_filename(file.filename)

        if not allowed_file(filename):
            continue

        basename = (
            f"{Path(filename).stem}_"
            f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
            f"{Path(filename).suffix}"
        )

        save_path = UPLOAD_FOLDER / basename
        file.save(save_path)

        try:
            pil = Image.open(save_path).convert("RGB")
        except:
            continue

        orig_url, prep_url = save_preview_and_return_urls(pil, basename)
        idx, conf, top5 = predict_pil(pil)

# STRICT VALIDATION
        if conf < CONFIDENCE_THRESHOLD or idx not in idx_to_name:
            results.append({
                "image_url": orig_url,
                "prediction": "Invalid Input",
                "confidence": f"{conf:.4f}"
            })
        else:
            results.append({
                "image_url": orig_url,
                "prediction": idx_to_name[idx],
                "confidence": f"{conf:.4f}"
            })



    if not results:
        return render_template("predict.html", error="No valid images uploaded")

    return render_template("predict.html", results=results)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

@app.route("/about")
def about():
    return render_template("about.html")



if __name__ == "__main__":
    print("Starting Flask app on ::5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
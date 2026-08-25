import torch
import numpy as np
import cv2
import os
import time
from PIL import Image
from torch import nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
import joblib
import torch.serialization

# ✅ Correct imports from leaf_analysis
from leaf_analysis import (
    calculate_visible_leaf_damage,
    estimate_co2_assimilation_loss,
    estimate_disease_progression,
    estimate_days_to_severe_infection
)

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL + PROTOTYPE PATHS
# =========================
MODEL_STAGE1_FULL = "models/model_1_full.pth"
MODEL_STAGE2_FULL = "models/model_2_full.pth"

PROTOS_STAGE1_PATH = "models/prototypes_stage1.pkl"
SCALER_STAGE1_PATH = "models/scaler_stage1.pkl"

PROTOS_STAGE2_PATH = "models/prototypes_stage2.pkl"
SCALER_STAGE2_PATH = "models/scaler_stage2.pkl"

MODEL_STAGE3_PATH = "models/KNN_stage3_ml.pkl"   # (clf, scaler)

# =========================
# CLASS LABELS
# =========================
STAGE1_CLASSES = ["Ganwar", "Paddy"]
STAGE2_CLASSES = ["healthy_G", "healthy_P", "unhealthy_G", "unhealthy_P"]
STAGE3_CLASSES = [
    "unhealthy_blackspot_G",
    "unhealthy_brownspot_P",
    "unhealthy_damaged_G",
    "unhealthy_damaged_P"
]

# =========================
# TRANSFORM (Stage 1 & 2)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# MODEL CLASS (must match Stage 1 & 2 training EXACTLY)
# =========================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return nn.functional.normalize(x, dim=1)

torch.serialization.add_safe_globals([Model])

# =========================
# LOAD MODELS + PROTOTYPES + SCALERS
# =========================
def load_models():
    model1 = torch.load(MODEL_STAGE1_FULL, map_location=DEVICE, weights_only=False)
    model1.eval()

    model2 = torch.load(MODEL_STAGE2_FULL, map_location=DEVICE, weights_only=False)
    model2.eval()

    protos1 = joblib.load(PROTOS_STAGE1_PATH)
    scaler1 = joblib.load(SCALER_STAGE1_PATH)

    protos2 = joblib.load(PROTOS_STAGE2_PATH)
    scaler2 = joblib.load(SCALER_STAGE2_PATH)

    clf3, scaler3 = joblib.load(MODEL_STAGE3_PATH)

    return model1, model2, protos1, scaler1, protos2, scaler2, clf3, scaler3

# =========================
# EMBEDDING (Stage 1 & 2)
# =========================
def get_embedding(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(img)
    return emb.cpu().numpy()[0]

# =========================
# COSINE SIMILARITY
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# HOG FEATURE (Stage 3)
# =========================
def extract_hog(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    return hog.compute(gray).flatten()

# =========================
# PROCESS ONE IMAGE
# =========================
def process_image(img_path, models):
    model1, model2, protos1, scaler1, protos2, scaler2, clf3, scaler3 = models

    print(f"\n==============================")
    print(f"Processing: {os.path.basename(img_path)}")
    print(f"==============================")

    # ---------- STAGE 1 ----------
    emb1 = get_embedding(model1, img_path)
    emb1_scaled = scaler1.transform([emb1])
    sims1 = {cls: cosine_similarity(emb1_scaled[0], proto) for cls, proto in protos1.items()}
    stage1_idx = max(sims1, key=sims1.get)
    stage1_label = STAGE1_CLASSES[stage1_idx]
    print("Stage 1:", stage1_label)

    # ---------- STAGE 2 ----------
    emb2 = get_embedding(model2, img_path)
    emb2_scaled = scaler2.transform([emb2])
    sims2 = {cls: cosine_similarity(emb2_scaled[0], proto) for cls, proto in protos2.items()}
    stage2_idx = max(sims2, key=sims2.get)
    stage2_label = STAGE2_CLASSES[stage2_idx]
    print("Stage 2:", stage2_label)

    if stage2_label.startswith("healthy"):
        print("Final Result:", stage2_label)
        return

    # ---------- STAGE 3 ----------
    feat = extract_hog(img_path)
    feat_scaled = scaler3.transform([feat])
    stage3_idx = clf3.predict(feat_scaled)[0]
    stage3_label = STAGE3_CLASSES[stage3_idx]

    print("Stage 3:", stage3_label)
    print("Final Result:", stage3_label)

    # ---------- Leaf Damage Analysis ----------
    damage_percent = calculate_visible_leaf_damage(img_path, disease_type=stage3_label)
    co2_loss = estimate_co2_assimilation_loss(damage_percent, disease_type=stage3_label)
    projected_severity = estimate_disease_progression(damage_percent, disease_type=stage3_label, days=7)
    days_to_severe = estimate_days_to_severe_infection(damage_percent, disease_type=stage3_label)

    print(f"Damage %: {damage_percent:.2f}%")
    print(f"CO₂ assimilation loss: {co2_loss} µmol/m²/s")

    if projected_severity is not None:
        print(f"Projected severity after 7 days: {projected_severity}%")

    print(f"Estimated days until severe infection (~90% damage): {days_to_severe} days")


# =========================
# MAIN (PROCESS MULTIPLE IMAGES + TIMING)
# =========================
if __name__ == "__main__":
    folder = "test_images"   # folder containing your images
    models = load_models()

    total_start = time.time()
    count = 0

    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            img_path = os.path.join(folder, file)

            img_start = time.time()
            process_image(img_path, models)
            img_end = time.time()

            print(f"Time for this image: {img_end - img_start:.4f} seconds")
            count += 1

    total_end = time.time()

    print("\n====================================")
    print(f"Total images processed: {count}")
    print(f"Total inference time: {total_end - total_start:.4f} seconds")
    print(f"Average time per image: {(total_end - total_start)/count:.4f} seconds")
    print("====================================")

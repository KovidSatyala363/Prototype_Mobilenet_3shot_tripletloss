import torch
import numpy as np
import cv2
import time
from PIL import Image
from torch import nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
import joblib
import torch.serialization

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
# PIPELINE
# =========================
def run_pipeline(img_path):

    start_time = time.time()   # <-- START TIMER

    model1, model2, protos1, scaler1, protos2, scaler2, clf3, scaler3 = load_models()

    # ---------- STAGE 1 ----------
    print("\n===== STAGE 1: GANWAR vs PADDY =====")
    emb1 = get_embedding(model1, img_path)
    emb1_scaled = scaler1.transform([emb1])

    sims1 = {cls: cosine_similarity(emb1_scaled[0], proto) for cls, proto in protos1.items()}
    stage1_idx = max(sims1, key=sims1.get)
    print("Stage 1 prediction:", STAGE1_CLASSES[stage1_idx])

    # ---------- STAGE 2 ----------
    print("\n===== STAGE 2: HEALTHY vs UNHEALTHY =====")
    emb2 = get_embedding(model2, img_path)
    emb2_scaled = scaler2.transform([emb2])

    sims2 = {cls: cosine_similarity(emb2_scaled[0], proto) for cls, proto in protos2.items()}
    stage2_idx = max(sims2, key=sims2.get)
    stage2_label = STAGE2_CLASSES[stage2_idx]

    print("Stage 2 prediction:", stage2_label)

    if stage2_label.startswith("healthy"):
        end_time = time.time()
        print("\nLeaf is classified as HEALTHY")
        print(f"\nTotal inference time: {end_time - start_time:.4f} seconds")
        return

    # ---------- STAGE 3 ----------
    print("\n===== STAGE 3: DISEASE TYPE =====")
    feat = extract_hog(img_path)
    feat_scaled = scaler3.transform([feat])
    stage3_idx = clf3.predict(feat_scaled)[0]

    print("Stage 3 prediction:", STAGE3_CLASSES[stage3_idx])

    end_time = time.time()   # <-- END TIMER
    print(f"\nTotal inference time: {end_time - start_time:.4f} seconds")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    test_image = "IMG_20250728_100749_kk-removebg-preview_preprocessed.png"
    run_pipeline(test_image)

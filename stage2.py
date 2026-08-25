# =========================
# STAGE 2 (4 CLASSES, SEMI-HARD TRIPLET MINING) — PROTOTYPE + COSINE VERSION
# Modified: plot fonts set to size=12 and bold; titles removed from plots.
# =========================

import os, shutil, random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# PATHS
# =========================
ROOT = "datasets/stage2"

TRAIN_DIR = "data/train2"
VAL_DIR = "data/val2"
TEST_DIR = "data/test2"

SUPPORT_DIR = "data/support_set2"
QUERY_DIR = "data/query_set2"


OUTPUT = "outputs/predicted_output_stage2"

MODEL_PATH = "models/model_2.pth"            # state_dict
MODEL_FULL_PATH = "models/model_2_full.pth"  # full model

CLASSES = ["healthy_G", "healthy_P", "unhealthy_G", "unhealthy_P"]

K_SUPPORT = 3
EPOCHS = 50
BATCH_SIZE = 64
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# 60/20/20 SPLIT
# =========================
def split_dataset():
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASSES:
            os.makedirs(os.path.join(d, cls), exist_ok=True)

    for cls in CLASSES:
        imgs = os.listdir(os.path.join(ROOT, cls))
        random.shuffle(imgs)

        total = len(imgs)
        train_n = int(0.6 * total)
        val_n = int(0.2 * total)

        train_imgs = imgs[:train_n]
        val_imgs = imgs[train_n:train_n + val_n]
        test_imgs = imgs[train_n + val_n:]

        for img in train_imgs:
            shutil.copy(os.path.join(ROOT, cls, img),
                        os.path.join(TRAIN_DIR, cls, img))

        for img in val_imgs:
            shutil.copy(os.path.join(ROOT, cls, img),
                        os.path.join(VAL_DIR, cls, img))

        for img in test_imgs:
            shutil.copy(os.path.join(ROOT, cls, img),
                        os.path.join(TEST_DIR, cls, img))

    print("Dataset split complete: 60% train, 20% val, 20% test.")

# =========================
# SUPPORT / QUERY SPLIT
# =========================
def create_support_query():
    for d in [SUPPORT_DIR, QUERY_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

    for cls in CLASSES:
        imgs = os.listdir(os.path.join(TEST_DIR, cls))
        random.shuffle(imgs)

        os.makedirs(os.path.join(SUPPORT_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(QUERY_DIR, cls), exist_ok=True)

        support = imgs[:K_SUPPORT]
        query = imgs[K_SUPPORT:]

        for img in support:
            shutil.copy(os.path.join(TEST_DIR, cls, img),
                        os.path.join(SUPPORT_DIR, cls, img))

        for img in query:
            shutil.copy(os.path.join(TEST_DIR, cls, img),
                        os.path.join(QUERY_DIR, cls, img))

    print("Support/Query split complete (NO LEAKAGE).")

# =========================
# DATASET CLASS
# =========================
class ImgDataset(Dataset):
    def __init__(self, root, train=False):
        self.train = train
        self.data = []
        for label, cls in enumerate(CLASSES):
            path = os.path.join(root, cls)
            for img in os.listdir(path):
                self.data.append((os.path.join(path, img), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, y = self.data[i]
        img = Image.open(path).convert("RGB")
        img = train_transform(img) if self.train else test_transform(img)
        return img, y, path

# =========================
# MODEL
# =========================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        freeze = int(0.7 * len(base.features))
        for i, l in enumerate(base.features):
            for p in l.parameters():
                p.requires_grad = (i >= freeze)

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

# =========================
# TRIPLET LOSS
# =========================
class TripletLoss(nn.Module):
    def __init__(self, margin=1.2):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        pos = torch.sum((a - p) ** 2, dim=1)
        neg = torch.sum((a - n) ** 2, dim=1)
        return torch.relu(pos - neg + self.margin).mean()

# =========================
# SEMI-HARD MINING
# =========================
def make_triplets(embeddings, labels, margin=1.2):
    a, p, n = [], [], []
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    for i in range(len(embeddings)):
        anchor = embeddings[i]
        label = labels[i]

        pos_mask = (labels == label)
        neg_mask = (labels != label)
        pos_mask[i] = False

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_dist = dist_matrix[i][pos_mask]
        neg_dist = dist_matrix[i][neg_mask]

        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]

        median_pos = torch.median(pos_dist)
        pos_diff = torch.abs(pos_dist - median_pos)
        semi_pos = embeddings[pos_idx[torch.argmin(pos_diff)]]

        valid_neg_mask = (neg_dist > pos_dist.min()) & (neg_dist < pos_dist.min() + margin)

        if valid_neg_mask.sum() > 0:
            valid_neg_dist = neg_dist[valid_neg_mask]
            valid_neg_idx = neg_idx[valid_neg_mask]
            semi_neg = embeddings[valid_neg_idx[torch.argmin(valid_neg_dist)]]
        else:
            semi_neg = embeddings[neg_idx[torch.argmin(neg_dist)]]

        a.append(anchor)
        p.append(semi_pos)
        n.append(semi_neg)

    if len(a) == 0:
        return None, None, None

    return torch.stack(a), torch.stack(p), torch.stack(n)

# =========================
# TRAINING
# =========================
def train_model():
    train_loader = DataLoader(ImgDataset(TRAIN_DIR, train=True),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ImgDataset(VAL_DIR, train=False),
                            batch_size=BATCH_SIZE, shuffle=False)

    model = Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = TripletLoss()

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    patience = 5
    bad_epochs = 0

    for e in range(EPOCHS):
        model.train()
        total_loss = 0
        total_triplets = 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            emb = model(x)
            a, p, n = make_triplets(emb, y)
            if a is None:
                continue

            loss = loss_fn(a, p, n)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(a)
            total_triplets += len(a)

        train_loss = total_loss / max(total_triplets, 1)
        train_losses.append(train_loss)

        # VALIDATION
        model.eval()
        val_total = 0
        val_triplets = 0

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                emb = model(x)
                a, p, n = make_triplets(emb, y)
                if a is None:
                    continue

                val_total += loss_fn(a, p, n).item() * len(a)
                val_triplets += len(a)

        val_loss = val_total / max(val_triplets, 1)
        val_losses.append(val_loss)

        print(f"Epoch {e+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {e+1}")
                break

    # SAVE LOSS CURVE (font size=12, bold, no title)
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch", fontsize=12, weight='bold')
    plt.ylabel("Triplet Loss", fontsize=12, weight='bold')
    plt.legend(prop={'size': 12, 'weight': 'bold'})
    plt.grid(True)
    # Title intentionally removed for publication style
    plt.savefig("outputs/loss_curve_stage2_prototype.png", dpi=300)
    plt.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs("models", exist_ok=True)

    # Save state_dict
    torch.save(model.state_dict(), MODEL_PATH)

    # Save full model
    torch.save(model, MODEL_FULL_PATH)

    print("\nSaved Stage‑2 full model:", MODEL_FULL_PATH)
    print("Saved Stage‑2 state_dict:", MODEL_PATH)

    return model

# =========================
# PROTOTYPE + COSINE CLASSIFIER
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def evaluate(model):
    model.eval()

    support = ImgDataset(SUPPORT_DIR, train=False)
    query = ImgDataset(QUERY_DIR, train=False)

    Xs, ys = [], []

    for x, y, _ in support:
        emb = model(x.unsqueeze(0).to(device)).cpu().detach().numpy()[0]
        Xs.append(emb)
        ys.append(y)

    Xs = np.array(Xs)
    ys = np.array(ys)

    # =========================
    # COMPUTE PROTOTYPES
    # =========================
    prototypes = {}
    for cls in range(len(CLASSES)):
        cls_embs = Xs[ys == cls]
        prototypes[cls] = np.mean(cls_embs, axis=0)

    # =========================
    # FIT SCALER (for final_pipeline compatibility)
    # =========================
    scaler = StandardScaler()
    scaler.fit(Xs)

    # =========================
    # SAVE PROTOTYPES + SCALER
    # =========================
    os.makedirs("models", exist_ok=True)
    joblib.dump(prototypes, "models/prototypes_stage2.pkl")
    joblib.dump(scaler, "models/scaler_stage2.pkl")

    print("\nSaved Stage‑2 Prototypes: models/prototypes_stage2.pkl")
    print("Saved Stage‑2 Scaler: models/scaler_stage2.pkl")

    preds, trues, paths = [], [], []

    # =========================
    # QUERY PREDICTION
    # =========================
    for x, y, p in query:
        emb = model(x.unsqueeze(0).to(device)).cpu().detach().numpy()[0]

        sims = {cls: cosine_similarity(emb, proto) for cls, proto in prototypes.items()}
        pred = max(sims, key=sims.get)

        preds.append(pred)
        trues.append(y)
        paths.append(p)

    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)

    for p, pred in zip(paths, preds):
        folder = CLASSES[pred]
        os.makedirs(f"{OUTPUT}/{folder}", exist_ok=True)
        shutil.copy(p, f"{OUTPUT}/{folder}/{os.path.basename(p)}")

    cm = confusion_matrix(trues, preds)

    print("\n===== STAGE 2 RESULTS (Prototype + Cosine) =====")
    print("Accuracy:", accuracy_score(trues, preds))
    print("Precision:", precision_score(trues, preds, average="macro"))
    print("Recall:", recall_score(trues, preds, average="macro"))
    print("F1:", f1_score(trues, preds, average="macro"))

    print("\nConfusion Matrix:")
    print(cm)

    history_path = os.path.join(OUTPUT, "history_stage2_prototype.txt")
    os.makedirs(OUTPUT, exist_ok=True)

    with open(history_path, "w") as f:
        f.write("===== STAGE 2 RESULTS (Prototype + Cosine) =====\n")
        f.write(f"Accuracy: {accuracy_score(trues, preds):.4f}\n")
        f.write(f"Precision: {precision_score(trues, preds, average='macro'):.4f}\n")
        f.write(f"Recall: {recall_score(trues, preds, average='macro'):.4f}\n")
        f.write(f"F1-score: {f1_score(trues, preds, average='macro'):.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")

    print(f"\nHistory saved to: {history_path}")

    # =========================
    # PLOT & SAVE CONFUSION MATRIX IMAGE (counts only, publication quality)
    # Font: Times New Roman, Size:14, Bold, No title
    # =========================
    class_names = CLASSES
    cm_array = np.array(cm)

    plt.figure(figsize=(8, 6))
    sns.set_style("white")

    # Professional research-paper colour map
    cmap = "YlGnBu"

    ax = sns.heatmap(
        cm_array,
        annot=False,
        fmt="d",
        cmap=cmap,
        cbar=True,
        linewidths=0.8,
        linecolor="black",
        square=False
    )

    # Annotate counts only (no percentages)
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            count = int(cm_array[i, j])

            cell_color = ax.collections[0].cmap(
                ax.collections[0].norm(cm_array[i, j])
            )

            r, g, b, _ = cell_color
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

            text_color = "white" if luminance < 0.55 else "black"

            ax.text(
                j + 0.5,
                i + 0.5,
                f"{count}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14,
                fontname="Times New Roman",
                fontweight="bold",
                color=text_color
            )

    # Axis labels
    ax.set_xlabel(
        "Predicted label",
        fontsize=14,
        fontname="Times New Roman",
        fontweight="bold"
    )

    ax.set_ylabel(
        "True label",
        fontsize=14,
        fontname="Times New Roman",
        fontweight="bold"
    )

    # Tick labels
    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
        fontsize=14,
        fontname="Times New Roman",
        fontweight="bold"
    )

    ax.set_yticklabels(
        class_names,
        rotation=0,
        fontsize=14,
        fontname="Times New Roman",
        fontweight="bold"
    )

    # Colorbar formatting
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    for label in cbar.ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")

    # No title added

    cm_path = os.path.join(OUTPUT, "confusion_matrix_stage2.png")

    plt.tight_layout()

    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"\nConfusion matrix image saved to: {cm_path}")
    print(f"History saved to: {history_path}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    split_dataset()
    create_support_query()
    model = train_model()
    evaluate(model)

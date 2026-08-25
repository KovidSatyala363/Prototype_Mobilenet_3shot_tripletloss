# =========================
# STAGE 3 (PURE ML - FINAL RESEARCH VERSION with 10 seeds, no CV)
# =========================

import os, random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================
DATASET = "datasets/stage3"
OUTPUT = "outputs/predicted_output_stage3"
MODEL_SAVE = "models"

CLASSES = [
    "unhealthy_blackspot_G",
    "unhealthy_brownspot_P",
    "unhealthy_damaged_G",
    "unhealthy_damaged_P"
]

# Exactly 10 seeds including 42
SEEDS = [0,1,2,3,4,5,6,7,8,42]

os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(MODEL_SAVE, exist_ok=True)

# =========================
# FEATURE EXTRACTION (HOG)
# =========================
def extract_features(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    return features.flatten()

# =========================
# LOAD DATA
# =========================
def load_data():
    X, y = [], []
    for idx, cls in enumerate(CLASSES):
        folder = os.path.join(DATASET, cls)
        for img in os.listdir(folder):
            path = os.path.join(folder, img)
            if not os.path.isfile(path):
                continue
            feat = extract_features(path)
            if feat is None:
                continue
            X.append(feat)
            y.append(idx)
    return np.array(X), np.array(y)

# =========================
# MAIN
# =========================
def main():

    X, y = load_data()

    if X.size == 0:
        raise RuntimeError(f"No data found in dataset path: {DATASET}")

    # ==========================================================
    # Models
    # ==========================================================
    models_dict = {
        "SVM": SVC(kernel='rbf', C=5, gamma='scale',
                   class_weight='balanced'),

        "KNN": KNeighborsClassifier(
            n_neighbors=3,
            weights='uniform',
            metric='euclidean'
        ),

        "RF": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            class_weight='balanced',
            random_state=42
        ),

        "DT": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        ),

        "XGB": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            objective='multi:softprob',
            num_class=4,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0
        )
    }

    # ==========================================================
    # Store results
    # ==========================================================

    all_results = {
        name:{
            "acc":[],
            "prec":[],
            "rec":[],
            "f1":[]
        }
        for name in models_dict
    }

    print("\n===== STAGE 3 RESULTS (PURE ML, 10 Seeds) =====")

    # ==========================================================
    # Training Loop
    # ==========================================================

    for seed in SEEDS:

        print(f"\n------ Seed {seed} ------")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=seed
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )

        for name, clf in models_dict.items():

            if name=="XGB":
                clf.fit(
                    X_train,
                    y_train,
                    sample_weight=sample_weights
                )
            else:
                clf.fit(
                    X_train,
                    y_train
                )

            preds = clf.predict(X_test)

            acc = accuracy_score(y_test,preds)

            prec = precision_score(
                y_test,
                preds,
                average="macro",
                zero_division=0
            )

            rec = recall_score(
                y_test,
                preds,
                average="macro",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                preds,
                average="macro",
                zero_division=0
            )

            all_results[name]["acc"].append(acc)
            all_results[name]["prec"].append(prec)
            all_results[name]["rec"].append(rec)
            all_results[name]["f1"].append(f1)

            print(
                f"{name} -> "
                f"Acc={acc:.3f}, "
                f"Prec={prec:.3f}, "
                f"Rec={rec:.3f}, "
                f"F1={f1:.3f}"
            )

            # ==================================================
            # Publication Quality Confusion Matrix
            # ==================================================

            cm = confusion_matrix(y_test,preds)

            plt.figure(figsize=(8,6))

            sns.set(style="white")

            cmap="YlGnBu"

            ax=sns.heatmap(
                cm,
                annot=False,
                fmt="d",
                cmap=cmap,
                linewidths=0.6,
                linecolor="gray",
                cbar=True,
                square=False
            )

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):

                    value=int(cm[i,j])

                    cell=ax.collections[0].cmap(
                        ax.collections[0].norm(cm[i,j])
                    )

                    r,g,b,_=cell

                    luminance=0.2126*r+0.7152*g+0.0722*b

                    txt="white" if luminance<0.55 else "black"

                    ax.text(
                        j+0.5,
                        i+0.5,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontname="Times New Roman",
                        weight="bold",
                        color=txt
                    )

            ax.set_xlabel(
                "Predicted label",
                fontsize=14,
                fontname="Times New Roman",
                weight="bold"
            )

            ax.set_ylabel(
                "True label",
                fontsize=14,
                fontname="Times New Roman",
                weight="bold"
            )

            ax.set_xticklabels(
                CLASSES,
                rotation=45,
                ha="right",
                fontsize=14,
                fontname="Times New Roman",
                weight="bold"
            )

            ax.set_yticklabels(
                CLASSES,
                rotation=0,
                fontsize=14,
                fontname="Times New Roman",
                weight="bold"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    OUTPUT,
                    f"{name}_cm_seed{seed}.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

    # ==========================================================
    # Summary Table
    # ==========================================================

    summary=[]

    for name in models_dict:

        summary.append([
            name,
            np.mean(all_results[name]["acc"]),
            np.std(all_results[name]["acc"]),
            np.mean(all_results[name]["prec"]),
            np.std(all_results[name]["prec"]),
            np.mean(all_results[name]["rec"]),
            np.std(all_results[name]["rec"]),
            np.mean(all_results[name]["f1"]),
            np.std(all_results[name]["f1"])
        ])

    df=pd.DataFrame(summary,columns=[
        "Model",
        "Acc_mean",
        "Acc_std",
        "Prec_mean",
        "Prec_std",
        "Rec_mean",
        "Rec_std",
        "F1_mean",
        "F1_std"
    ])

    df.to_csv(
        os.path.join(
            OUTPUT,
            "stage3_ml_results_10seeds.csv"
        ),
        index=False
    )

    # ==========================================================
    # Publication Quality F1 Bar Graph
    # ==========================================================

    models=[x[0] for x in summary]

    f1_means=[x[7] for x in summary]
    f1_std=[x[8] for x in summary]

    plt.figure(figsize=(8,6))

    plt.bar(
        models,
        f1_means,
        yerr=f1_std,
        capsize=6,
        color="#4C72B0",
        edgecolor="black",
        linewidth=1.2
    )
    plt.xlabel(
        "Machine Learning Models",
        fontsize=14,
        fontname="Times New Roman",
        weight="bold"
    )

    plt.ylabel(
        "F1-score",
        fontsize=14,
        fontname="Times New Roman",
        weight="bold"
    )

    plt.xticks(

        fontsize=14,
        fontname="Times New Roman",
        weight="bold"
    )

    plt.yticks(
        fontsize=14,
        fontname="Times New Roman",
        weight="bold"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT,
            "stage3_f1_bar.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # ==========================================================
    # Publication Quality Accuracy Bar Graph
    # ==========================================================

    acc_means=[x[1] for x in summary]
    acc_std=[x[2] for x in summary]

    plt.figure(figsize=(8,6))

    plt.bar(
        models,
        acc_means,
        yerr=acc_std,
        capsize=6,
        color="#55A868",
        edgecolor="black",
        linewidth=1.2
    )
    plt.xlabel(
        "Machine Learning Models",
        fontsize=14,
        fontname="Times New Roman",
        weight="bold"
    )

    plt.ylabel(
        "Accuracy",
        fontsize=14,
        fontname="Times New Roman",
        weight="bold"
    )

    plt.xticks(
        fontsize=14,
        fontname="Times New Roman",
        weight="bold"
    )

    plt.yticks(
        fontsize=14,
        fontname="Times New Roman",
        weight="bold"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT,
            "stage3_accuracy_bar.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


if __name__=="__main__":
    main()

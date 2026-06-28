"""
CIVICLEAR NLP Text Classification Training

Input CSV columns:
- text
- label

Output:
- models/civiclear_nlp_model.joblib
- reports/nlp_metrics.json
- reports/classification_report.txt
- reports/confusion_matrix.png
"""

from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # Accept common alternative column names
    rename_map = {}
    if "description" in df.columns and "text" not in df.columns:
        rename_map["description"] = "text"
    if "violation_type" in df.columns and "label" not in df.columns:
        rename_map["violation_type"] = "label"
    if "category" in df.columns and "label" not in df.columns:
        rename_map["category"] = "label"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. CSV must contain text,label")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df.dropna()
    df = df[(df["text"] != "") & (df["label"] != "")]
    return df


def plot_confusion_matrix(y_true, y_pred, labels, output_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="CIVICLEAR NLP Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train(args):
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    full_csv = Path(args.csv)
    train_csv = Path(args.train_csv) if args.train_csv else None
    valid_csv = Path(args.valid_csv) if args.valid_csv else None
    test_csv = Path(args.test_csv) if args.test_csv else None

    if train_csv and test_csv and train_csv.exists() and test_csv.exists():
        train_df = load_csv(train_csv)
        test_df = load_csv(test_csv)
        if valid_csv and valid_csv.exists():
            valid_df = load_csv(valid_csv)
            # Use train + valid for final training, test for evaluation
            train_df = pd.concat([train_df, valid_df], ignore_index=True)
    else:
        df = load_csv(full_csv)
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=df["label"],
        )

    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Label distribution:")
    print(train_df["label"].value_counts())

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=args.max_features,
        )),
        ("classifier", LogisticRegression(
            max_iter=args.max_iter,
            class_weight="balanced",
            solver="liblinear",
        )),
    ])

    model.fit(train_df["text"], train_df["label"])

    y_true = test_df["label"]
    y_pred = model.predict(test_df["text"])
    labels = sorted(train_df["label"].unique().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print(report_text)

    model_path = model_dir / "civiclear_nlp_model.joblib"
    joblib.dump(model, model_path)

    metrics_path = report_dir / "nlp_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "labels": labels,
            "classification_report": report_dict,
        }, f, indent=4)

    report_path = report_dir / "classification_report.txt"
    report_path.write_text(f"Accuracy: {accuracy * 100:.2f}%\n\n{report_text}", encoding="utf-8")

    cm_path = report_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, labels, cm_path)

    # Save sample predictions
    sample_texts = [
        "May nakaparadang sasakyan sa kalsada",
        "May basura na nakaharang sa sidewalk",
        "May hollow blocks at buhangin sa daan",
        "May vendor na nakaharang sa bangketa",
        "Malinis ang daan at walang obstruction",
    ]
    probabilities = model.predict_proba(sample_texts)
    predictions = model.predict(sample_texts)
    class_names = model.named_steps["classifier"].classes_

    sample_rows = []
    for text, pred, probs in zip(sample_texts, predictions, probabilities):
        confidence = float(max(probs) * 100)
        sample_rows.append({
            "text": text,
            "prediction": pred,
            "confidence": round(confidence, 2),
        })

    sample_path = report_dir / "sample_predictions.csv"
    pd.DataFrame(sample_rows).to_csv(sample_path, index=False)

    print("\nSaved files:")
    print("-", model_path)
    print("-", metrics_path)
    print("-", report_path)
    print("-", cm_path)
    print("-", sample_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/civiclear_nlp_text_dataset_500.csv")
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--valid-csv", default="data/valid.csv")
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(args)

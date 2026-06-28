from pathlib import Path
import argparse
import joblib


def predict(model_path: Path, texts):
    model = joblib.load(model_path)
    preds = model.predict(texts)
    probs = model.predict_proba(texts)

    for text, pred, prob in zip(texts, preds, probs):
        confidence = max(prob) * 100
        print("Text:", text)
        print("Prediction:", pred)
        print("Confidence:", round(confidence, 2), "%")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/civiclear_nlp_model.joblib")
    args = parser.parse_args()

    sample_texts = [
        "May sasakyan na nakaparada sa daan",
        "May tambak na basura sa gilid ng kalsada",
        "May construction materials na nakaharang sa sidewalk",
        "Walang violation, malinis ang daan",
    ]

    predict(Path(args.model), sample_texts)

"""
Optional: Import rows from a Hugging Face dataset and convert them to CIVICLEAR format.

Important:
- This does NOT automatically know your DILG/CIVICLEAR labels.
- Use this only if you find a Hugging Face dataset that has relevant text and labels.
- You must map the dataset's labels to your CIVICLEAR labels manually.
"""

from pathlib import Path
import argparse
import pandas as pd
from datasets import load_dataset

VALID_LABELS = {
    "construction_materials",
    "garbage_debris",
    "illegal_parking",
    "road_obstruction",
    "sidewalk_obstruction",
    "no_violation",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True, help="Example: username/dataset_name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output", default="data/huggingface_imported_rows.csv")
    args = parser.parse_args()

    ds = load_dataset(args.dataset_name, split=args.split)
    df = pd.DataFrame(ds)

    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column: {args.text_column}")
    if args.label_column not in df.columns:
        raise ValueError(f"Missing label column: {args.label_column}")

    out = pd.DataFrame({
        "text": df[args.text_column].astype(str),
        "label": df[args.label_column].astype(str),
    })

    print("Original labels found:")
    print(out["label"].value_counts())
    print("\nBefore training, manually map these labels to:")
    print(sorted(VALID_LABELS))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()

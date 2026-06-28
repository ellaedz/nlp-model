# CIVICLEAR NLP Training Package

This package is for training the **text/NLP model** for CIVICLEAR / DILG-RC.

The NLP model reads the citizen's report description and predicts the possible violation type.

Example:

```text
Input: May sasakyan na nakaparada sa kalsada
Output: illegal_parking
Confidence: 88.42%
```

## What is included

```text
civiclear-nlp-training/
├── data/
│   ├── civiclear_nlp_text_dataset_500.csv
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   └── dataset_summary.csv
├── notebooks/
│   └── CIVICLEAR_NLP_Training_Colab.ipynb
├── api/
│   └── nlp_api.py
├── train_nlp.py
├── test_nlp_model.py
├── optional_huggingface_merge.py
├── requirements.txt
└── README.md
```

## Dataset labels

The CSV uses the same label names as the image/CV model:

```text
construction_materials
garbage_debris
illegal_parking
road_obstruction
sidewalk_obstruction
no_violation
```

## Important note about Hugging Face

The included 500-row CSV is a **synthetic prototype dataset** for your exact thesis labels.

Hugging Face is useful if you later find a dataset with similar traffic, obstruction, or complaint text. However, most public Hugging Face NLP datasets will not already use your exact CIVICLEAR/DILG labels, so you must manually map them before training.

Use `optional_huggingface_merge.py` only if you already found a Hugging Face dataset that you want to import.

## Google Colab training guide

### Step 1: Upload this folder to Google Drive

Upload the whole folder:

```text
civiclear-nlp-training
```

Recommended location:

```text
MyDrive/civiclear-nlp-training
```

### Step 2: Open the notebook

Open:

```text
notebooks/CIVICLEAR_NLP_Training_Colab.ipynb
```

### Step 3: Run all cells

The notebook will:

```text
1. Install dependencies
2. Mount Google Drive
3. Load the CSV dataset
4. Train the NLP model
5. Evaluate the model
6. Save the trained model and reports
```

## Local / VS Code training guide

Open terminal inside the folder:

```bash
cd civiclear-nlp-training
```

Create environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Activate on Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train_nlp.py
```

Test:

```bash
python test_nlp_model.py
```

## Output files after training

After training, you should get:

```text
models/civiclear_nlp_model.joblib
reports/nlp_metrics.json
reports/classification_report.txt
reports/confusion_matrix.png
reports/sample_predictions.csv
```

The most important file is:

```text
civiclear_nlp_model.joblib
```

This is the NLP version of the image model's `best.pt`.

## Run NLP API

After training, run:

```bash
uvicorn api.nlp_api:app --host 127.0.0.1 --port 8002 --reload
```

Test endpoint:

```text
POST http://127.0.0.1:8002/predict-text
```

Body:

```json
{
  "text": "May sasakyan na nakaparada sa kalsada"
}
```

Response:

```json
{
  "detected": true,
  "violation_type": "illegal_parking",
  "confidence": 88.42,
  "message": "Text prediction completed"
}
```

## Laravel integration later

Laravel can send the citizen description to the NLP API.

Example result shown in web/mobile:

```text
Text Analysis: Illegal Parking
Text Confidence: 88.42%
Policy Category: Road Clearing Obstruction
Status: Pending Review
```

## What to send back to the thesis leader

Send these files:

```text
models/civiclear_nlp_model.joblib
reports/nlp_metrics.json
reports/classification_report.txt
reports/confusion_matrix.png
reports/sample_predictions.csv
```

## Notes

This is a starter/prototype NLP model using TF-IDF + Logistic Regression. It is fast and easy to explain for thesis defense. You can improve it later with a transformer model if needed.

# CIVICLEAR NLP Model Training

This repository is for training the **NLP/text classification model** of the CIVICLEAR / DILG-RC Road Clearing Violation Reporting System.

The NLP model reads a citizen's report description and predicts the possible violation type.

Example:

```text
Input: May sasakyan na nakaparada sa kalsada
Output: illegal_parking
Confidence: 88.42%
```

---

## Repository Structure

```text
nlp-model/
├── api/
│   └── nlp_api.py
├── data/
│   ├── civiclear_nlp_text_dataset_500.csv
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   └── dataset_summary.csv
├── models/
│   └── .gitkeep
├── notebooks/
│   └── CIVICLEAR_NLP_Training_Colab.ipynb
├── reports/
│   └── .gitkeep
├── train_nlp.py
├── test_nlp_model.py
├── optional_huggingface_merge.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset Labels

The NLP model uses the same label categories as the CV/image model:

```text
construction_materials
garbage_debris
illegal_parking
road_obstruction
sidewalk_obstruction
no_violation
```

---

## Important Notes

The `data/` folder contains the text dataset used for training.

The `models/` folder is empty before training. After training, it will contain the trained NLP model:

```text
models/civiclear_nlp_model.joblib
```

The `reports/` folder is empty before training. After training, it will contain evaluation results:

```text
reports/nlp_metrics.json
reports/classification_report.txt
reports/confusion_matrix.png
reports/sample_predictions.csv
```

Do not delete `.gitkeep` files. They are only placeholders so GitHub will show the empty folders.

---

# Option 1 — Train Using Google Colab

This is the recommended method.

## Step 1 — Open the Notebook

Open:

```text
notebooks/CIVICLEAR_NLP_Training_Colab.ipynb
```

If the notebook is not uploaded yet, upload it first inside the `notebooks/` folder.

## Step 2 — Set Runtime

In Google Colab:

```text
Runtime → Change runtime type → Hardware accelerator
```

For this NLP model, GPU is optional. CPU is okay because the model uses TF-IDF + Logistic Regression.

## Step 3 — Run All Cells

The notebook will:

```text
1. Install dependencies
2. Load the CSV dataset
3. Train the NLP model
4. Test the NLP model
5. Save the trained model
6. Save evaluation reports
```

## Step 4 — Download the Output Files

After training, download/send these files:

```text
models/civiclear_nlp_model.joblib
reports/nlp_metrics.json
reports/classification_report.txt
reports/confusion_matrix.png
reports/sample_predictions.csv
```

The most important file is:

```text
models/civiclear_nlp_model.joblib
```

This is the trained NLP model.

---

# Option 2 — Train Locally Using VS Code

Use this if you want to train on your computer.

## Step 1 — Clone the Repository

```bash
git clone https://github.com/ellaedz/nlp-model.git
cd nlp-model
```

## Step 2 — Create Virtual Environment

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

## Step 3 — Install Requirements

```bash
pip install -r requirements.txt
```

## Step 4 — Train the NLP Model

```bash
python train_nlp.py
```

This will create:

```text
models/civiclear_nlp_model.joblib
reports/nlp_metrics.json
reports/classification_report.txt
reports/confusion_matrix.png
reports/sample_predictions.csv
```

## Step 5 — Test the Trained Model

```bash
python test_nlp_model.py
```

---

# How to Know if Training Worked

Training is successful if these files are created:

```text
models/civiclear_nlp_model.joblib
reports/nlp_metrics.json
reports/classification_report.txt
reports/confusion_matrix.png
reports/sample_predictions.csv
```

Sample output should look like:

```text
Text: May nakaparadang sasakyan sa kalsada
Prediction: illegal_parking
Confidence: 88.42%
```

---

# How the NLP Model Will Be Used Later

After training, the Laravel system will send the citizen's report description to the NLP API.

Example request:

```json
{
  "text": "May sasakyan na nakaharang sa kalsada"
}
```

Example response:

```json
{
  "detected": true,
  "violation_type": "illegal_parking",
  "confidence": 88.42,
  "message": "Text prediction completed"
}
```

The web dashboard and mobile app can display:

```text
Text Analysis: Illegal Parking
Text Confidence: 88.42%
Policy Category: Road Clearing Obstruction
Status: Pending Review
```

---

# Run the NLP API

After training, run:

```bash
uvicorn api.nlp_api:app --host 127.0.0.1 --port 8002 --reload
```

Endpoint:

```text
POST http://127.0.0.1:8002/predict-text
```

Body:

```json
{
  "text": "May sasakyan na nakaparada sa kalsada"
}
```

---

# Files to Send Back After Training

Send these files to the thesis system developer:

```text
models/civiclear_nlp_model.joblib
reports/nlp_metrics.json
reports/classification_report.txt
reports/confusion_matrix.png
reports/sample_predictions.csv
```

The trained model file is:

```text
civiclear_nlp_model.joblib
```

Do not train again after this file is created. The system will only use it for prediction/inference.

---

# Simple Explanation

```text
CSV dataset = used to train the NLP model
civiclear_nlp_model.joblib = trained NLP model
Laravel/mobile app = uses the trained model later
```

Training happens once.

After training, the system only predicts the violation type from citizen text reports.

# EE559 Final Project: AI-Generated Text Detection

This repository contains the final program implementation for an EE559 machine learning project. The goal of the project is to classify text samples as either human-written or AI-generated using several machine learning and deep learning approaches. The dataset is from Kaggle: [AI and Human Text Dataset](https://www.kaggle.com/datasets/hasanyiitakbulut/ai-and-human-text-dataset/data)

The project compares four modeling strategies:

1. TF-IDF + Logistic Regression baseline
2. Bayesian Logistic Regression with Laplace approximation
3. Frozen DistilBERT feature extractor with a neural classification head
4. Sentence-BERT embeddings + CatBoost classifier

## Project Overview

AI-generated text detection is formulated as a binary text classification problem. Each input sample is a text document, and the model predicts whether the text belongs to the human-written class or the AI-generated class.

The dataset is expected to be split into three CSV files:

```text
train.csv
val.csv
test.csv
```

Each CSV file should contain at least the following columns:

| Column | Description |
|---|---|
| `text` | Input text sample |
| `label` | Binary class label |

The label values should be encoded consistently across all three files. For example:

```text
0 = human-written text
1 = AI-generated text
```

## Repository Structure

```text
.
├── README.md
├── train.csv
├── val.csv
├── test.csv
├── linear_regression.ipynb
├── bayesian_model.ipynb
├── bert_model.ipynb
└── catboost.py
```

## Models

### 1. TF-IDF + Logistic Regression

File:

```text
linear_regression.ipynb
```

This notebook implements a traditional machine learning baseline.

Main steps:

1. Load `train.csv`, `val.csv`, and `test.csv`.
2. Extract the `text` and `label` columns.
3. Convert text into TF-IDF features.
4. Train a logistic regression classifier.
5. Evaluate the model on the validation and test sets.

Key configuration:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
```

This model is useful as a baseline because it is simple, fast, and interpretable. It helps determine whether more advanced models provide meaningful improvement.

### 2. Bayesian Logistic Regression

File:

```text
bayesian_model.ipynb
```

This notebook implements Bayesian logistic regression using a maximum a posteriori estimate and Laplace approximation.

Main steps:

1. Load the train, validation, and test datasets.
2. Convert text into TF-IDF features.
3. Reduce TF-IDF dimensionality using Truncated SVD.
4. Standardize the dense feature vectors.
5. Add a bias term.
6. Optimize the negative log posterior with BFGS.
7. Compute the posterior covariance using the Hessian.
8. Generate Bayesian predictive probabilities.
9. Evaluate the model on train, validation, and test sets.

The model uses a Gaussian prior on the weight vector:

```python
alpha = 1.0
```

The negative log posterior combines the logistic regression negative log likelihood with an L2-style prior term. The Laplace approximation is then used to estimate the posterior covariance around the MAP solution.

This model is useful because it provides a probabilistic interpretation of logistic regression and incorporates uncertainty into the prediction process.

### 3. DistilBERT Classifier

File:

```text
bert_model.ipynb
```

This notebook implements a transformer-based classifier using DistilBERT.

Main steps:

1. Load and combine the training and validation data.
2. Tokenize text samples using the DistilBERT tokenizer.
3. Use `distilbert-base-uncased` as the language model backbone.
4. Freeze the DistilBERT parameters.
5. Train a linear classification head on top of the `[CLS]` representation.
6. Evaluate the model on the test set.

Key configuration:

```python
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
MAX_LEN = 256
```

The DistilBERT backbone is frozen in this implementation, meaning only the final classifier layer is trained. This reduces training cost and makes the model easier to run on limited hardware.

The notebook also supports Apple Silicon acceleration through PyTorch MPS when available.

### 4. Sentence-BERT + CatBoost

File:

```text
catboost.py
```

This script implements a hybrid embedding-based gradient boosting model.

Main steps:

1. Load `train.csv`, `val.csv`, and `test.csv`.
2. Fill missing text values.
3. Encode labels using `LabelEncoder`.
4. Convert text into dense sentence embeddings using Sentence-BERT.
5. Train a CatBoost classifier.
6. Evaluate the model using accuracy, precision, recall, F1-score, AUC, classification report, and confusion matrix.
7. Save the trained CatBoost model and preprocessing objects.

Sentence embedding model:

```python
all-MiniLM-L6-v2
```

CatBoost configuration:

```python
CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    auto_class_weights="Balanced",
    verbose=100,
    od_type="Iter",
    od_wait=100
)
```

Saved outputs:

```text
catboost_sbert_model.cbm
label_encoder.pkl
sentence_transformer.pkl
```

This model combines semantic sentence embeddings with a strong tabular-style classifier. It is often more effective than plain TF-IDF when the distinction between classes depends on sentence-level meaning rather than only surface-level word frequency.

## Installation

Create and activate a Python environment:

```bash
python -m venv venv
source venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install numpy pandas scipy scikit-learn torch transformers tqdm sentence-transformers catboost joblib
```

If you are using Jupyter Notebook:

```bash
pip install notebook ipykernel
```

## How to Run

### Step 1: Prepare the dataset

Place the following files in the project root directory:

```text
train.csv
val.csv
test.csv
```

Each file should include:

```text
text,label
```

Example:

```csv
text,label
"This is a sample text.",0
"This paragraph was generated by an AI model.",1
```

### Step 2: Run the baseline model

Open and run:

```text
linear_regression.ipynb
```

This gives a quick baseline using TF-IDF and logistic regression.

### Step 3: Run the Bayesian model

Open and run:

```text
bayesian_model.ipynb
```

This trains the Bayesian logistic regression model and reports train, validation, and test performance.

### Step 4: Run the DistilBERT model

Open and run:

```text
bert_model.ipynb
```

This trains a frozen DistilBERT classifier and evaluates it on the test set.

### Step 5: Run the CatBoost model

Run the following command:

```bash
python catboost.py
```

After training, the script saves the model and preprocessing files:

```text
catboost_sbert_model.cbm
label_encoder.pkl
sentence_transformer.pkl
```

## Evaluation Metrics

The models are evaluated using standard binary classification metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | Among predicted positive samples, how many are correct |
| Recall | Among actual positive samples, how many are found |
| F1-score | Harmonic mean of precision and recall |
| AUC | Ranking quality based on predicted probabilities |
| Confusion Matrix | Counts of true positives, false positives, true negatives, and false negatives |

The CatBoost model reports the most complete set of metrics, including AUC and confusion matrices for both validation and test sets.

## Method Comparison

| Model | Feature Representation | Classifier | Strength |
|---|---|---|---|
| TF-IDF + Logistic Regression | Sparse word and n-gram features | Logistic Regression | Fast and interpretable baseline |
| Bayesian Logistic Regression | TF-IDF + SVD dense features | Bayesian Logistic Regression | Adds uncertainty-aware prediction |
| DistilBERT Classifier | Transformer contextual embeddings | Linear neural classifier | Captures contextual language patterns |
| Sentence-BERT + CatBoost | Sentence-level semantic embeddings | CatBoost | Combines semantic features with strong boosting |

## Notes

- The code assumes that the dataset files are located in the same directory as the notebooks and Python script.
- The text column must be named `text`.
- The label column must be named `label`.
- For the BERT model, a GPU or Apple Silicon MPS device can speed up training, but CPU execution is also supported.
- For the CatBoost model, the trained model and preprocessing objects are saved after training.

## Possible Future Improvements

Several improvements could be added in future work:

1. Fine-tune the full DistilBERT model instead of freezing the backbone.
2. Add hyperparameter tuning for CatBoost and logistic regression.
3. Compare more transformer models such as BERT, RoBERTa, or DeBERTa.
4. Add cross-validation for more reliable performance estimation.
5. Save all model results into a single comparison table.
6. Add an inference script for predicting the label of new text samples.
7. Add data analysis plots showing text length, class balance, and feature distributions.

## Conclusion

This project investigates multiple approaches for detecting AI-generated text. The baseline TF-IDF logistic regression model provides a simple reference point, while the Bayesian model adds probabilistic interpretation. The DistilBERT model introduces contextual language representation, and the Sentence-BERT + CatBoost model combines semantic embeddings with a powerful gradient boosting classifier.

Together, these models provide a comprehensive comparison between traditional machine learning, Bayesian learning, transformer-based representation learning, and embedding-based ensemble classification for AI-generated text detection.
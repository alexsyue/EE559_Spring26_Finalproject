import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib



RANDOM_SEED = 42
TEXT_COL = "text"
LABEL_COL = "label"


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")
for df in [train_df, val_df, test_df]:
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

le = LabelEncoder()
y_train = le.fit_transform(train_df[LABEL_COL])
y_val = le.transform(val_df[LABEL_COL])
y_test = le.transform(test_df[LABEL_COL])

embedder = SentenceTransformer(EMBED_MODEL_NAME)

X_train = embedder.encode(
    train_df[TEXT_COL].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

X_val = embedder.encode(
    val_df[TEXT_COL].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

X_test = embedder.encode(
    test_df[TEXT_COL].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_SEED,
    auto_class_weights="Balanced",  # 类别不平衡时很有用
    verbose=100,
    od_type="Iter",
    od_wait=100
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)


val_proba = model.predict_proba(X_val)[:, 1]
val_pred = (val_proba >= 0.5).astype(int)

class_names = [str(c) for c in le.classes_]

print("\n===== Validation Metrics =====")
print("Accuracy :", accuracy_score(y_val, val_pred))
print("Precision:", precision_score(y_val, val_pred))
print("Recall   :", recall_score(y_val, val_pred))
print("F1       :", f1_score(y_val, val_pred))
print("AUC      :", roc_auc_score(y_val, val_proba))
print("\nClassification Report:")
print(classification_report(y_val, val_pred, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))


test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= 0.5).astype(int)

print("\n===== Test Metrics =====")
print("Accuracy :", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred))
print("Recall   :", recall_score(y_test, test_pred))
print("F1       :", f1_score(y_test, test_pred))
print("AUC      :", roc_auc_score(y_test, test_proba))
print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))


model.save_model("catboost_sbert_model.cbm")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(embedder, "sentence_transformer.pkl")

print("\n模型已保存：catboost_sbert_model.cbm")
print("标签编码器已保存：label_encoder.pkl")
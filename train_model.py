import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

DATA_PATH = r"C:\Users\acer\Downloads\fake_job_postings.csv"
MODEL_OUT = "fake_job_model.joblib"


def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_text_column(df: pd.DataFrame) -> pd.Series:
    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    available = [c for c in text_cols if c in df.columns]
    if not available:
        raise ValueError(f"Expected text columns not found. Columns: {df.columns.tolist()}")

    combined = df[available].fillna("").agg(" ".join, axis=1)
    return combined.apply(clean_text)


def choose_threshold_for_precision(y_true, y_proba, target_precision=0.90):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # thresholds length is len(precisions)-1
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    ok = precisions >= target_precision
    if not np.any(ok):
        return None, None, None

    # pick the threshold that achieves the target precision with best recall
    best_idx = np.argmax(recalls[ok])
    thr = thresholds[ok][best_idx]
    return float(thr), float(precisions[ok][best_idx]), float(recalls[ok][best_idx])


def main():
    df = pd.read_csv(DATA_PATH)

    if "fraudulent" not in df.columns:
        raise ValueError("Target column 'fraudulent' not found.")

    X_text = build_text_column(df)
    y = df["fraudulent"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=60000,   # keeps it fast + helps convergence
        )),
        ("clf", LogisticRegression(
            solver="liblinear",     # change solver to avoid the saga convergence issues
            class_weight="balanced",
            C=1.0,
            max_iter=10000,
            random_state=42
            # NOTE: do NOT pass "penalty" -> avoids your sklearn 1.8 warning
        ))
    ])

    model.fit(X_train, y_train)

    clf = model.named_steps["clf"]
    # liblinear gives n_iter_ as an array; show max iterations used
    n_iter_used = int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else -1
    print(f"\nSolver iterations used: {n_iter_used} (max_iter={clf.max_iter})")

    y_proba = model.predict_proba(X_test)[:, 1]

    # Default threshold = 0.5
    y_pred_default = (y_proba >= 0.5).astype(int)
    print("\n=== Evaluation @ threshold 0.50 (default) ===")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_default))
    print(classification_report(y_test, y_pred_default, digits=4))

    roc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC (Average Precision): {ap:.4f}")

    # Choose threshold for high precision (your requirement)
    target_precision = 0.90
    thr, prec, rec = choose_threshold_for_precision(y_test, y_proba, target_precision)

    if thr is None:
        print(f"\nCould not reach precision ≥ {target_precision:.2f}; using threshold=0.50")
        chosen_thr = 0.5
    else:
        print(f"\n=== Chosen threshold for precision ≥ {target_precision:.2f} ===")
        print(f"Threshold: {thr:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        chosen_thr = thr

        y_pred_tuned = (y_proba >= chosen_thr).astype(int)
        print("\nEvaluation @ tuned threshold")
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_tuned))
        print(classification_report(y_test, y_pred_tuned, digits=4))

    artifact = {"pipeline": model, "threshold": chosen_thr}
    joblib.dump(artifact, MODEL_OUT)
    print(f"\nSaved: {MODEL_OUT}")


if __name__ == "__main__":
    main()
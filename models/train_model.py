"""Phases 4-6 - Train, evaluate and save the career prediction model.

Pipeline design
---------------
A student's profile has two useful signals:
    * skills   -> free text  -> TF-IDF features
    * cgpa     -> a number    -> scaled numeric feature
We combine them with a ColumnTransformer, then feed the result to a
classifier. Wrapping everything in a single sklearn Pipeline means the SAME
preprocessing is applied automatically at prediction time - we only have to
ship one object.

Phase 4 (Train)   : fit three candidate models.
Phase 5 (Evaluate): compare them with cross-validation + a hold-out test set,
                    print a classification report and save a confusion matrix.
Phase 6 (Save)    : pickle the best pipeline (with metadata) to career_model.pkl.

Run:
    python -m models.train_model
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend - we only save PNGs, never show windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.config import CAREER_DATASET_PATH, CAREERS, MODEL_PATH, MODELS_DIR
from utils.data_cleaning import clean_dataframe, skills_to_list


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and clean the career dataset.

    Returns:
        X: DataFrame with feature columns ['skills', 'cgpa'].
        y: Series of career labels.
    """
    df = pd.read_csv(CAREER_DATASET_PATH)
    df = clean_dataframe(df)
    X = df[["skills", "cgpa"]]
    y = df["career"]
    return X, y


def build_pipeline(classifier) -> Pipeline:
    """Wrap preprocessing + a classifier into one pipeline.

    We use `skills_to_list` (from data_cleaning) as the TF-IDF tokenizer so
    multi-word skills like "machine learning" stay intact as single tokens.
    Because that function lives in an importable module, the pipeline can be
    safely pickled and reloaded later.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "skills",
                TfidfVectorizer(
                    tokenizer=skills_to_list,
                    token_pattern=None,
                    lowercase=False,
                ),
                "skills",  # 1D text column
            ),
            ("cgpa", StandardScaler(), ["cgpa"]),  # 2D numeric column
        ]
    )
    return Pipeline([("prep", preprocessor), ("clf", classifier)])


def candidate_models() -> Dict[str, Pipeline]:
    """Return the candidate models we want to compare."""
    return {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, C=5.0)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=42)
        ),
        "SVM (RBF)": build_pipeline(
            SVC(kernel="rbf", C=5.0, probability=True, random_state=42)
        ),
    }


def train_and_select() -> None:
    """Full Phase 4-6 workflow: train, evaluate, select best, save."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results: Dict[str, float] = {}
    fitted: Dict[str, Pipeline] = {}

    print("=" * 60)
    print("PHASE 4/5 - Training and cross-validating candidate models")
    print("=" * 60)
    for name, pipe in candidate_models().items():
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        pipe.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, pipe.predict(X_test))
        results[name] = cv.mean()
        fitted[name] = pipe
        print(f"{name:22} | CV acc: {cv.mean():.3f} +/- {cv.std():.3f} "
              f"| test acc: {test_acc:.3f}")

    best_name = max(results, key=results.get)
    best_pipe = fitted[best_name]
    print(f"\nBest model: {best_name} (CV acc {results[best_name]:.3f})")

    # ------------------------------------------------------------------
    # Phase 5 - detailed evaluation of the winner
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"PHASE 5 - Evaluation report for {best_name}")
    print("=" * 60)
    y_pred = best_pipe.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save a confusion matrix image for the report / documentation.
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=CAREERS, xticks_rotation=45, ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix -> {cm_path}")

    # ------------------------------------------------------------------
    # Phase 6 - persist the winning pipeline + metadata with joblib/pickle
    # ------------------------------------------------------------------
    artifact = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "careers": sorted(y.unique().tolist()),
        "cv_accuracy": results[best_name],
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nPHASE 6 - Saved model artifact -> {MODEL_PATH}")


if __name__ == "__main__":
    train_and_select()

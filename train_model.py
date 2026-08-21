"""
train_model.py — Trains an intent classifier from intents.json.

Pipeline: TF-IDF vectorization -> Logistic Regression classifier.
Logistic Regression is chosen over something heavier (deep learning) because
with a small, hand-labeled intents dataset (this scale of project), a simple
linear classifier on TF-IDF features is fast, interpretable, and avoids
overfitting that a neural network would be prone to on so little data —
a deliberate, defensible choice worth explaining in an interview.
"""

import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

from preprocess import tokenize_and_lemmatize


def load_training_data(path="intents.json"):
    with open(path) as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(tokenize_and_lemmatize(pattern))
            labels.append(intent["tag"])
    return texts, labels, data


def train():
    texts, labels, raw_data = load_training_data()
    print(f"Loaded {len(texts)} training examples across {len(set(labels))} intents")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    # Compare two candidate classifiers via cross-validation on this small
    # dataset. LinearSVC often edges out Logistic Regression on short-text
    # TF-IDF classification because it maximizes margin rather than
    # likelihood, which tends to generalize a bit better with sparse,
    # high-dimensional bag-of-words/n-gram features.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5.0),
        "Linear SVM": LinearSVC(C=1.0, max_iter=5000),
    }
    cv_results = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X, y, cv=skf)
        cv_results[name] = scores.mean()
        print(f"5-fold CV accuracy — {name}: {scores.mean():.1%} (+/- {scores.std():.1%})")

    best_name = max(cv_results, key=cv_results.get)
    print(f"\nSelected model: {best_name}")

    # Train/test split for a held-out classification report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if best_name == "Linear SVM":
        clf = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=3)
    else:
        clf = LogisticRegression(max_iter=1000, C=5.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\nHeld-out test accuracy: {accuracy_score(y_test, y_pred):.1%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Final model trained on ALL data (for actual deployment/inference —
    # the held-out split above is only to report honest metrics).
    # Wrapped in CalibratedClassifierCV when using SVM so predict_proba
    # is available for confidence-based fallback at inference time.
    if best_name == "Linear SVM":
        final_clf = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=3)
    else:
        final_clf = LogisticRegression(max_iter=1000, C=5.0)
    final_clf.fit(X, y)

    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(final_clf, "intent_classifier.pkl")

    # Save responses lookup separately (not part of the model — this is
    # business content, kept out of the ML artifact)
    responses = {i["tag"]: i["responses"] for i in raw_data["intents"]}
    with open("responses.json", "w") as f:
        json.dump(responses, f, indent=2)

    print("\nSaved: vectorizer.pkl, intent_classifier.pkl, responses.json")
    return cv_results[best_name]


if __name__ == "__main__":
    train()

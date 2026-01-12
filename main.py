#!/usr/bin/env python3

# -------------------------------------------------------------------
# Environment (must be set before ML imports)
# -------------------------------------------------------------------
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -------------------------------------------------------------------
# Standard library
# -------------------------------------------------------------------
import sys
import argparse
import re
import numpy as np
from typing import Optional

# -------------------------------------------------------------------
# Paths & Labels
# -------------------------------------------------------------------
BASE_DIR = os.getcwd()

DISTILBERT_PATH = os.path.join(BASE_DIR, "models", "distilbert")
LINEARSVC_PATH = os.path.join(BASE_DIR, "models", "linearsvc")

BIAS_LABELS = ["Center", "Lean Left", "Lean Right", "Left", "Right"]


# -------------------------------------------------------------------
# Argument Parsing (FIRST)
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        prog="bias_predictor",
        description=(
            "Predict political bias of a news article.\n\n"
            "Models:\n"
            "- DistilBERT : transformer, no lemmatization\n"
            "- LinearSVC  : TF-IDF + lemmatization\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--model",
        choices=["distilbert", "linearsvc", "both"],
        default="both",
        help="Which model to run (default: both)",
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Path to text file (otherwise reads from stdin)",
    )

    return parser.parse_args()

# -------------------------------------------------------------------
# Clear Screen
# -------------------------------------------------------------------
def clear_screen():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

# -------------------------------------------------------------------
# Input
# -------------------------------------------------------------------
def read_input(file_path: str | None) -> str:
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    clear_screen()
    print("Enter the content of article: \n"+'\033[93m'+"(press Enter, then Ctrl+Z and finally Enter again)\n"+'\033[0m')
    return sys.stdin.read()


# -------------------------------------------------------------------
# Text Cleaning
# -------------------------------------------------------------------
URL_RE = re.compile(r"http\S+|www\S+")
MULTI_SPACE_RE = re.compile(r"\s+")
HTML_RE = re.compile(r"<.*?>")
REPLACE_BY_SPACE_RE = re.compile(r"[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile(r"[^0-9a-z #+_]")


def basic_clean(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = REPLACE_BY_SPACE_RE.sub(" ", text)
    text = BAD_SYMBOLS_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def lemmatize(text: str) -> str:
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    doc = nlp(text)

    tokens = [
        t.lemma_
        for t in doc
        if t.is_alpha and not t.is_stop and len(t.text) > 3
    ]
    return " ".join(tokens)


# -------------------------------------------------------------------
# Model Loaders (Lazy)
# -------------------------------------------------------------------
def load_distilbert():
    import torch
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
    model.eval()
    return tokenizer, model


def load_linearsvc():
    import joblib

    vec = joblib.load(os.path.join(LINEARSVC_PATH, "tfidf_vectorizer.pkl"))
    model = joblib.load(os.path.join(LINEARSVC_PATH, "linearsvc_model.pkl"))
    return vec, model


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------
def predict_distilbert(text, tokenizer, model):
    import torch

    cleaned = basic_clean(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].tolist()
    pred = BIAS_LABELS[int(np.argmax(probs))]
    return probs, pred


def predict_linearsvc(text, vectorizer, model):
    from sklearn.preprocessing import LabelEncoder

    cleaned = lemmatize(basic_clean(text))
    X = vectorizer.transform([cleaned])

    scores = model.decision_function(X)[0]
    probs = np.exp(scores) / np.sum(np.exp(scores))

    le = LabelEncoder()
    le.fit(BIAS_LABELS)

    pred_enc = model.predict(X)[0]
    pred = le.inverse_transform([pred_enc])[0]
    return probs.tolist(), pred


# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------
def print_result(name, probs, pred):
    print(f"\nModel: {name}")
    print('\033[90m')
    for label, p in zip(BIAS_LABELS, probs):
        print(f"{label:<12}: {p:.4f}")
    print('\033[0m')
    print(f"Bias: {pred}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = parse_args()
    text = read_input(args.file)

    outputs = []

    if not text.strip():
        raise ValueError("Empty input")

    if args.model in ("distilbert", "both"):
        tokenizer, model = load_distilbert()
        probs, pred = predict_distilbert(text, tokenizer, model)
        outputs.append(("DistilBERT", probs, pred))

    if args.model in ("linearsvc", "both"):
        vectorizer, model = load_linearsvc()
        probs, pred = predict_linearsvc(text, vectorizer, model)
        outputs.append(("LinearSVC", probs, pred))

    clear_screen()

    for (model, probs, pred) in outputs:
        print_result(model,probs,pred)

if __name__ == "__main__":
    main()
"""
Spam Email Classifier — Naive Bayes from Scratch
Classifies emails as spam or ham using a Multinomial Naive Bayes model.
Built without sklearn — implements bag-of-words, TF weighting, and
Laplace smoothing by hand so the math is fully visible.

Usage:
  python spam_classifier.py              # demo with built-in dataset
  python spam_classifier.py --classify "Win a free prize now!!!"
  python spam_classifier.py --interactive
"""

import re
import sys
import math
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ── Text preprocessing ────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "this", "that", "be",
    "are", "was", "were", "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "i", "you", "we", "they",
    "he", "she", "your", "our", "my", "me", "us", "them", "from"
}


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\$[\d,.]+", " MONEY ", text)
    text = re.sub(r"\d+%", " PERCENT ", text)
    text = re.sub(r"!{2,}", " MULTIEXCLAIM ", text)
    text = re.sub(r"[^a-z\s_]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ── Naive Bayes ───────────────────────────────────────────────────────────────

@dataclass
class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes with Laplace smoothing.
    P(class | words) ∝ P(class) × ∏ P(word | class)
    Computed in log-space to avoid underflow.
    """
    alpha: float = 1.0          # Laplace smoothing

    # Learned parameters
    log_prior: Dict[str, float] = field(default_factory=dict)
    log_likelihood: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    vocab: set = field(default_factory=set)
    classes: List[str] = field(default_factory=list)
    word_counts: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    class_totals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def fit(self, X: List[str], y: List[str]):
        class_doc_counts: Dict[str, int] = defaultdict(int)
        self.vocab = set()

        for text, label in zip(X, y):
            tokens = tokenize(text)
            class_doc_counts[label] += 1
            for token in tokens:
                self.word_counts[label][token] += 1
                self.class_totals[label] += 1
                self.vocab.add(token)

        total_docs = len(y)
        self.classes = list(class_doc_counts.keys())

        # Log prior: log P(class)
        for cls in self.classes:
            self.log_prior[cls] = math.log(class_doc_counts[cls] / total_docs)

        # Log likelihood with Laplace smoothing: log P(word | class)
        V = len(self.vocab)
        for cls in self.classes:
            total = self.class_totals[cls]
            for word in self.vocab:
                count = self.word_counts[cls].get(word, 0)
                self.log_likelihood[cls][word] = math.log((count + self.alpha) / (total + self.alpha * V))

        return self

    def predict_proba(self, text: str) -> Dict[str, float]:
        tokens = tokenize(text)
        scores = {}
        for cls in self.classes:
            score = self.log_prior[cls]
            for token in tokens:
                if token in self.vocab:
                    score += self.log_likelihood[cls][token]
                # Unknown words: skip (or use UNK smoothing)
            scores[cls] = score

        # Convert log-scores to probabilities via softmax
        max_score = max(scores.values())
        exp_scores = {cls: math.exp(s - max_score) for cls, s in scores.items()}
        total = sum(exp_scores.values())
        return {cls: v / total for cls, v in exp_scores.items()}

    def predict(self, text: str) -> Tuple[str, float]:
        proba = self.predict_proba(text)
        label = max(proba, key=proba.get)
        return label, proba[label]

    def top_features(self, cls: str, n: int = 10) -> List[Tuple[str, float]]:
        """Return the n words most associated with this class."""
        other = [c for c in self.classes if c != cls]
        if not other:
            return []
        other_cls = other[0]
        ratios = []
        for word in self.vocab:
            ll_cls   = self.log_likelihood[cls].get(word, -20)
            ll_other = self.log_likelihood[other_cls].get(word, -20)
            ratios.append((word, ll_cls - ll_other))
        ratios.sort(key=lambda x: x[1], reverse=True)
        return ratios[:n]

    def save(self, path: str):
        data = {
            "alpha": self.alpha,
            "log_prior": self.log_prior,
            "log_likelihood": {k: dict(v) for k, v in self.log_likelihood.items()},
            "vocab": list(self.vocab),
            "classes": self.classes,
            "word_counts": {k: dict(v) for k, v in self.word_counts.items()},
            "class_totals": dict(self.class_totals),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.alpha = data["alpha"]
        self.log_prior = data["log_prior"]
        self.log_likelihood = defaultdict(dict, {k: v for k, v in data["log_likelihood"].items()})
        self.vocab = set(data["vocab"])
        self.classes = data["classes"]
        self.word_counts = defaultdict(lambda: defaultdict(int), {k: defaultdict(int, v) for k, v in data["word_counts"].items()})
        self.class_totals = defaultdict(int, data["class_totals"])


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(clf: NaiveBayesClassifier, X_test: List[str], y_test: List[str]) -> dict:
    tp = fp = tn = fn = 0
    for text, true_label in zip(X_test, y_test):
        pred, _ = clf.predict(text)
        if pred == "spam" and true_label == "spam": tp += 1
        elif pred == "spam" and true_label == "ham": fp += 1
        elif pred == "ham" and true_label == "ham": tn += 1
        else: fn += 1

    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ── Built-in dataset ──────────────────────────────────────────────────────────

DATASET = [
    # SPAM
    ("Congratulations! You've won a $1000 gift card. Click here NOW to claim your prize!!!", "spam"),
    ("URGENT: Your account has been compromised. Verify immediately or lose access forever.", "spam"),
    ("Free money! Make $5000 a week from home. No experience needed. Act fast!", "spam"),
    ("You have been selected for a special offer. Buy now and get 80% OFF. Limited time!!!", "spam"),
    ("Dear winner, You have won our lottery. Send your details to claim $500,000 prize.", "spam"),
    ("Lose 30 pounds in 30 days guaranteed! Click here for the secret doctors don't want you to know.", "spam"),
    ("Your PayPal account is suspended. Verify your identity now at URL to avoid permanent ban.", "spam"),
    ("HOT SINGLES IN YOUR AREA want to meet you tonight. Click here free no credit card needed.", "spam"),
    ("Investment opportunity! 300% returns guaranteed. Send $100 now and receive $400 in 24 hours.", "spam"),
    ("FINAL WARNING: Your computer has a virus. Call our support number immediately 1-800-SCAM.", "spam"),
    ("Exclusive offer for you only! Discount meds online no prescription needed cheap prices.", "spam"),
    ("You owe back taxes. The IRS will arrest you today. Call immediately to avoid prosecution.", "spam"),
    ("Make money online fast! Proven system earns $10,000 monthly with zero effort guaranteed.", "spam"),
    ("Claim your free iPhone now! You are our lucky visitor. Fill in the form to receive it.", "spam"),
    ("Business proposal: I need a trustworthy partner to transfer $10 million from Nigeria.", "spam"),
    # HAM
    ("Hey, are you free this weekend? Wanted to catch up over coffee.", "ham"),
    ("Please review the attached quarterly report before tomorrow's meeting.", "ham"),
    ("Your order #4821 has been shipped. Expected delivery is Friday.", "ham"),
    ("Can you send me the Python script we discussed? I need it for the presentation.", "ham"),
    ("Reminder: Team standup at 10 AM. Please bring your sprint updates.", "ham"),
    ("I found a bug in the auth module. Pull request is open for review.", "ham"),
    ("Thanks for your help yesterday. Really appreciated the explanation.", "ham"),
    ("The lab assignment is due Thursday at midnight. Don't forget!", "ham"),
    ("Your appointment is confirmed for Monday at 2:30 PM.", "ham"),
    ("Just wanted to let you know the server migration is complete. Everything looks good.", "ham"),
    ("Can we reschedule the interview to next week? Something came up.", "ham"),
    ("Here is the reading list for next semester's distributed systems course.", "ham"),
    ("The network was down this morning but it's back now. Ticket closed.", "ham"),
    ("Forwarding the meeting notes from yesterday. Let me know if anything's missing.", "ham"),
    ("Good work on the project! The client was really happy with the demo.", "ham"),
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_result(text: str, clf: NaiveBayesClassifier):
    label, confidence = clf.predict(text)
    proba = clf.predict_proba(text)
    icon = "🔴 SPAM" if label == "spam" else "🟢 HAM"
    print(f"\n  Input    : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Verdict  : {icon}  ({confidence*100:.1f}% confidence)")
    print(f"  Spam prob: {proba.get('spam', 0)*100:.1f}%   Ham prob: {proba.get('ham', 0)*100:.1f}%")


def main():
    # Split dataset 80/20
    random_seed = 42
    data = DATASET[:]

    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data  = data[split:]

    X_train = [d[0] for d in train_data]
    y_train = [d[1] for d in train_data]
    X_test  = [d[0] for d in test_data]
    y_test  = [d[1] for d in test_data]

    clf = NaiveBayesClassifier()
    clf.fit(X_train, y_train)

    if "--classify" in sys.argv:
        idx = sys.argv.index("--classify")
        text = " ".join(sys.argv[idx + 1:])
        print_result(text, clf)
        return

    if "--interactive" in sys.argv:
        print("\n  Spam Classifier — Interactive Mode")
        print("  Type an email and press Enter. 'quit' to exit.\n")
        while True:
            try:
                text = input("  > ").strip()
                if text.lower() in ("quit", "exit", "q"):
                    break
                if text:
                    print_result(text, clf)
            except (KeyboardInterrupt, EOFError):
                break
        return

    # Default: demo
    metrics = evaluate(clf, X_test, y_test)

    print("\n" + "=" * 55)
    print("  SPAM CLASSIFIER — NAIVE BAYES DEMO")
    print("=" * 55)
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print(f"  Vocabulary size  : {len(clf.vocab)}")
    print()
    print(f"  Accuracy  : {metrics['accuracy']*100:.1f}%")
    print(f"  Precision : {metrics['precision']*100:.1f}%  (of predicted spam, how many were spam)")
    print(f"  Recall    : {metrics['recall']*100:.1f}%  (of actual spam, how many did we catch)")
    print(f"  F1 Score  : {metrics['f1']*100:.1f}%")
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              Spam    Ham")
    print(f"  Actual Spam   {metrics['tp']:>3}     {metrics['fn']:>3}")
    print(f"  Actual Ham    {metrics['fp']:>3}     {metrics['tn']:>3}")

    print(f"\n  Top spam indicators:")
    for word, score in clf.top_features("spam", 8):
        print(f"    {word:<20} log-ratio: {score:.2f}")

    print(f"\n  Sample predictions:")
    samples = [
        "Congratulations you won a free prize click here now!!!",
        "Here are the notes from our meeting yesterday.",
        "URGENT verify your bank account or it will be closed.",
        "The code review is scheduled for Thursday morning.",
    ]
    for s in samples:
        print_result(s, clf)

    print(f"\n  Usage: python spam_classifier.py --classify 'your email text here'")
    print(f"         python spam_classifier.py --interactive\n")


if __name__ == "__main__":
    main()

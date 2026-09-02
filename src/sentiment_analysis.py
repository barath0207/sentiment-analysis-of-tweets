"""
Sentiment Analysis of Tweets
Third-Year B.E. CSE Academic Project

Run:
    python src/sentiment_analysis.py
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import nltk


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tweets.csv"
RESULTS_PATH = ROOT / "results"


def prepare_nltk():
    """Download the small NLTK resources needed by the project."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


prepare_nltk()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Clean tweet text while retaining useful words."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in text.split()
        if word not in STOP_WORDS and len(word) > 1
    ]
    return " ".join(tokens)


def train_and_evaluate():
    """Load data, train the model, evaluate it, and save the confusion matrix."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required = {"tweet", "sentiment"}
    if not required.issubset(data.columns):
        raise ValueError("CSV must contain 'tweet' and 'sentiment' columns.")

    data = data.dropna(subset=["tweet", "sentiment"]).copy()
    data["clean_tweet"] = data["tweet"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_tweet"],
        data["sentiment"],
        test_size=0.20,
        random_state=42,
        stratify=data["sentiment"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print("\n=== Sentiment Analysis Results ===")
    print(f"Dataset size : {len(data)}")
    print(f"Training set : {len(X_train)}")
    print(f"Test set     : {len(X_test)}")
    print(f"Accuracy     : {accuracy:.2%}\n")
    print(classification_report(y_test, predictions))

    labels = sorted(data["sentiment"].unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)

    RESULTS_PATH.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Tweet Sentiment - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "confusion_matrix.png", dpi=160)
    plt.close()

    return model


def interactive_prediction(model):
    """Predict sentiment for tweets entered by the user."""
    print("\nEnter a tweet to predict its sentiment.")
    print("Type 'exit' to finish.")

    while True:
        try:
            tweet = input("\nTweet: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if tweet.lower() == "exit":
            break
        if not tweet:
            print("Please enter some text.")
            continue

        cleaned = clean_text(tweet)
        prediction = model.predict([cleaned])[0]
        print(f"Predicted sentiment: {prediction.capitalize()}")


if __name__ == "__main__":
    try:
        trained_model = train_and_evaluate()
        interactive_prediction(trained_model)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

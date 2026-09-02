# Sentiment Analysis of Tweets

A machine learning project that classifies tweets as **Positive, Negative, or Neutral**.

## Overview

Social media contains a large amount of short, informal text. Sentiment analysis uses Natural Language Processing (NLP) and machine learning to identify the emotional polarity expressed in that text.

This project demonstrates a complete beginner-to-intermediate NLP pipeline:

**Tweet → Text Cleaning → TF-IDF Features → Logistic Regression → Sentiment Prediction**

## Features

- Tweet text preprocessing
- Removal of URLs, mentions, hashtags, punctuation, and stopwords
- TF-IDF feature extraction
- Logistic Regression classification
- Train/test split
- Accuracy and classification report
- Confusion matrix
- Prediction for custom tweets
- Reproducible command-line workflow

## Technologies Used

- Python 3.10+
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

## Project Structure

```text
sentiment-analysis-of-tweets/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── tweets.csv
├── notebooks/
│   └── sentiment_analysis.ipynb
├── src/
│   └── sentiment_analysis.py
├── results/
│   └── README.md
└── screenshots/
    └── README.md
```

## Dataset

The repository includes a small self-contained demonstration dataset in `data/tweets.csv` so the project can be executed without downloading an external dataset.

For a larger academic experiment, the same pipeline can be used with a real Twitter/X dataset containing `tweet` and `sentiment` columns.

> **Note:** The included dataset is a demonstration dataset created for this repository. It should not be presented as real-world Twitter data.

## Installation

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd sentiment-analysis-of-tweets

python -m venv .venv
```

Activate the virtual environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python src/sentiment_analysis.py
```

The program trains the model, displays evaluation metrics, saves a confusion matrix to `results/confusion_matrix.png`, and allows you to enter your own tweets for prediction.

## Example Predictions

```text
I love this new phone!       → Positive
This service is terrible.    → Negative
The meeting starts at 5 PM.  → Neutral
```

## Methodology

### 1. Text Preprocessing
Tweets are normalized by:
- converting text to lowercase
- removing URLs
- removing mentions
- removing hashtag symbols while retaining the word
- removing punctuation and extra spaces
- removing common stopwords

### 2. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts cleaned text into numerical features. Words that are useful for distinguishing tweets receive higher importance.

### 3. Classification

A Logistic Regression classifier is trained using the TF-IDF vectors.

### 4. Evaluation

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Future Enhancements

- Use a much larger real-world Twitter/X dataset
- Add multilingual sentiment analysis
- Compare Logistic Regression with Naive Bayes, SVM, and transformer models
- Add a web interface using Flask or Streamlit
- Perform real-time sentiment analysis from collected social-media data
- Add emotion categories such as joy, anger, sadness, and fear

## Academic Context

**Project Type:** Third-Year Academic Project  
**Domain:** Machine Learning / Natural Language Processing  
**Application:** Social Media Sentiment Analysis

## Author

Barath Kumar S, B.E CSE, CARE College of Engineering and @barath0207

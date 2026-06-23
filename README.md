# Spam Email Classifier – Naive Bayes From Scratch

A machine learning project that classifies emails as spam or legitimate (ham) using a Multinomial Naive Bayes algorithm implemented entirely from scratch. The project demonstrates the mathematics behind text classification without relying on external machine learning libraries such as scikit-learn.

## Features

* Custom text preprocessing and tokenization
* Stopword removal
* Bag-of-Words feature extraction
* Laplace smoothing implementation
* Probability-based classification
* Confidence score calculation
* Interactive classification mode
* Model save/load functionality
* Performance evaluation metrics

## How It Works

The classifier learns word patterns from labeled email samples and calculates:

* Prior probabilities for each class
* Word likelihoods for spam and ham
* Posterior probabilities using Bayes' Theorem

Predictions are made by selecting the class with the highest probability score.

## Machine Learning Concepts

This project demonstrates:

* Supervised learning
* Multinomial Naive Bayes
* Text preprocessing
* Feature extraction
* Probability theory
* Classification metrics
* Natural Language Processing fundamentals

## Installation

```bash
git clone https://github.com/yourusername/spam-email-classifier.git
cd spam-email-classifier
```

## Usage

Run the demo:

```bash
python spam_classifier.py
```

Classify a custom email:

```bash
python spam_classifier.py --classify "Win a free iPhone now!"
```

Interactive mode:

```bash
python spam_classifier.py --interactive
```

## Evaluation Metrics

The classifier reports:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

## Technologies Used

* Python 3
* Naive Bayes
* Natural Language Processing
* JSON Serialization
* Regular Expressions

## Learning Objectives

This project was built to understand the inner workings of spam detection systems and gain hands-on experience with machine learning algorithms, text processing, and probabilistic classification without using pre-built ML frameworks.

## License

MIT License

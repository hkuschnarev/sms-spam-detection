# SMS Spam Detection

A Naive Bayes classifier for detecting spam SMS messages, with two text
representations — **Bag-of-Words** and **TF-IDF** — both implemented
from scratch in NumPy.

---

## Results

Test set: 1,115 messages (954 ham / 161 spam) · 80/20 split, seed 42

| Model                  | Accuracy | Spam Precision | Spam Recall | Spam F1 |
| ---------------------- | -------- | -------------- | ----------- | ------- |
| **Bag-of-Words + NB**  | **0.98** | 0.94           | 0.96        | **0.95** |
| TF-IDF + NB            | 0.97     | 0.90           | 0.93        | 0.91    |

Bag-of-Words slightly outperformed TF-IDF on this dataset. Both models
classify ham messages near-perfectly (precision 0.99); the harder class
is spam due to class imbalance (~85 % ham / ~15 % spam).

---

## Approach

### Preprocessing
- Lowercasing
- Punctuation removed (replaced with whitespace)
- Digits removed
- Whitespace tokenization
- English stopword removal (NLTK)
- Porter stemming

### Feature Extraction (from scratch)
- **Bag-of-Words** — word counts per document, vocabulary built from
  training data only
- **TF-IDF** — `tf × idf` with smoothed IDF:
  `idf = log((1 + N) / (1 + df)) + 1`

### Classifier
`sklearn.naive_bayes.MultinomialNB` with default Laplace smoothing
(`alpha = 1.0`).

---

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
(UCI Machine Learning Repository) — 5,574 labeled SMS messages.

The dataset is **not** included in this repo. Download it manually and
place `SMSSpamCollection` (the tab-separated file) somewhere accessible,
then update the path inside the notebook:

```python
spam_detector(r"path/to/SMSSpamCollection")
```

---

## Setup

```bash
# clone the repo
git clone https://github.com/hkuschnarev/sms-spam-detection.git
cd sms-spam-detection

# install dependencies
pip install -r requirements.txt

# launch the notebook
jupyter notebook sms_spam_detection.ipynb
```

On the first run, NLTK will download the English stopwords corpus.

---

## Findings

**Strengths**
- Simple, fast pipeline that reaches 98 % accuracy
- Both feature extractors implemented from scratch — no `CountVectorizer`
  / `TfidfVectorizer`

**Limitations**
- Stopword removal can strip SMS-specific spam signals
- Whitespace tokenization ignores emojis, URLs, special characters
- BoW / TF-IDF discard word order and context
- New words at test time fall out of vocabulary

**Possible improvements**
- Use n-grams (bi-/tri-grams) to capture context
- Add hand-crafted features (uppercase ratio, special-char count, URL flag,
  message length)
- Retrain without stemming and compare
- Use a logistic regression or small transformer for comparison

---

## Tech Stack

Python · NumPy · scikit-learn · NLTK · Jupyter Notebook

---

## Project Context

Originally developed as part of the *Natural Language Processing* course at
Technische Hochschule Ingolstadt (B.Sc. Artificial Intelligence).

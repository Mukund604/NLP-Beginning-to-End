"""Measures how much each text representation from the notebooks matters on IMDB sentiment.

Exact duplicate reviews are dropped before splitting so none can sit in both train and test.
Usage, from the repo root: python "4. Comparing Representations/compare.py"
"""
import json
import pathlib
import re
import time

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

HERE = pathlib.Path(__file__).parent
frame = pd.read_csv(HERE.parent / "Dataset/IMDB Dataset.csv")
before = len(frame)
frame = frame.drop_duplicates("review")
text = frame.review.map(lambda t: re.sub(r"<[^>]+>", " ", t))
y = (frame.sentiment == "positive").astype(int)
a, b, y_a, y_b = train_test_split(text, y, test_size=0.2, stratify=y, random_state=0)

lr = lambda: LogisticRegression(max_iter=2000, C=4)
MODELS = {
    "Bag of words + naive Bayes": make_pipeline(CountVectorizer(min_df=2), MultinomialNB()),
    "Bag of words + logistic regression": make_pipeline(CountVectorizer(min_df=2), LogisticRegression(max_iter=2000)),
    "TF-IDF + logistic regression": make_pipeline(TfidfVectorizer(min_df=2, sublinear_tf=True), lr()),
    "TF-IDF with bigrams + logistic regression": make_pipeline(
        TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True), lr()),
    "Latent semantic analysis (300 dimensions) + logistic regression": make_pipeline(
        TfidfVectorizer(min_df=2, sublinear_tf=True), TruncatedSVD(300, random_state=0), LogisticRegression(max_iter=2000)),
}
rows = []
for name, model in MODELS.items():
    start = time.time()
    pred = model.fit(a, y_a).predict(b)
    rows.append(dict(model=name, accuracy=accuracy_score(y_b, pred), f1=f1_score(y_b, pred), seconds=time.time() - start))
    print(rows[-1], flush=True)
(HERE / "results.json").write_text(json.dumps(dict(reviews_before=before, reviews_after=len(frame), test_size=len(b), rows=rows), indent=2))

# NLP, Beginning to End

Notebooks that work through the core NLP pipeline step by step, from raw text to word embeddings, with notes alongside the code. Examples use the IMDB reviews CSV in `Dataset/`.

| Notebook | Covers | Libraries |
| --- | --- | --- |
| [1. Text Preprocessing](1.Text%20Preprocessing/1.ipynb) | Lower casing, removing HTML tags, URLs and punctuation, spelling correction, stop words, emoji handling, tokenization, stemming and lemmatization | pandas, NLTK, TextBlob, emoji |
| [2. Text Representation](2.%20Text%20Representation/2.ipynb) | Bag of words, n-grams, TF-IDF | scikit-learn |
| [3. Word2Vec](3.%20Word2Vec/3.ipynb) | Word embeddings with Gensim | Gensim |

## How much each representation matters

[`compare.py`](4.%20Comparing%20Representations/compare.py) runs the ideas from the notebooks on the same task: IMDB sentiment, using 49,582 reviews after dropping 418 exact duplicates so none sit in both train and test, with a 20% test split of 9,917.

| Representation and model | Test accuracy | F1 |
| :--- | ---: | ---: |
| Bag of words + naive Bayes | 84.9% | 0.845 |
| Latent semantic analysis (300 dimensions) + logistic regression | 88.3% | 0.884 |
| Bag of words + logistic regression | 89.0% | 0.891 |
| TF-IDF + logistic regression | 90.2% | 0.903 |
| **TF-IDF with bigrams + logistic regression** | **91.6%** | **0.917** |

Adding bigrams helps most among the TF-IDF models, likely because they capture phrases like "not good" that single words miss. Compressing to 300 dimensions costs about 2 points against plain TF-IDF.

```
pip install pandas scikit-learn
python "4. Comparing Representations/compare.py"
```

## Run it

```
pip install pandas nltk textblob emoji scikit-learn gensim jupyter
jupyter notebook
```

Then open the notebooks in order. Notebook 3 loads pretrained vectors through Gensim's `KeyedVectors`, so you'll need a copy of those vectors locally.

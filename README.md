# NLP, Beginning to End

Notebooks that work through the core NLP pipeline step by step, from raw text to word embeddings, with notes alongside the code. Examples use the IMDB reviews CSV in `Dataset/`.

| Notebook | Covers | Libraries |
| --- | --- | --- |
| [1. Text Preprocessing](1.Text%20Preprocessing/1.ipynb) | Lower casing, removing HTML tags, URLs and punctuation, spelling correction, stop words, emoji handling, tokenization, stemming and lemmatization | pandas, NLTK, TextBlob, emoji |
| [2. Text Representation](2.%20Text%20Represenataion/2.ipynb) | Bag of words, n-grams, TF-IDF | scikit-learn |
| [3. Word2Vec](3.%20Word2Vec/3.ipynb) | Word embeddings with Gensim | Gensim |

## Run it

```
pip install pandas nltk textblob emoji scikit-learn gensim jupyter
jupyter notebook
```

Then open the notebooks in order. Notebook 3 loads pretrained vectors through Gensim's `KeyedVectors`, so you'll need a copy of those vectors locally.

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary


# --------------------------------
# 1. NLTK resources
# --------------------------------

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# --------------------------------
# 2. Load dataset
# --------------------------------

file_path = "data/news-article-categories.csv"

df = pd.read_csv(file_path)

df["body"] = df["body"].fillna("")

df["text"] = df["title"] + " " + df["body"]


# --------------------------------
# 3. Preprocessing
# --------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return words


processed_documents = [
    preprocess_text(text)
    for text in df["text"]
]


# --------------------------------
# 4. Create dictionary
# --------------------------------

dictionary = Dictionary(processed_documents)


print("\n===== ORIGINAL DICTIONARY =====")
print("Vocabulary size:", len(dictionary))


# --------------------------------
# 5. Remove extremely rare/common words
# --------------------------------

dictionary.filter_extremes(
    no_below=5,
    no_above=0.5
)


print("\n===== FILTERED DICTIONARY =====")
print("Vocabulary size:", len(dictionary))


# --------------------------------
# 6. Create Bag-of-Words corpus
# --------------------------------

bow_corpus = [
    dictionary.doc2bow(document)
    for document in processed_documents
]


# --------------------------------
# 7. Display examples
# --------------------------------

print("\n===== DICTIONARY SAMPLE =====")

for word_id, word in list(dictionary.id2token.items())[:20]:
    print(word_id, ":", word)


print("\n===== BAG-OF-WORDS SAMPLE =====")

for i in range(3):
    print(f"\nDocument {i + 1}:")
    print(bow_corpus[i][:20])


# --------------------------------
# 8. Corpus statistics
# --------------------------------

print("\n===== CORPUS INFORMATION =====")

print("Number of documents:", len(bow_corpus))

print("Vocabulary size:", len(dictionary))

print(
    "Average unique words per document:",
    round(
        sum(len(doc) for doc in bow_corpus) / len(bow_corpus),
        2
    )
)
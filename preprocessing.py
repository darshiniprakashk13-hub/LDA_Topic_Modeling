import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# --------------------------------
# 1. Load dataset
# --------------------------------

file_path = "data/news-article-categories.csv"

df = pd.read_csv(file_path)

print("Original number of documents:", len(df))


# --------------------------------
# 2. Handle missing body values
# --------------------------------

df["body"] = df["body"].fillna("")


# --------------------------------
# 3. Combine title and body
# --------------------------------

df["text"] = df["title"] + " " + df["body"]


# --------------------------------
# 4. Initialize preprocessing tools
# --------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# --------------------------------
# 5. Preprocessing function
# --------------------------------

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Keep only alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenization
    words = text.split()

    # Remove stop words and short words
    words = [
        word
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return words


# --------------------------------
# 6. Apply preprocessing
# --------------------------------

processed_documents = df["text"].apply(preprocess_text)


# --------------------------------
# 7. Display examples
# --------------------------------

print("\n===== PREPROCESSING EXAMPLES =====")

for i in range(3):
    print("\nOriginal:")
    print(df["text"].iloc[i][:500])

    print("\nProcessed:")
    print(processed_documents.iloc[i][:50])


# --------------------------------
# 8. Basic statistics
# --------------------------------

total_words = sum(len(doc) for doc in processed_documents)

print("\n===== PREPROCESSING SUMMARY =====")
print("Number of documents:", len(processed_documents))
print("Total words after preprocessing:", total_words)

average_words = total_words / len(processed_documents)

print("Average words per document:", round(average_words, 2))
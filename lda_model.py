import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.corpora import Dictionary
from gensim.models import LdaModel


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # ========================================================
    # Download NLTK resources
    # ========================================================

    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


    # ========================================================
    # Load dataset
    # ========================================================

    file_path = "data/news-article-categories.csv"

    df = pd.read_csv(file_path)

    # Handle missing body values
    df["body"] = df["body"].fillna("")

    # Combine title and body
    df["text"] = (
        df["title"] + " " + df["body"]
    )


    # ========================================================
    # Text preprocessing
    # ========================================================

    stop_words = set(
        stopwords.words("english")
    )

    lemmatizer = WordNetLemmatizer()


    def preprocess_text(text):

        # Convert to lowercase
        text = text.lower()

        # Remove numbers, punctuation and special characters
        text = re.sub(
            r"[^a-z\s]",
            " ",
            text
        )

        # Split into words
        words = text.split()

        # Remove stopwords and short words
        words = [
            word
            for word in words
            if word not in stop_words
            and len(word) > 2
        ]

        # Lemmatization
        words = [
            lemmatizer.lemmatize(word)
            for word in words
        ]

        return words


    # ========================================================
    # Process all documents
    # ========================================================

    processed_documents = [
        preprocess_text(text)
        for text in df["text"]
    ]


    print("\n===================================")
    print("TEXT PREPROCESSING COMPLETED")
    print("===================================")

    print(
        "Number of documents:",
        len(processed_documents)
    )

    total_words = sum(
        len(document)
        for document in processed_documents
    )

    print(
        "Total processed words:",
        total_words
    )


    # ========================================================
    # Create dictionary
    # ========================================================

    dictionary = Dictionary(
        processed_documents
    )


    print("\n===================================")
    print("DICTIONARY")
    print("===================================")

    print(
        "Original vocabulary size:",
        len(dictionary)
    )


    # Remove very rare and very common words

    dictionary.filter_extremes(
        no_below=5,
        no_above=0.5
    )


    print(
        "Filtered vocabulary size:",
        len(dictionary)
    )


    # ========================================================
    # Create Bag-of-Words corpus
    # ========================================================

    bow_corpus = [
        dictionary.doc2bow(document)
        for document in processed_documents
    ]


    print("\n===================================")
    print("BAG-OF-WORDS CORPUS")
    print("===================================")

    print(
        "Number of documents:",
        len(bow_corpus)
    )

    print(
        "Vocabulary size:",
        len(dictionary)
    )


    # ========================================================
    # TRAIN FINAL LDA MODEL
    # ========================================================

    print("\n===================================")
    print("TRAINING FINAL LDA MODEL")
    print("===================================")

    print("Number of topics: 7")
    print("Training model...")


    lda_model = LdaModel(

        corpus=bow_corpus,

        id2word=dictionary,

        # FINAL MODEL
        num_topics=7,

        # Reproducible results
        random_state=42,

        # Number of training passes
        passes=5,

        # Iterations per document
        iterations=100,

        # Number of documents processed together
        chunksize=200,

        # Automatically learn document-topic prior
        alpha="auto",

        # Automatically learn topic-word prior
        eta="auto"
    )


    print(
        "LDA model training completed."
    )


    # ========================================================
    # DISPLAY DISCOVERED TOPICS
    # ========================================================

    print("\n===================================")
    print("DISCOVERED TOPICS")
    print("===================================")


    for topic_id in range(7):

        print(
            f"\nTopic {topic_id + 1}:"
        )

        topic_words = lda_model.show_topic(
            topic_id,
            topn=10
        )

        for word, probability in topic_words:

            print(
                f"{word}: "
                f"{probability:.4f}"
            )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    lda_model.save(
        "lda_model.gensim"
    )

    dictionary.save(
        "dictionary.gensim"
    )


    print("\n===================================")
    print("MODEL SAVED")
    print("===================================")

    print(
        "LDA model saved as: "
        "lda_model.gensim"
    )

    print(
        "Dictionary saved as: "
        "dictionary.gensim"
    )


    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print("\n===================================")
    print("FINAL MODEL INFORMATION")
    print("===================================")

    print(
        "Documents:",
        len(bow_corpus)
    )

    print(
        "Topics:",
        lda_model.num_topics
    )

    print(
        "Vocabulary:",
        len(dictionary)
    )

    print(
        "Random state: 42"
    )

    print(
        "Passes: 5"
    )

    print(
        "Iterations: 100"
    )

    print(
        "Model training completed successfully."
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
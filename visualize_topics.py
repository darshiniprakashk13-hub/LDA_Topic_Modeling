import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel


def main():

    # =====================================================
    # NLTK resources
    # =====================================================

    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    stop_words = set(
        stopwords.words("english")
    )

    lemmatizer = WordNetLemmatizer()


    # =====================================================
    # Load dataset
    # =====================================================

    df = pd.read_csv(
        "data/news-article-categories.csv"
    )

    df["body"] = df["body"].fillna("")

    df["text"] = (
        df["title"] + " " + df["body"]
    )


    # =====================================================
    # Preprocessing
    # =====================================================

    def preprocess_text(text):

        text = text.lower()

        text = re.sub(
            r"[^a-z\s]",
            " ",
            text
        )

        words = text.split()

        words = [
            word
            for word in words
            if word not in stop_words
            and len(word) > 2
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


    # =====================================================
    # Load dictionary and final 7-topic model
    # =====================================================

    dictionary = Dictionary.load(
        "dictionary.gensim"
    )

    lda_model = LdaModel.load(
        "lda_model.gensim"
    )


    print("\n===================================")
    print("GENERATING TOPIC VISUALIZATIONS")
    print("===================================")

    print(
        "Number of topics:",
        lda_model.num_topics
    )


    # =====================================================
    # Generate one image for every topic
    # =====================================================

    for topic_id in range(
        lda_model.num_topics
    ):

        topic_words = lda_model.show_topic(
            topic_id,
            topn=10
        )

        words = [
            word
            for word, probability
            in topic_words
        ]

        probabilities = [
            probability
            for word, probability
            in topic_words
        ]


        # -----------------------------------------------
        # Create chart
        # -----------------------------------------------

        plt.figure(
            figsize=(10, 6)
        )

        plt.barh(
            words[::-1],
            probabilities[::-1]
        )

        plt.xlabel(
            "Probability"
        )

        plt.ylabel(
            "Word"
        )

        plt.title(
            f"Topic {topic_id + 1} - "
            "Top 10 Words"
        )

        plt.tight_layout()


        # -----------------------------------------------
        # Save image
        # -----------------------------------------------

        filename = (
            f"topic_{topic_id + 1}.png"
        )

        plt.savefig(
            filename,
            dpi=150
        )

        plt.close()


        print(
            f"Created: {filename}"
        )


    print("\n===================================")
    print("VISUALIZATION COMPLETED")
    print("===================================")


if __name__ == "__main__":
    main()
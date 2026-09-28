import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel


def main():

    # =====================================================
    # Download NLTK resources
    # =====================================================

    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")


    # =====================================================
    # Load dataset
    # =====================================================

    file_path = "data/news-article-categories.csv"

    df = pd.read_csv(file_path)

    df["body"] = df["body"].fillna("")

    df["text"] = (
        df["title"] + " " + df["body"]
    )


    # =====================================================
    # Preprocessing
    # =====================================================

    stop_words = set(
        stopwords.words("english")
    )

    lemmatizer = WordNetLemmatizer()


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
    # Create dictionary and corpus
    # =====================================================

    dictionary = Dictionary(
        processed_documents
    )

    dictionary.filter_extremes(
        no_below=5,
        no_above=0.5
    )


    bow_corpus = [
        dictionary.doc2bow(document)
        for document in processed_documents
    ]


    # =====================================================
    # Load trained LDA model
    # =====================================================

    lda_model = LdaModel.load(
        "lda_model.gensim"
    )


    print("\n===================================")
    print("LDA MODEL EVALUATION")
    print("===================================")


    print(
        "Number of topics:",
        lda_model.num_topics
    )

    print(
        "Vocabulary size:",
        len(dictionary)
    )


    # =====================================================
    # Perplexity
    # =====================================================

    perplexity = lda_model.log_perplexity(
        bow_corpus
    )

    print(
        "\nLog Perplexity:",
        round(perplexity, 4)
    )


    # =====================================================
    # Coherence Score
    # =====================================================

    coherence_model = CoherenceModel(
        model=lda_model,
        texts=processed_documents,
        dictionary=dictionary,
        coherence="c_v",
        processes=1
    )

    coherence_score = (
        coherence_model.get_coherence()
    )


    print(
        "Coherence Score:",
        round(coherence_score, 4)
    )


    # =====================================================
    # Topic Words
    # =====================================================

    print("\n===================================")
    print("TOPIC WORDS")
    print("===================================")


    for topic_id in range(
        lda_model.num_topics
    ):

        words = lda_model.show_topic(
            topic_id,
            topn=10
        )

        print(
            f"\nTopic {topic_id + 1}:"
        )

        for word, probability in words:

            print(
                f"{word}: "
                f"{probability:.4f}"
            )


    print("\n===================================")
    print("EVALUATION COMPLETED")
    print("===================================")


if __name__ == "__main__":
    main()
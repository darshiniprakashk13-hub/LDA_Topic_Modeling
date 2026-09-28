import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel


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

    file_path = "data/news-article-categories.csv"

    df = pd.read_csv(file_path)

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
    # Dictionary
    # =====================================================

    dictionary = Dictionary(
        processed_documents
    )

    dictionary.filter_extremes(
        no_below=5,
        no_above=0.5
    )


    # =====================================================
    # Bag of Words
    # =====================================================

    bow_corpus = [
        dictionary.doc2bow(document)
        for document in processed_documents
    ]


    # =====================================================
    # Compare different topic numbers
    # =====================================================

    topic_numbers = [3, 5, 7, 10]

    results = []


    print("\n===================================")
    print("LDA TOPIC NUMBER COMPARISON")
    print("===================================")


    for num_topics in topic_numbers:

        print(
            f"\nTraining LDA with "
            f"{num_topics} topics..."
        )


        lda_model = LdaModel(

            corpus=bow_corpus,

            id2word=dictionary,

            num_topics=num_topics,

            random_state=42,

            passes=5,

            iterations=100,

            chunksize=200,

            alpha="auto",

            eta="auto"
        )


        # -------------------------------------------------
        # Coherence
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Perplexity
        # -------------------------------------------------

        log_perplexity = (
            lda_model.log_perplexity(
                bow_corpus
            )
        )


        results.append({

            "Number of Topics":
                num_topics,

            "Coherence Score":
                coherence_score,

            "Log Perplexity":
                log_perplexity
        })


        print(
            f"Topics: {num_topics}"
        )

        print(
            f"Coherence: "
            f"{coherence_score:.4f}"
        )

        print(
            f"Log Perplexity: "
            f"{log_perplexity:.4f}"
        )


    # =====================================================
    # Results table
    # =====================================================

    results_df = pd.DataFrame(
        results
    )


    print("\n===================================")
    print("FINAL COMPARISON")
    print("===================================")

    print(
        results_df.to_string(
            index=False
        )
    )


    # =====================================================
    # Save results
    # =====================================================

    results_df.to_csv(
        "topic_comparison.csv",
        index=False
    )


    print(
        "\nResults saved to:"
    )

    print(
        "topic_comparison.csv"
    )


if __name__ == "__main__":

    main()
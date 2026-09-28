import pandas as pd
from gensim.models import LdaModel


def main():

    # =====================================================
    # Load final LDA model
    # =====================================================

    lda_model = LdaModel.load(
        "lda_model.gensim"
    )


    print("\n===================================")
    print("FINAL LDA TOPIC SUMMARY")
    print("===================================")


    # =====================================================
    # Create topic summary
    # =====================================================

    rows = []


    for topic_id in range(
        lda_model.num_topics
    ):

        topic_words = lda_model.show_topic(
            topic_id,
            topn=10
        )


        # Create readable word list
        words = [
            word
            for word, probability
            in topic_words
        ]


        # Create probability list
        probabilities = [
            round(probability, 4)
            for word, probability
            in topic_words
        ]


        rows.append({

            "Topic":
                f"Topic {topic_id + 1}",

            "Top Words":
                ", ".join(words),

            "Probabilities":
                ", ".join(
                    map(
                        str,
                        probabilities
                    )
                )
        })


    # =====================================================
    # Create DataFrame
    # =====================================================

    summary_df = pd.DataFrame(
        rows
    )


    # =====================================================
    # Display summary
    # =====================================================

    print(
        summary_df.to_string(
            index=False
        )
    )


    # =====================================================
    # Save CSV
    # =====================================================

    summary_df.to_csv(
        "topic_summary.csv",
        index=False
    )


    print("\n===================================")
    print("SUMMARY SAVED")
    print("===================================")

    print(
        "File: topic_summary.csv"
    )


if __name__ == "__main__":
    main()

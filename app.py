import streamlit as st
import pandas as pd
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel

import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NewsScope | LDA Topic Modeling",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at top left,
            rgba(99, 102, 241, 0.12),
            transparent 35%
        ),
        radial-gradient(
            circle at top right,
            rgba(14, 165, 233, 0.10),
            transparent 30%
        ),
        #f7f9fc;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* SIDEBAR */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #111827 0%,
            #172554 100%
        );
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}


/* HERO */

.hero {
    padding: 3rem;
    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a8a 55%,
            #312e81 100%
        );

    color: white;

    box-shadow:
        0 20px 50px rgba(15, 23, 42, 0.18);

    margin-bottom: 2rem;
}

.hero-badge {
    display: inline-block;

    padding: 0.45rem 0.9rem;

    border-radius: 999px;

    background: rgba(255,255,255,0.12);

    border: 1px solid rgba(255,255,255,0.18);

    color: #dbeafe;

    font-size: 0.75rem;

    font-weight: 700;

    letter-spacing: 0.06em;

    margin-bottom: 1rem;
}

.hero-title {
    color: white;

    font-size: 3.2rem;

    font-weight: 800;

    margin: 0;

    line-height: 1.1;
}

.hero-text {
    color: #dbeafe;

    font-size: 1.05rem;

    line-height: 1.7;

    max-width: 850px;

    margin-top: 1rem;
}


/* METRIC CARDS */

.metric-card {
    background: white;

    border: 1px solid #e2e8f0;

    border-radius: 18px;

    padding: 1.3rem;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.06);

    min-height: 125px;
}

.metric-label {
    color: #64748b;

    font-size: 0.75rem;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 0.07em;
}

.metric-value {
    color: #111827;

    font-size: 2rem;

    font-weight: 800;

    margin-top: 0.35rem;
}

.metric-description {
    color: #94a3b8;

    font-size: 0.78rem;

    margin-top: 0.25rem;
}


/* TOPIC CARDS */

.topic-card {
    background: white;

    border: 1px solid #e2e8f0;

    border-radius: 20px;

    padding: 1.4rem;

    margin-bottom: 1rem;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.055);
}

.topic-number {
    color: #6366f1;

    font-size: 0.75rem;

    font-weight: 800;

    letter-spacing: 0.08em;

    text-transform: uppercase;
}

.topic-name {
    color: #111827;

    font-size: 1.25rem;

    font-weight: 800;

    margin-top: 0.25rem;

    margin-bottom: 0.6rem;
}

.topic-words {
    color: #64748b;

    font-size: 0.9rem;

    line-height: 1.6;
}


/* INFO CARDS */

.info-card {
    background: white;

    border: 1px solid #e2e8f0;

    border-radius: 18px;

    padding: 1.5rem;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.05);
}


/* FOOTER */

.footer {
    text-align: center;

    color: #94a3b8;

    font-size: 0.8rem;

    padding-top: 2rem;

    padding-bottom: 1rem;
}

</style>
""")


# =========================================================
# NLTK RESOURCES
# =========================================================

@st.cache_resource
def load_nltk():

    nltk.download(
        "stopwords",
        quiet=True
    )

    nltk.download(
        "wordnet",
        quiet=True
    )

    nltk.download(
        "omw-1.4",
        quiet=True
    )

    return (
        set(stopwords.words("english")),
        WordNetLemmatizer()
    )


stop_words, lemmatizer = load_nltk()


# =========================================================
# PREPROCESSING
# =========================================================

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


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(
        "data/news-article-categories.csv"
    )

    df["body"] = df["body"].fillna("")

    df["text"] = (
        df["title"] + " " + df["body"]
    )

    return df


df = load_dataset()


# =========================================================
# LOAD LDA MODEL
# =========================================================

@st.cache_resource
def load_model():

    dictionary = Dictionary.load(
        "dictionary.gensim"
    )

    lda_model = LdaModel.load(
        "lda_model.gensim"
    )

    return dictionary, lda_model


dictionary, lda_model = load_model()


# =========================================================
# TOPIC INTERPRETATIONS
# =========================================================

interpretations = {

    0: "People, Life & General Social Issues",

    1: "Politics, Trump & Media",

    2: "Education, Community & Social Issues",

    3: "Law, Police & Courts",

    4: "Technology & Business",

    5: "Sexual Assault, Harassment & Abuse",

    6: "Science, Environment & World"
}


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.html("""
    <div style="
        padding-top:10px;
        padding-bottom:15px;
    ">

        <div style="
            font-size:1.45rem;
            font-weight:800;
        ">
            🧠 NewsScope
        </div>

        <div style="
            color:#cbd5e1;
            font-size:0.82rem;
            margin-top:4px;
        ">
            LDA Topic Modeling Dashboard
        </div>

    </div>
    """)

    st.divider()

    st.markdown("### 📌 Project")

    st.caption(
        "Discover hidden themes in news articles "
        "using Latent Dirichlet Allocation."
    )

    st.markdown("### ⚙️ Model")

    st.write("**Algorithm:** LDA")
    st.write("**Topics:** 7")
    st.write("**Inference:** Variational")
    st.write("**Passes:** 5")
    st.write("**Iterations:** 100")

    st.divider()

    st.markdown("### 📊 Evaluation")

    st.write("**Coherence:** 0.4985")
    st.write("**Documents:** 6,877")
    st.write("**Vocabulary:** 20,837")

    st.divider()

    st.caption(
        "Academic Project • LDA Topic Modeling"
    )


# =========================================================
# HERO
# =========================================================

st.html("""
<div class="hero">

    <div class="hero-badge">
        ARTIFICIAL INTELLIGENCE • NLP • TOPIC MODELING
    </div>

    <div class="hero-title">
        NewsScope
    </div>

    <div class="hero-text">
        Discover hidden topics and patterns in thousands
        of news articles using Latent Dirichlet Allocation.
        Explore topic distributions, visualize important
        words, and analyze your own articles.
    </div>

</div>
""")


# =========================================================
# DASHBOARD METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.html("""
    <div class="metric-card">

        <div class="metric-label">
            Documents
        </div>

        <div class="metric-value">
            6,877
        </div>

        <div class="metric-description">
            News articles analyzed
        </div>

    </div>
    """)


with c2:

    st.html("""
    <div class="metric-card">

        <div class="metric-label">
            Topics
        </div>

        <div class="metric-value">
            7
        </div>

        <div class="metric-description">
            Discovered by LDA
        </div>

    </div>
    """)


with c3:

    st.html("""
    <div class="metric-card">

        <div class="metric-label">
            Vocabulary
        </div>

        <div class="metric-value">
            20,837
        </div>

        <div class="metric-description">
            Filtered terms
        </div>

    </div>
    """)


with c4:

    st.html("""
    <div class="metric-card">

        <div class="metric-label">
            Coherence
        </div>

        <div class="metric-value">
            0.4985
        </div>

        <div class="metric-description">
            Topic coherence score
        </div>

    </div>
    """)


st.write("")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Dataset",
        "🧠 Discover Topics",
        "📄 Document Analysis",
        "📈 Visualizations",
        "✍️ Analyze Article",
        "📊 Model Evaluation"
    ]
)


# =========================================================
# TAB 1 — DATASET
# =========================================================

with tab1:

    st.markdown("## Dataset Overview")

    st.caption(
        "Explore the news article collection used by the project."
    )

    col1, col2 = st.columns(2)


    with col1:

        st.html("""
        <div class="info-card">

            <h3>📰 About the Dataset</h3>

            <p>
                The dataset contains news articles with
                category labels, titles, and article bodies.
                For LDA training, the title and article body
                are combined into a single document.
            </p>

        </div>
        """)


    with col2:

        st.html(f"""
        <div class="info-card">

            <h3>🔢 Dataset Statistics</h3>

            <p>
                <b>Articles:</b> {len(df):,}
            </p>

            <p>
                <b>Categories:</b> {df["category"].nunique()}
            </p>

            <p>
                <b>Vocabulary:</b> 20,837
            </p>

        </div>
        """)


    st.write("")

    st.markdown("### 🔍 Article Preview")

    st.dataframe(
        df[
            ["category", "title"]
        ].head(10),

        use_container_width=True,

        hide_index=True
    )


    st.markdown("### 📚 Category Distribution")

    category_counts = (
        df["category"]
        .value_counts()
        .reset_index()
    )

    category_counts.columns = [
        "Category",
        "Articles"
    ]

    st.bar_chart(
        category_counts.set_index(
            "Category"
        )
    )


    st.info(
        "The category column is not used as input "
        "when training the LDA model."
    )


# =========================================================
# TAB 2 — DISCOVERED TOPICS
# =========================================================

with tab2:

    st.markdown("## 🧠 Discovered Topics")

    st.caption(
        "Seven latent topics discovered by the trained LDA model."
    )


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


        st.html(f"""
        <div class="topic-card">

            <div class="topic-number">
                TOPIC {topic_id + 1}
            </div>

            <div class="topic-name">
                {interpretations[topic_id]}
            </div>

            <div class="topic-words">
                {" • ".join(words)}
            </div>

        </div>
        """)


        with st.expander(
            f"View word probabilities — Topic {topic_id + 1}"
        ):

            topic_table = pd.DataFrame(
                topic_words,
                columns=[
                    "Word",
                    "Probability"
                ]
            )

            topic_table["Probability"] = (
                topic_table["Probability"]
                .round(4)
            )

            st.dataframe(
                topic_table,
                use_container_width=True,
                hide_index=True
            )


    # DOWNLOAD TOPIC SUMMARY

    st.markdown("### ⬇️ Download Results")


    rows = []


    for topic_id in range(
        lda_model.num_topics
    ):

        topic_words = lda_model.show_topic(
            topic_id,
            topn=10
        )

        rows.append({
            "Topic":
                f"Topic {topic_id + 1}",

            "Interpretation":
                interpretations[topic_id],

            "Top Words":
                ", ".join(
                    word
                    for word, probability
                    in topic_words
                ),

            "Probabilities":
                ", ".join(
                    str(round(probability, 4))
                    for word, probability
                    in topic_words
                )
        })


    topic_summary = pd.DataFrame(
        rows
    )


    st.download_button(
        "⬇️ Download Topic Summary",

        topic_summary.to_csv(
            index=False
        ).encode("utf-8"),

        "topic_summary.csv",

        "text/csv"
    )


# =========================================================
# TAB 3 — DOCUMENT ANALYSIS
# =========================================================

with tab3:

    st.markdown("## 📄 Document Analysis")

    st.caption(
        "See how an article is distributed across the seven topics."
    )


    selected = st.slider(
        "Select article",

        min_value=1,

        max_value=len(df),

        value=1
    )


    article = df.iloc[
        selected - 1
    ]


    st.html(f"""
    <div class="info-card">

        <h3>
            {article["title"]}
        </h3>

        <p>
            <b>Dataset Category:</b>
            {article["category"]}
        </p>

    </div>
    """)


    processed = preprocess_text(
        article["text"]
    )


    bow = dictionary.doc2bow(
        processed
    )


    distribution = lda_model[
        bow
    ]


    distribution = sorted(
        distribution,
        key=lambda x: x[1],
        reverse=True
    )


    st.markdown("### 🎯 Topic Distribution")


    topic_rows = []


    for topic_id in range(
        lda_model.num_topics
    ):

        probability = 0.0


        for tid, prob in distribution:

            if tid == topic_id:

                probability = prob

                break


        topic_rows.append({
            "Topic":
                f"Topic {topic_id + 1}",

            "Interpretation":
                interpretations[topic_id],

            "Probability":
                round(
                    probability,
                    4
                )
        })


        st.write(
            f"**Topic {topic_id + 1} — "
            f"{interpretations[topic_id]}**"
        )


        st.progress(
            float(
                min(
                    probability,
                    1.0
                )
            )
        )


    document_df = pd.DataFrame(
        topic_rows
    )


    st.dataframe(
        document_df,

        use_container_width=True,

        hide_index=True
    )


    if distribution:

        top_topic = distribution[0][0]

        top_probability = distribution[0][1]

        st.success(
            f"🎯 Primary Topic: "
            f"**Topic {top_topic + 1} — "
            f"{interpretations[top_topic]}** "
            f"({top_probability:.2%})"
        )


    st.download_button(
        "⬇️ Download Document Analysis",

        document_df.to_csv(
            index=False
        ).encode("utf-8"),

        "document_topic_distribution.csv",

        "text/csv"
    )


# =========================================================
# TAB 4 — VISUALIZATIONS
# =========================================================

with tab4:

    st.markdown("## 📈 Topic Visualizations")

    st.caption(
        "Top words and their probability within each topic."
    )


    for topic_id in range(
        lda_model.num_topics
    ):

        st.markdown(
            f"### Topic {topic_id + 1} — "
            f"{interpretations[topic_id]}"
        )


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


        fig, ax = plt.subplots(
            figsize=(10, 4.5)
        )


        ax.barh(
            words[::-1],
            probabilities[::-1]
        )


        ax.set_xlabel(
            "P(word | topic)"
        )

        ax.set_ylabel(
            "Word"
        )

        ax.grid(
            axis="x",
            alpha=0.2
        )

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)


        plt.tight_layout()


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


# =========================================================
# TAB 5 — ANALYZE YOUR OWN ARTICLE
# =========================================================

with tab5:

    st.markdown("## ✍️ Analyze Your Own Article")

    st.caption(
        "Paste a news article and discover its topic distribution."
    )


    user_article = st.text_area(
        "Article text",

        height=280,

        placeholder="Paste your news article here..."
    )


    analyze_button = st.button(
        "🔍 Analyze Article",
        use_container_width=True
    )


    if analyze_button:

        if not user_article.strip():

            st.warning(
                "Please enter some article text first."
            )

        else:

            processed = preprocess_text(
                user_article
            )


            bow = dictionary.doc2bow(
                processed
            )


            distribution = lda_model[
                bow
            ]


            distribution = sorted(
                distribution,
                key=lambda x: x[1],
                reverse=True
            )


            if not distribution:

                st.warning(
                    "The article does not contain enough "
                    "words from the model vocabulary."
                )

            else:

                top_topic = distribution[0][0]

                top_probability = distribution[0][1]


                st.success(
                    f"🎯 Primary Topic: "
                    f"**Topic {top_topic + 1} — "
                    f"{interpretations[top_topic]}**"
                )


                st.write(
                    f"Highest probability: "
                    f"**{top_probability:.2%}**"
                )


                st.markdown(
                    "### 📊 Topic Probabilities"
                )


                for topic_id in range(
                    lda_model.num_topics
                ):

                    probability = 0.0


                    for tid, prob in distribution:

                        if tid == topic_id:

                            probability = prob

                            break


                    st.write(
                        f"**Topic {topic_id + 1} — "
                        f"{interpretations[topic_id]}**"
                    )


                    st.progress(
                        float(
                            min(
                                probability,
                                1.0
                            )
                        )
                    )


                    st.caption(
                        f"{probability:.2%}"
                    )


# =========================================================
# TAB 6 — MODEL EVALUATION
# =========================================================

with tab6:

    st.markdown("## 📊 Model Evaluation")

    st.caption(
        "Evaluate the trained LDA model and compare different topic counts."
    )


    # -----------------------------------------------------
    # CURRENT MODEL METRICS
    # -----------------------------------------------------

    eval1, eval2, eval3 = st.columns(3)


    with eval1:

        st.html("""
        <div class="metric-card">

            <div class="metric-label">
                Coherence Score
            </div>

            <div class="metric-value">
                0.4985
            </div>

            <div class="metric-description">
                Topic interpretability
            </div>

        </div>
        """)


    with eval2:

        st.html("""
        <div class="metric-card">

            <div class="metric-label">
                Log Perplexity
            </div>

            <div class="metric-value">
                -8.4373
            </div>

            <div class="metric-description">
                Model likelihood measure
            </div>

        </div>
        """)


    with eval3:

        st.html("""
        <div class="metric-card">

            <div class="metric-label">
                Final Topics
            </div>

            <div class="metric-value">
                7
            </div>

            <div class="metric-description">
                Final LDA configuration
            </div>

        </div>
        """)


    st.write("")


    # -----------------------------------------------------
    # EVALUATION EXPLANATION
    # -----------------------------------------------------

    st.markdown("### 🔬 Evaluation Method")

    st.info(
        """
        The LDA model was evaluated using topic coherence and
        log perplexity. Several topic counts were tested to
        examine the resulting topic structure. Topic coherence
        was used as the main measure for comparing topic
        interpretability.
        """
    )


    # -----------------------------------------------------
    # TOPIC COMPARISON FILE
    # -----------------------------------------------------

    if os.path.exists(
        "topic_comparison.csv"
    ):

        comparison_df = pd.read_csv(
            "topic_comparison.csv"
        )


        st.markdown(
            "### 📈 Topic Count Comparison"
        )


        st.dataframe(
            comparison_df,

            use_container_width=True,

            hide_index=True
        )


        # -------------------------------------------------
        # IDENTIFY TOPIC COLUMN
        # -------------------------------------------------

        topic_column = None

        for column in comparison_df.columns:

            name = column.lower()

            if (
                "topic" in name
                or "num" in name
            ):

                topic_column = column

                break


        # -------------------------------------------------
        # COHERENCE COLUMN
        # -------------------------------------------------

        coherence_column = None

        for column in comparison_df.columns:

            if "coherence" in column.lower():

                coherence_column = column

                break


        # -------------------------------------------------
        # COHERENCE CHART
        # -------------------------------------------------

        if (
            topic_column is not None
            and coherence_column is not None
        ):

            st.markdown(
                "### 📊 Coherence by Number of Topics"
            )


            coherence_chart = comparison_df[
                [
                    topic_column,
                    coherence_column
                ]
            ].copy()


            coherence_chart = (
                coherence_chart
                .set_index(topic_column)
            )


            st.bar_chart(
                coherence_chart
            )


        # -------------------------------------------------
        # PERPLEXITY COLUMN
        # -------------------------------------------------

        perplexity_column = None

        for column in comparison_df.columns:

            if "perplexity" in column.lower():

                perplexity_column = column

                break


        # -------------------------------------------------
        # PERPLEXITY CHART
        # -------------------------------------------------

        if (
            topic_column is not None
            and perplexity_column is not None
        ):

            st.markdown(
                "### 📉 Log Perplexity by Number of Topics"
            )


            perplexity_chart = comparison_df[
                [
                    topic_column,
                    perplexity_column
                ]
            ].copy()


            perplexity_chart = (
                perplexity_chart
                .set_index(topic_column)
            )


            st.line_chart(
                perplexity_chart
            )


        # -------------------------------------------------
        # DOWNLOAD COMPARISON
        # -------------------------------------------------

        st.markdown(
            "### ⬇️ Download Evaluation"
        )


        st.download_button(
            "Download Model Comparison",

            comparison_df.to_csv(
                index=False
            ).encode("utf-8"),

            "topic_comparison.csv",

            "text/csv"
        )


    else:

        st.warning(
            "topic_comparison.csv was not found. "
            "Run compare_topics.py first."
        )


    # -----------------------------------------------------
    # FINAL MODEL CONFIGURATION
    # -----------------------------------------------------

    st.markdown(
        "### ⚙️ Final Model Configuration"
    )


    final_config = pd.DataFrame({

        "Parameter": [

            "Algorithm",

            "Number of Topics",

            "Vocabulary Size",

            "Passes",

            "Iterations",

            "Random State",

            "Inference Method"

        ],

        "Value": [

            "Latent Dirichlet Allocation",

            "7",

            "20,837",

            "5",

            "100",

            "42",

            "Online Variational Bayes"

        ]

    })


    st.dataframe(
        final_config,

        use_container_width=True,

        hide_index=True
    )


    # -----------------------------------------------------
    # FINAL MODEL SUMMARY
    # -----------------------------------------------------

    st.success(
        """
        The final LDA model contains 7 topics and a vocabulary
        of 20,837 filtered terms. The model achieved a coherence
        score of 0.4985 on the evaluation dataset.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    🧠 NewsScope • LDA Topic Modeling • Natural Language Processing

    <br><br>

    Built with Python • Gensim • Pandas • Streamlit

</div>
""")
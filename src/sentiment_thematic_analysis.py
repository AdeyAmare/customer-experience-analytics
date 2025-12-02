import os
import re
import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# --- CONFIGURATION & SETUP ---
try:
    from utils.config import DATA_PATHS
except ImportError:
    print("Warning: utils/config.py not found. Using default paths.")
    DATA_PATHS = {
        "processed_reviews": "data/processed/processed_reviews.csv",
        "sentiment_results": "data/results/sentiment_thematic_results.csv",
    }

# Ensure required NLTK data is present (quiet to avoid noisy outputs)
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize STOPWORDS set
STOPWORDS = set(stopwords.words("english"))

# Load spaCy model (download if missing)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Enable tqdm for pandas operations
tqdm.pandas()

# Load DistilBERT sentiment model (optional)
print("Loading DistilBERT Sentiment Model...")
try:
    SENTIMENT_MODEL = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"⚠️ Could not load DistilBERT model. Error: {e}")
    SENTIMENT_MODEL = None

# --- THEME KEYWORDS ---
THEME_KEYWORDS = {
    "Mobile App & Digital": [
        "app", "login", "update", "interface", "screen", "feature",
        "digital", "mobile", "phone", "crash", "biometric", "ui", "ux"
    ],
    "Customer Service": [
        "service", "staff", "manager", "support", "wait", "queue", "call",
        "agent", "teller", "rude", "polite", "help", "cs"
    ],
    "Transactions & Fees": [
        "money", "transfer", "transaction", "fee", "charge", "balance",
        "deposit", "withdrawal", "payment", "rate", "limit"
    ],
    "Security & Fraud": [
        "fraud", "scam", "security", "alert", "hack", "safe",
        "verification", "otp", "locked", "blocked", "account"
    ],
    "Cards & Accounts": [
        "card", "debit", "credit", "account", "opening", "closure",
        "limit", "atm", "branch"
    ]
}

# --- SENTIMENT CLASS ---
class SentimentNLP:
    """Sentiment analysis helper that runs VADER, TextBlob and (optionally) DistilBERT.

    This class handles loading input review data, running three sentiment
    approaches (VADER, TextBlob, and DistilBERT if available), and exposing
    the dataframe with sentiment columns.

    Attributes:
        input_path (str): Path to input CSV containing processed reviews.
        output_path (str): Where final results are intended to be stored.
        df (pd.DataFrame|None): DataFrame containing loaded reviews and results.
        analyzer (SentimentIntensityAnalyzer): VADER analyzer instance.
    """

    def __init__(self, input_path=None, output_path=None):
        """Initialize SentimentNLP.

        Args:
            input_path (str, optional): Custom path to processed reviews CSV.
            output_path (str, optional): Path where results will be saved later.
        """
        self.input_path = input_path or DATA_PATHS["processed_reviews"]
        self.output_path = output_path or DATA_PATHS["sentiment_results"]
        self.df = None
        self.analyzer = SentimentIntensityAnalyzer()

    def load_data(self):
        """Load review data into the object's dataframe.

        If the configured input file does not exist, this method creates a small
        dummy dataframe for testing and development. The method ensures the
        `review_text` column exists and is string typed.

        Raises:
            ValueError: If loaded file does not contain a 'review_text' column.
        """
        print(f"Loading data from {self.input_path}...")
        if not os.path.exists(self.input_path):
            print("⚠️ File not found. Creating dummy data for testing...")
            data = {
                "bank_name": ["Bank A", "Bank B", "Bank A", "Bank B", "Bank A", "Bank B"],
                "review_text": [
                    "The app crashes every time I login, which is super annoying.",
                    "Great customer service and friendly staff at the branch.",
                    "High fees on transfer! I hate the new charge.",
                    "I love the security features and biometrics. Very safe.",
                    "It's okay, nothing special, but the credit card limit is low.",
                    "The mobile banking is fine, but the wait time to speak to an agent is terrible."
                ],
                "rating": [1, 5, 2, 5, 3, 2]
            }
            self.df = pd.DataFrame(data)
        else:
            self.df = pd.read_csv(self.input_path)

        # Validate and normalize
        if "review_text" not in self.df.columns:
            raise ValueError("Input data must contain a 'review_text' column.")
        self.df["review_text"] = self.df["review_text"].fillna("").astype(str)
        print(f"✅ Loaded {len(self.df)} reviews.")

    def _get_label(self, score):
        """Convert a numeric sentiment score to a POSITIVE/NEGATIVE/NEUTRAL label.

        Args:
            score (float): Numeric score (typically -1.0..1.0).

        Returns:
            str: One of "POSITIVE", "NEGATIVE", or "NEUTRAL".
        """
        if score >= 0.05:
            return "POSITIVE"
        elif score <= -0.05:
            return "NEGATIVE"
        else:
            return "NEUTRAL"

    def _run_vader(self):
        """Run VADER sentiment analysis and attach compound scores and labels.

        Adds two columns to self.df:
            - vader_score: float compound score
            - vader_label: mapped label via _get_label
        """
        print("-> Running VADER...")
        # Compute compound score using VADER
        self.df["vader_score"] = self.df["review_text"].progress_apply(
            lambda t: self.analyzer.polarity_scores(t)["compound"]
        )
        # Map compound score to label
        self.df["vader_label"] = self.df["vader_score"].apply(self._get_label)

    def _run_textblob(self):
        """Run TextBlob sentiment polarity and attach scores and labels.

        Adds:
            - textblob_score: float polarity (-1..1)
            - textblob_label: mapped label via _get_label
        """
        print("-> Running TextBlob...")
        def _tb_score(text):
            if not text:
                return 0.0
            try:
                return TextBlob(text).sentiment.polarity
            except Exception:
                # In case TextBlob fails unexpectedly, return neutral
                return 0.0

        self.df["textblob_score"] = self.df["review_text"].progress_apply(_tb_score)
        self.df["textblob_label"] = self.df["textblob_score"].apply(self._get_label)

    def _run_distilbert(self):
        """Run DistilBERT sentiment classifier if model is available.

        If the global SENTIMENT_MODEL is None, this method will skip running
        the model and populate neutral defaults for the related columns.

        Adds:
            - distilbert_score: confidence score (0..1)
            - distilbert_label: model label (e.g., "POSITIVE"/"NEGATIVE"/other)
        """
        if SENTIMENT_MODEL is None:
            print("-> Skipping DistilBERT: Model not loaded.")
            # preserve existing column names but set neutral defaults
            self.df["distilbert_score"] = 0.0
            self.df["distilbert_label"] = "NEUTRAL"
            return

        print("-> Running DistilBERT...")

        def get_sentiment(text):
            """Return model result dict for a single text safely."""
            if not text.strip():
                return {"label": "NEUTRAL", "score": 0.0}
            try:
                # pipeline returns a list like [{'label': 'POSITIVE', 'score': 0.99}]
                return SENTIMENT_MODEL(text, truncation=True)[0]
            except Exception:
                # On any failure return a neutral placeholder
                return {"label": "NEUTRAL", "score": 0.0}

        # Apply model to each review text and unpack results
        results = self.df["review_text"].progress_apply(get_sentiment)
        self.df["distilbert_label"] = results.apply(lambda x: x.get("label", "NEUTRAL"))
        self.df["distilbert_score"] = results.apply(lambda x: x.get("score", 0.0))

    def run_analysis(self):
        """Run the configured sentiment analyses in sequence.

        Order:
            1. VADER (lexicon-based)
            2. TextBlob (polarity)
            3. DistilBERT (transformer-based, optional)

        Raises:
            ValueError: if data has not been loaded via load_data().
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        print("Running all three Sentiment Models...")
        self._run_vader()
        self._run_textblob()
        self._run_distilbert()
        print("✅ Sentiment Analysis complete.")

    def aggregate_sentiment(self):
        """Print aggregated sentiment means grouped by bank_name and rating.

        This is a lightweight utility to inspect mean sentiment scores per bank
        and rating level. It collects any column with "score" in the name.
        """
        print("\n--- Aggregated Sentiment (Mean Score) ---")
        score_cols = [c for c in self.df.columns if "score" in c]
        if not score_cols:
            print("Skipping aggregation: Sentiment scores not available.")
            return
        agg_df = self.df.groupby(["bank_name", "rating"])[score_cols].mean().reset_index()
        print(agg_df)

    def get_dataframe(self):
        """Return the internal dataframe with sentiment results.

        Returns:
            pd.DataFrame: Dataframe with original reviews and sentiment columns.
        """
        return self.df


# --- THEMATIC CLASS ---
class ThematicNLP:
    """Thematic analysis using TF-IDF keyword extraction and keyword->theme mapping.

    The class expects a dataframe with a 'review_text' column and will:
      - Preprocess text (basic cleaning, lemmatization, stopword removal)
      - Extract top TF-IDF keywords for each document
      - Map keywords to coarse-grained themes via THEME_KEYWORDS

    Attributes:
        df (pd.DataFrame): Working dataframe copy for thematic processing.
    """

    def __init__(self, df):
        """Initialize ThematicNLP with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame which must contain a 'review_text' column.
        """
        if "review_text" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'review_text' column.")
        self.df = df.copy()
        self.vectorizer_params = {'min_df': 2, 'max_df': 0.85, 'stop_words': 'english', 'ngram_range': (1, 2)}

    def preprocess_text_for_tfidf(self, text):
        """Preprocess a single text string for TF-IDF.

        Steps:
            - Lowercase
            - Remove non-alpha characters
            - Tokenize and lemmatize with spaCy
            - Remove stopwords and tokens shorter than 3 characters

        Args:
            text (str): Raw review text.

        Returns:
            str: Space-joined preprocessed tokens suitable for TF-IDF.
        """
        text = str(text).lower()
        # keep only letters and spaces
        text = re.sub(r"[^a-z\s]", " ", text)
        # run through spaCy
        doc = nlp(text)
        tokens = [
            token.lemma_ for token in doc
            if token.text not in STOPWORDS and len(token.text) > 2 and token.is_alpha
        ]
        return " ".join(tokens)

    def extract_tfidf_keywords(self):
        """Compute TF-IDF across documents and extract top keywords per document.

        The method stores extracted keywords per-row in a new column
        'extracted_keywords'. If there is insufficient non-empty text, it
        sets the column to empty lists and returns early.

        Notes:
            - Uses unigrams and bigrams (1,2)
            - Filters extremely common tokens with max_df=0.85
            - Filters very rare tokens with min_df=2 (matching original logic)
        """
        print("Preprocessing text for TF-IDF...")
        # Create preprocessed text column
        self.df["preprocessed_text"] = self.df["review_text"].progress_apply(
            self.preprocess_text_for_tfidf
        )

        # If all preprocessed texts are empty or whitespace, bail out
        if self.df["preprocessed_text"].str.strip().eq("").all():
            print("⚠️ Not enough non-empty text for TF-IDF.")
            # Keep schema consistent: create extracted_keywords column with empty lists
            self.df["extracted_keywords"] = [[] for _ in range(len(self.df))]
            # drop preprocessed text to avoid clutter
            self.df.drop(columns=["preprocessed_text"], inplace=True)
            return

        print("Calculating TF-IDF scores...")
        # Initialize vectorizer with same configuration as original
        vectorizer = TfidfVectorizer(**self.vectorizer_params)
        # Fit-transform the preprocessed corpus
        tfidf_matrix = vectorizer.fit_transform(self.df["preprocessed_text"])
        feature_names = vectorizer.get_feature_names_out()

        def top_keywords(i, top_n=5):
            """Return the top_n features for document index i by TF-IDF score."""
            scores = tfidf_matrix[i].toarray().flatten()
            # If document has all-zero vector (no features), return empty list
            if not np.any(scores):
                return []
            indices = scores.argsort()[-top_n:][::-1]
            return [feature_names[j] for j in indices if scores[j] > 0]

        # Extract top keywords for each document
        self.df["extracted_keywords"] = [
            top_keywords(i) for i in tqdm(range(len(self.df)), desc="Extracting Keywords")
        ]

        # Clean up temporary column
        self.df.drop(columns=["preprocessed_text"], inplace=True)

    def assign_theme(self, keywords):
        """Map a list of keywords to one or more themes using THEME_KEYWORDS.

        The mapping is a simple membership test: each keyword and its tokenized
        parts are checked against the theme keyword lists.

        Args:
            keywords (list[str]): Extracted keywords for a single document.

        Returns:
            str: Comma-separated list of matched themes or 'General / Unspecified'.
        """
        terms = set()
        for kw in keywords:
            if not kw:
                continue
            # add the exact keyword (could be bigram) and its parts
            terms.add(kw)
            terms.update(kw.split())

        found = [
            theme for theme, kws in THEME_KEYWORDS.items()
            if any(t in kws for t in terms)
        ]
        if found:
            # return sorted unique list for determinism
            return ", ".join(sorted(set(found)))
        else:
            return "General / Unspecified"

    def process(self):
        """Run the full thematic extraction and theme assignment pipeline.

        Returns:
            pd.DataFrame: DataFrame with a new 'identified_theme' column.
        """
        print("Running Thematic Analysis...")
        self.extract_tfidf_keywords()
        # Map extracted keywords to themes; ensure result present even when keywords empty
        self.df["identified_theme"] = self.df["extracted_keywords"].progress_apply(self.assign_theme)
        # Remove the temporary extracted keywords column to keep final output tidy
        self.df.drop(columns=["extracted_keywords"], inplace=True)
        print("✅ Theme assignment complete.")
        return self.df


# --- PIPELINE CLASS ---
class BankReviewPipeline:
    """A small pipeline orchestrator that runs sentiment and thematic analysis.

    Usage:
        pipeline = BankReviewPipeline(input_path="...", output_path="...")
        final_df = pipeline.run()

    The pipeline performs:
        1. Load data
        2. Run sentiment analyses
        3. Aggregate and print basic sentiment stats
        4. Run thematic analysis (TF-IDF + theme mapping)
        5. Save final DataFrame to CSV at output_path
    """

    def __init__(self, input_path=None, output_path=None):
        """Initialize the orchestrator.

        Args:
            input_path (str, optional): Path to the input CSV (overrides config).
            output_path (str, optional): Path to the final CSV results.
        """
        self.sentiment = SentimentNLP(input_path, output_path)
        self.output_path = output_path or DATA_PATHS["sentiment_results"]

    def run(self):
        """Execute the full pipeline and persist results.

        Returns:
            pd.DataFrame: Final DataFrame containing original data plus sentiment
                          and identified_theme columns.

        Side effects:
            - Prints progress messages to stdout
            - Writes final CSV to self.output_path (directory created if needed)
        """
        print("\n--- Starting Bank Review NLP Pipeline ---")
        # Load and analyze sentiment
        self.sentiment.load_data()
        self.sentiment.run_analysis()
        df_sentiment = self.sentiment.get_dataframe()

        # Print a quick aggregation to help inspection
        self.sentiment.aggregate_sentiment()

        # Run thematic analysis on the sentiment-augmented DataFrame
        thematic = ThematicNLP(df_sentiment)
        final_df = thematic.process()

        # Ensure output directory exists and save
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final_df.to_csv(self.output_path, index=False)
        print(f"\n✅ Saved FINAL results to {self.output_path}")

        # Print theme distribution summary
        print("\n--- Theme Distribution ---")
        try:
            # Some rows may contain comma-separated multiple themes; value_counts works as-is
            print(final_df["identified_theme"].value_counts())
        except Exception:
            print("Could not compute theme distribution (column missing or malformed).")

        print("\n--- Pipeline Run Complete ---")
        return final_df




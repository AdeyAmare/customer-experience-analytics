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

from utils.config import DATA_PATHS

# --- CONFIGURATION & SETUP ---
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

tqdm.pandas()

# --- RULE-BASED THEME DICTIONARY ---
THEME_KEYWORDS = {
    "Mobile App & Digital": [
        "app", "login", "update", "interface", "screen", "feature", 
        "digital", "mobile", "phone", "crash", "biometric"
    ],
    "Customer Service": [
        "service", "staff", "manager", "support", "wait", "queue", 
        "call", "agent", "teller", "rude", "polite", "help"
    ],
    "Transactions & Fees": [
        "money", "transfer", "transaction", "fee", "charge", 
        "balance", "deposit", "withdrawal", "payment", "rate"
    ],
    "Security & Fraud": [
        "fraud", "scam", "security", "alert", "hack", "safe", 
        "verification", "otp", "locked", "blocked"
    ],
    "Cards & Accounts": [
        "card", "debit", "credit", "account", "opening", "closure", 
        "limit", "atm"
    ]
}

class SentimentNLP:
    """Pipeline for computing VADER and TextBlob sentiment scores and labels."""

    def __init__(self, input_path: str = None, output_path: str = None):
        """Initialize SentimentNLP.

        Args:
            input_path (str, optional): Path to input CSV containing reviews. Defaults to DATA_PATHS["processed_reviews"].
            output_path (str, optional): Path to save sentiment results. Defaults to DATA_PATHS["sentiment_results"].
        """
        self.input_path = input_path or DATA_PATHS["processed_reviews"]
        self.output_path = output_path or DATA_PATHS["sentiment_results"]
        self.df: pd.DataFrame = None

    def load_data(self) -> None:
        """Load review data from CSV or create dummy data if file does not exist.

        Fills missing review_text with empty strings and converts to string type.
        """
        print(f"Loading data from {self.input_path}...")
        if not os.path.exists(self.input_path):
            print("⚠️ File not found. Creating dummy data for testing...")
            data = {
                "bank_name": ["Bank A", "Bank B", "Bank A", "Bank B"],
                "review_text": [
                    "The app crashes every time I login.",
                    "Great customer service and friendly staff.",
                    "High fees on transfer!",
                    "I love the security features."
                ],
                "rating": [1, 5, 2, 5]
            }
            self.df = pd.DataFrame(data)
        else:
            self.df = pd.read_csv(self.input_path)
        
        self.df["review_text"] = self.df["review_text"].fillna("").astype(str)
        print(f"✅ Loaded {len(self.df)} reviews.")

    def run_analysis(self) -> None:
        """Compute sentiment scores and labels using VADER and TextBlob."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Running Sentiment Models...")
        analyzer = SentimentIntensityAnalyzer()

        # VADER scores
        self.df["vader_score"] = self.df["review_text"].progress_apply(
            lambda t: analyzer.polarity_scores(t)["compound"]
        )
        self.df["vader_label"] = self.df["vader_score"].apply(
            lambda s: "POSITIVE" if s >= 0.05 else ("NEGATIVE" if s <= -0.05 else "NEUTRAL")
        )

        # TextBlob scores
        def get_tb(text: str) -> float:
            try:
                return TextBlob(text).sentiment.polarity
            except Exception:
                return 0.0

        self.df["textblob_score"] = self.df["review_text"].progress_apply(get_tb)
        self.df["textblob_label"] = self.df["textblob_score"].apply(
            lambda s: "POSITIVE" if s >= 0.05 else ("NEGATIVE" if s <= -0.05 else "NEUTRAL")
        )
        print("✅ Sentiment Analysis complete.")

    def get_dataframe(self) -> pd.DataFrame:
        """Return the DataFrame containing sentiment results.

        Returns:
            pd.DataFrame: Reviews with sentiment scores and labels.
        """
        return self.df


class ThematicNLP:
    """Assigns themes to reviews based on noun extraction and a rule-based dictionary."""

    def __init__(self, df: pd.DataFrame):
        """Initialize ThematicNLP with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing reviews.
        """
        self.df = df.copy()

    def preprocess_nouns_only(self, text: str) -> list[str]:
        """Extract nouns and proper nouns from text after cleaning.

        Args:
            text (str): Raw review text.

        Returns:
            list[str]: List of lemmatized noun tokens.
        """
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        doc = nlp(text)
        tokens = [
            token.lemma_ for token in doc
            if token.pos_ in ["NOUN", "PROPN"]
            and token.text not in STOPWORDS
            and len(token.text) > 2
        ]
        return tokens

    def assign_theme(self, tokens: list[str]) -> str:
        """Assign theme(s) to a review based on extracted nouns.

        Args:
            tokens (list[str]): List of noun tokens from a review.

        Returns:
            str: Comma-separated theme(s) or 'General / Unspecified'.
        """
        found_themes = []
        for theme, keywords in THEME_KEYWORDS.items():
            if any(token in keywords for token in tokens):
                found_themes.append(theme)
        if not found_themes:
            return "General / Unspecified"
        return ", ".join(sorted(set(found_themes)))

    def process(self) -> pd.DataFrame:
        """Run thematic analysis on the DataFrame.

        Returns:
            pd.DataFrame: Reviews with assigned themes.
        """
        print("Running Thematic Analysis (Noun Extraction + Rules)...")
        self.df["noun_tokens"] = self.df["review_text"].progress_apply(self.preprocess_nouns_only)
        self.df["identified_theme"] = self.df["noun_tokens"].progress_apply(self.assign_theme)
        self.df.drop(columns=["noun_tokens"], inplace=True)
        print("✅ Theme assignment complete.")
        return self.df


class BankReviewPipeline:
    """Full pipeline combining sentiment and thematic analysis."""

    def __init__(self, input_path: str = None, output_path: str = None):
        """Initialize pipeline with optional input/output paths.

        Args:
            input_path (str, optional): Path to CSV of processed reviews.
            output_path (str, optional): Path to save final results.
        """
        self.sentiment = SentimentNLP(input_path, output_path)
        self.output_path = output_path or DATA_PATHS["sentiment_results"]

    def run(self) -> pd.DataFrame:
        """Run sentiment analysis, thematic analysis, save results, and show summary.

        Returns:
            pd.DataFrame: DataFrame with sentiment and theme annotations.
        """
        # Sentiment
        self.sentiment.load_data()
        self.sentiment.run_analysis()
        df_sentiment = self.sentiment.get_dataframe()

        # Thematic
        thematic = ThematicNLP(df_sentiment)
        final_df = thematic.process()

        # Save results
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final_df.to_csv(self.output_path, index=False)
        print(f"✅ Saved FINAL results with Themes to {self.output_path}")

        # Show summary
        print("\n--- Theme Distribution ---")
        print(final_df["identified_theme"].value_counts())

        return final_df

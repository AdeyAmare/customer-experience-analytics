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
# Update this path to match your local setup

nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# Load Spacy (ensure you have run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

tqdm.pandas()

# --- RULE-BASED THEME DICTIONARY ---
# This maps specific keywords to the themes required by the task.
THEME_KEYWORDS = {
    "Mobile App & Digital": ["app", "login", "update", "interface", "screen", "feature", "digital", "mobile", "phone", "crash", "biometric"],
    "Customer Service": ["service", "staff", "manager", "support", "wait", "queue", "call", "agent", "teller", "rude", "polite", "help"],
    "Transactions & Fees": ["money", "transfer", "transaction", "fee", "charge", "balance", "deposit", "withdrawal", "payment", "rate"],
    "Security & Fraud": ["fraud", "scam", "security", "alert", "hack", "safe", "verification", "otp", "locked", "blocked"],
    "Cards & Accounts": ["card", "debit", "credit", "account", "opening", "closure", "limit", "atm"]
}

class SentimentNLP:
    """Compute VADER and TextBlob sentiment."""

    def __init__(self, input_path=None, output_path=None):
        self.input_path = input_path or DATA_PATHS["processed_reviews"]
        self.output_path = output_path or DATA_PATHS["sentiment_results"]
        self.df = None

    def load_data(self):
        print(f"Loading data from {self.input_path}...")
        if not os.path.exists(self.input_path):
            # Create dummy data if file missing (for testing purposes)
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

    def run_analysis(self):
        print("Running Sentiment Models...")
        analyzer = SentimentIntensityAnalyzer()
        
        # VADER
        self.df["vader_score"] = self.df["review_text"].progress_apply(lambda t: analyzer.polarity_scores(t)["compound"])
        self.df["vader_label"] = self.df["vader_score"].apply(lambda s: "POSITIVE" if s >= 0.05 else ("NEGATIVE" if s <= -0.05 else "NEUTRAL"))

        # TextBlob
        def get_tb(text):
            try:
                return TextBlob(text).sentiment.polarity
            except:
                return 0.0

        self.df["textblob_score"] = self.df["review_text"].progress_apply(get_tb)
        self.df["textblob_label"] = self.df["textblob_score"].apply(lambda s: "POSITIVE" if s >= 0.05 else ("NEGATIVE" if s <= -0.05 else "NEUTRAL"))
        print("✅ Sentiment Analysis complete.")

    def get_dataframe(self):
        return self.df

class ThematicNLP:
    """Assigns themes based on Noun extraction and Rules."""

    def __init__(self, df):
        self.df = df.copy()

    def preprocess_nouns_only(self, text):
        """
        Uses spaCy to extract ONLY Nouns.
        Nouns represent 'features' (app, fee, staff), which are best for theming.
        """
        text = str(text).lower()
        # Remove special chars
        text = re.sub(r"[^a-z\s]", "", text)
        doc = nlp(text)
        
        # KEEP ONLY NOUNS (pos_ == 'NOUN') and filter stopwords
        tokens = [
            token.lemma_ for token in doc 
            if token.pos_ in ["NOUN", "PROPN"] 
            and token.text not in STOPWORDS 
            and len(token.text) > 2
        ]
        return tokens

    def assign_theme(self, tokens):
        """
        Rule-Based Classification.
        Checks if extracted nouns match our specific Banking Dictionary.
        """
        found_themes = []
        
        # Check every token against our dictionary
        for theme, keywords in THEME_KEYWORDS.items():
            if any(token in keywords for token in tokens):
                found_themes.append(theme)
        
        # If multiple themes found, join them. If none, label 'General'.
        if not found_themes:
            return "General / Unspecified"
        return ", ".join(list(set(found_themes)))

    def process(self):
        print("Running Thematic Analysis (Noun Extraction + Rules)...")
        
        # 1. Extract Nouns
        # We store list of tokens temporarily to match against dictionary
        self.df["noun_tokens"] = self.df["review_text"].progress_apply(self.preprocess_nouns_only)
        
        # 2. Assign Themes (Row-Level)
        self.df["identified_theme"] = self.df["noun_tokens"].progress_apply(self.assign_theme)
        
        # Clean up helper column
        self.df = self.df.drop(columns=["noun_tokens"])
        
        print("✅ Theme assignment complete.")
        return self.df


class BankReviewPipeline:
    """Full pipeline combining sentiment + thematic analysis."""

    def __init__(self, input_path=None, output_path=None):
        self.sentiment = SentimentNLP(input_path, output_path)
        self.output_path = output_path or DATA_PATHS["sentiment_results"]

    def run(self):
        # 1. Run Sentiment
        self.sentiment.load_data()
        self.sentiment.run_analysis()
        df_sentiment = self.sentiment.get_dataframe()

        # 2. Run Thematic (Pass the sentiment DF into Thematic)
        thematic = ThematicNLP(df_sentiment)
        final_df = thematic.process()

        # 3. Save Final Results
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final_df.to_csv(self.output_path, index=False)
        print(f"✅ Saved FINAL results with Themes to {self.output_path}")
        
        # 4. Show Summary
        print("\n--- Theme Distribution ---")
        print(final_df["identified_theme"].value_counts())
        
        return final_df


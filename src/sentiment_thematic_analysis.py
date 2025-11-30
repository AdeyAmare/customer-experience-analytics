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

nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

tqdm.pandas()

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
    "Mobile App & Digital": ["app", "login", "update", "interface", "screen", "feature", "digital", "mobile", "phone", "crash", "biometric", "ui", "ux"],
    "Customer Service": ["service", "staff", "manager", "support", "wait", "queue", "call", "agent", "teller", "rude", "polite", "help", "cs"],
    "Transactions & Fees": ["money", "transfer", "transaction", "fee", "charge", "balance", "deposit", "withdrawal", "payment", "rate", "limit"],
    "Security & Fraud": ["fraud", "scam", "security", "alert", "hack", "safe", "verification", "otp", "locked", "blocked", "account"],
    "Cards & Accounts": ["card", "debit", "credit", "account", "opening", "closure", "limit", "atm", "branch"]
}

# --- SENTIMENT CLASS ---
class SentimentNLP:
    def __init__(self, input_path=None, output_path=None):
        self.input_path = input_path or DATA_PATHS["processed_reviews"]
        self.output_path = output_path or DATA_PATHS["sentiment_results"]
        self.df = None
        self.analyzer = SentimentIntensityAnalyzer()

    def load_data(self):
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
        self.df["review_text"] = self.df["review_text"].fillna("").astype(str)
        print(f"✅ Loaded {len(self.df)} reviews.")

    def _get_label(self, score):
        if score >= 0.05:
            return "POSITIVE"
        elif score <= -0.05:
            return "NEGATIVE"
        else:
            return "NEUTRAL"

    def _run_vader(self):
        print("-> Running VADER...")
        self.df["vader_score"] = self.df["review_text"].progress_apply(lambda t: self.analyzer.polarity_scores(t)["compound"])
        self.df["vader_label"] = self.df["vader_score"].apply(self._get_label)

    def _run_textblob(self):
        print("-> Running TextBlob...")
        self.df["textblob_score"] = self.df["review_text"].progress_apply(lambda t: TextBlob(t).sentiment.polarity if t else 0.0)
        self.df["textblob_label"] = self.df["textblob_score"].apply(self._get_label)

    def _run_distilbert(self):
        if SENTIMENT_MODEL is None:
            print("-> Skipping DistilBERT: Model not loaded.")
            self.df["distilbert_score"] = 0.0
            self.df["distilbert_label"] = "NEUTRAL"
            return
        print("-> Running DistilBERT...")
        def get_sentiment(text):
            if not text.strip():
                return {"label": "NEUTRAL", "score": 0.0}
            try:
                return SENTIMENT_MODEL(text, truncation=True)[0]
            except Exception:
                return {"label": "NEUTRAL", "score": 0.0}
        results = self.df["review_text"].progress_apply(get_sentiment)
        self.df["distilbert_label"] = results.apply(lambda x: x["label"])
        self.df["distilbert_score"] = results.apply(lambda x: x["score"])

    def run_analysis(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        print("Running all three Sentiment Models...")
        self._run_vader()
        self._run_textblob()
        self._run_distilbert()
        print("✅ Sentiment Analysis complete.")

    def aggregate_sentiment(self):
        print("\n--- Aggregated Sentiment (Mean Score) ---")
        score_cols = [c for c in self.df.columns if 'score' in c]
        if not score_cols:
            print("Skipping aggregation: Sentiment scores not available.")
            return
        agg_df = self.df.groupby(["bank_name", "rating"])[score_cols].mean().reset_index()
        
        print(agg_df)
    
    def get_dataframe(self):
        return self.df

# --- THEMATIC CLASS ---
class ThematicNLP:
    def __init__(self, df):
        self.df = df.copy()

    def preprocess_text_for_tfidf(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.text not in STOPWORDS and len(token.text) > 2]
        return " ".join(tokens)

    def extract_tfidf_keywords(self):
        print("Preprocessing text for TF-IDF...")
        
        self.df["preprocessed_text"] = self.df["review_text"].progress_apply(self.preprocess_text_for_tfidf)
        
        if self.df["preprocessed_text"].str.strip().eq('').all():
            print("⚠️ Not enough non-empty text for TF-IDF.")
            self.df["extracted_keywords"] = ""
            self.df.drop(columns=["preprocessed_text"], inplace=True)
            return
        
        print("Calculating TF-IDF scores...")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=2)
        tfidf_matrix = vectorizer.fit_transform(self.df["preprocessed_text"])
        feature_names = vectorizer.get_feature_names_out()
        
        def top_keywords(i, top_n=5):
            scores = tfidf_matrix[i].toarray().flatten()
            indices = scores.argsort()[-top_n:][::-1]
            return [feature_names[j] for j in indices if scores[j] > 0]
        self.df["extracted_keywords"] = [top_keywords(i) for i in tqdm(range(len(self.df)), desc="Extracting Keywords")]
        self.df.drop(columns=["preprocessed_text"], inplace=True)

    def assign_theme(self, keywords):
        terms = set()
        for kw in keywords:
            terms.add(kw)
            terms.update(kw.split())
        found = [theme for theme, kws in THEME_KEYWORDS.items() if any(t in kws for t in terms)]
        return ", ".join(sorted(set(found))) if found else "General / Unspecified"

    def process(self):
        print("Running Thematic Analysis...")
        self.extract_tfidf_keywords()
        self.df["identified_theme"] = self.df["extracted_keywords"].progress_apply(self.assign_theme)
        self.df.drop(columns=["extracted_keywords"], inplace=True)
        print("✅ Theme assignment complete.")
        return self.df

# --- PIPELINE CLASS ---
class BankReviewPipeline:
    def __init__(self, input_path=None, output_path=None):
        self.sentiment = SentimentNLP(input_path, output_path)
        self.output_path = output_path or DATA_PATHS["sentiment_results"]

    def run(self):
        print("\n--- Starting Bank Review NLP Pipeline ---")
        self.sentiment.load_data()
        self.sentiment.run_analysis()
        df_sentiment = self.sentiment.get_dataframe()
        self.sentiment.aggregate_sentiment()
        thematic = ThematicNLP(df_sentiment)
        final_df = thematic.process()
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final_df.to_csv(self.output_path, index=False)
        print(f"\n✅ Saved FINAL results to {self.output_path}")
        print("\n--- Theme Distribution ---")
        print(final_df["identified_theme"].value_counts())
        print("\n--- Pipeline Run Complete ---")
        return final_df


"""
Sentiment Analysis Script (VADER vs. TextBlob)

Features:
    * Sentiment scoring & Label classification
    * Comparison Dashboards:
        1. Model Diagnostics (Histograms, Heatmaps, Agreement, Boxplots)
        2. Bank Analysis (Average Scores, Proportional Distributions)
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk

# NLP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# -------------------------------
# CONFIG & SETUP
# -------------------------------
nltk.download("vader_lexicon", quiet=True)
sns.set_theme(style="whitegrid")
tqdm.pandas()

# Default Paths
try:
    from .utils.config import DATA_PATHS
except ImportError:
    class Config:
        DATA_PATHS = {
            "processed_reviews": "./data/processed/reviews_processed.csv",
            "sentiment_results": "./data/processed/reviews_with_sentiment_textblob.csv",
        }
    DATA_PATHS = Config.DATA_PATHS

class SentimentNLP:
    """Pipeline for computing and visualizing VADER and TextBlob sentiment."""

    def __init__(self, input_path=None, output_path=None):
        self.input_path = input_path or DATA_PATHS["processed_reviews"]
        self.output_path = output_path or DATA_PATHS["sentiment_results"]
        self.df = None
        
        # Color Palettes
        self.sentiment_palette = {
            "POSITIVE": "#A8E6CF",  # Green
            "NEUTRAL": "#E1E1E1",   # Gray
            "NEGATIVE": "#FF8B94"   # Red
        }
   
        self.model_palette = ["skyblue", "#C39BD3"]

    # -------------------------------
    # 1. Data Loading
    # -------------------------------
    def load_data(self):
        """Loads and normalizes the reviews dataset."""
        print(f"Loading data from {self.input_path}...")

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"❌ Error: '{self.input_path}' not found.")

        self.df = pd.read_csv(self.input_path)
        
        # Ensure data integrity
        required_cols = ["review_text", "bank_name", "rating"]
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"Missing columns. Required: {required_cols}")

        self.df["review_text"] = self.df["review_text"].fillna("").astype(str)
        print(f"✅ Loaded {len(self.df)} reviews.")

    # -------------------------------
    # 2. NLP Processing
    # -------------------------------
    def run_analysis(self):
        """Runs both VADER and TextBlob analysis."""
        print("Running Sentiment Models...")
        
        # --- VADER ---
        analyzer = SentimentIntensityAnalyzer()
        self.df["vader_score"] = self.df["review_text"].progress_apply(
            lambda t: analyzer.polarity_scores(t)["compound"]
        )
        self.df["vader_label"] = self.df["vader_score"].apply(
            lambda s: "POSITIVE" if s >= 0.05 else ("NEGATIVE" if s <= -0.05 else "NEUTRAL")
        )

        # --- TextBlob ---
        # Using a wrapper to handle exceptions cleanly
        def get_tb(text):
            try:
                return TextBlob(text).sentiment.polarity
            except:
                return 0.0

        self.df["textblob_score"] = self.df["review_text"].progress_apply(get_tb)
        self.df["textblob_label"] = self.df["textblob_score"].apply(
            lambda s: "POSITIVE" if s >= 0.05 else ("NEGATIVE" if s <= -0.05 else "NEUTRAL")
        )
        
        print("✅ Analysis complete.")

    # -------------------------------
    # 3. Visualization Grids
    # -------------------------------
    def plot_model_comparison_grid(self):
        """Generates Grid 1: Model Diagnostics, Correlations, and Heatmaps."""
        print("Generating Model Comparison Grid...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Diagnostics & Comparison', fontsize=16, weight='bold')

        # [0,0] VADER Distribution
        sns.histplot(self.df["vader_score"], kde=True, color="skyblue", ax=axes[0, 0])
        axes[0, 0].set_title("VADER Score Distribution")

        # [0,1] TextBlob Distribution
        sns.histplot(self.df["textblob_score"], kde=True, color="purple", ax=axes[0, 1])
        axes[0, 1].set_title("TextBlob Score Distribution")

        # [0,2] Agreement Count
        self.df["agreement"] = self.df["vader_label"] == self.df["textblob_label"]
        sns.countplot(x="agreement", hue="agreement", data=self.df, palette="viridis", ax=axes[0, 2])
        axes[0, 2].set_title(f"Model Agreement (Match Rate: {self.df['agreement'].mean():.1%})")

        # [1,0] Heatmap: VADER vs Rating (Requested to be inside grid)
        conf_vader = pd.crosstab(self.df["rating"], self.df["vader_label"])
        sns.heatmap(conf_vader, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0])
        axes[1, 0].set_title("Heatmap: Rating vs VADER Label")

        # [1,1] Heatmap: TextBlob vs Rating
        conf_tb = pd.crosstab(self.df["rating"], self.df["textblob_label"])
        sns.heatmap(conf_tb, annot=True, fmt="d", cmap="Purples", ax=axes[1, 1])
        axes[1, 1].set_title("Heatmap: Rating vs TextBlob Label")

        # [1,2] Boxplot: Sentiment vs Rating
        melted_scores = self.df.melt(
            id_vars="rating", value_vars=["vader_score", "textblob_score"], 
            var_name="Model", value_name="Score"
        )
        sns.boxplot(x="rating", y="Score", hue="Model", data=melted_scores, 
                    palette=self.model_palette, ax=axes[1, 2])
        axes[1, 2].set_title("Score Distribution vs Star Rating")

        plt.tight_layout()
        plt.show()

    def plot_bank_sentiment_grid(self):
        """Generates Grid 2: Bank Averages and Proportional Distributions."""
        print("Generating Bank Sentiment Grid...")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Bank Sentiment Analysis', fontsize=16, weight='bold')

        # [0] Average Scores by Bank
        bank_scores = self.df.groupby("bank_name")[["vader_score", "textblob_score"]].mean().reset_index()
        melted_bank = bank_scores.melt(id_vars="bank_name", var_name="Model", value_name="Avg Score")
        
        sns.barplot(x="bank_name", y="Avg Score", hue="Model", data=melted_bank, 
                    palette=self.model_palette, ax=axes[0])
        axes[0].set_title("Average Sentiment Score by Bank")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

        # Helper for Proportions
        def plot_proportions(col_label, ax_idx, title):
            props = (self.df.groupby("bank_name")[col_label]
                     .value_counts(normalize=True).mul(100)
                     .rename("Proportion").reset_index())
            
            sns.barplot(x="bank_name", y="Proportion", hue=col_label, data=props,
                        palette=self.sentiment_palette, ax=axes[ax_idx])
            axes[ax_idx].set_title(title)
            axes[ax_idx].tick_params(axis='x', rotation=45)

        # [1] VADER Proportions (Requested New Feature)
        plot_proportions("vader_label", 1, "VADER: Sentiment Distribution")

        # [2] TextBlob Proportions
        plot_proportions("textblob_label", 2, "TextBlob: Sentiment Distribution")

        plt.tight_layout()
        plt.show()

    # -------------------------------
    # 4. Main Execution
    # -------------------------------
    def process(self):
        """Executes full pipeline."""
        self.load_data()
        self.run_analysis()

        # Metrics
        c_vader = self.df["rating"].corr(self.df["vader_score"])
        c_tb = self.df["rating"].corr(self.df["textblob_score"])
        print(f"\n📊 Correlations with Rating -> VADER: {c_vader:.3f} | TextBlob: {c_tb:.3f}")

        # Visualization
        self.plot_model_comparison_grid()
        self.plot_bank_sentiment_grid()

        # Save
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        print(f"✅ Saved results to {self.output_path}")

        # Return Summary
        return self.df.groupby(["bank_name", "rating"]).agg({
            "vader_score": "mean", "textblob_score": "mean", "review_text": "count"
        })

if __name__ == "__main__":
    try:
        analyzer = SentimentNLP()
        summary = analyzer.process()
        print("\n--- Summary Preview ---")
        print(summary.head())
    except Exception as e:
        print(f"\n❌ Error: {e}")
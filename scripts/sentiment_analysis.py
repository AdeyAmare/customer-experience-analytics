"""
Sentiment Analysis Script (Standalone)
- Tokenization, stop-word removal, lemmatization
- VADER sentiment analysis
- DistilBERT (SST-2) sentiment analysis (FIXED: uses raw text, neutral threshold)
- Comparison, Aggregation by bank and rating, Visualization
"""

import os
import sys
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# -------------------------------
# CONFIGURATION & SETUP
# -------------------------------

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Plotting style
sns.set(style="whitegrid")
tqdm.pandas()

# Fallback for data paths since external config import might break standalone execution
from .utils.config import DATA_PATHS

class SentimentNLP:
    def __init__(self, input_path=None, output_path=None):
        """
        Initialize the analyzer with paths and NLP tools.
        """
        self.input_path = input_path or DATA_PATHS['processed_reviews']
        self.output_path = output_path or DATA_PATHS['sentiment_results']
        self.df = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    # -------------------------------
    # 1. Load Data
    # -------------------------------
    def load_data(self):
        print(f"Loading data from {self.input_path}...")
        if os.path.exists(self.input_path):
            self.df = pd.read_csv(self.input_path)
            # Ensure required columns exist for demonstration (e.g., if using dummy data)
            if 'review_text' not in self.df.columns or 'bank_name' not in self.df.columns or 'rating' not in self.df.columns:
                 raise ValueError("DataFrame must contain 'review_text', 'bank_name', and 'rating' columns.")
            print(f"Loaded {len(self.df)} reviews.")
        else:
            raise FileNotFoundError(f"Input file not found at {self.input_path}")

    # -------------------------------
    # 2. Preprocessing
    # -------------------------------
    def preprocess_text(self, text):
        """
        Cleans text for VADER (Tokenization, stop-word removal, lemmatization).
        """
        if pd.isna(text) or str(text).strip() == '':
            return ''
        
        # Keep only letters, lowercase
        text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
        tokens = word_tokenize(text)
        
        # Lemmatize and remove stopwords
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)

    def run_preprocessing(self):
        print("Running VADER preprocessing...")
        # Create 'clean_text' for VADER analysis
        self.df['clean_text'] = self.df['review_text'].progress_apply(self.preprocess_text)
        
        # Filter out empty rows
        initial_count = len(self.df)
        self.df = self.df[self.df['clean_text'].str.strip() != '']
        print(f"Removed {initial_count - len(self.df)} empty reviews.")

    # -------------------------------
    # 3. Sentiment Analysis (VADER)
    # -------------------------------
    def run_vader(self):
        print("Running VADER sentiment analysis...")
        analyzer = SentimentIntensityAnalyzer()
        
        # Uses clean_text
        self.df['vader_score'] = self.df['clean_text'].apply(
            lambda t: analyzer.polarity_scores(str(t))['compound']
        )
        
        self.df['vader_label'] = self.df['vader_score'].apply(
            lambda s: 'POSITIVE' if s >= 0.05 else ('NEGATIVE' if s <= -0.05 else 'NEUTRAL')
        )
        print("VADER complete.")

    # -------------------------------
    # 4. Sentiment Analysis (DistilBERT - FIXED)
    # -------------------------------
    def run_bert(self):
        """
        Runs DistilBERT. KEY FIX: Uses 'review_text' (RAW) input.
        """
        print("Running DistilBERT sentiment analysis...")
        
        # device=-1 forces CPU (safer for beginner machines without GPU)
        classifier = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english", 
            device=-1
        )

        labels, scores = [], []

        # Use the RAW text ('review_text') for BERT 
        # because the model needs stopwords/context.
        for text in tqdm(self.df['review_text'], desc="BERT Processing"):
            safe_text = str(text)[:1000] # Safe character truncation
            
            try:
                # Truncation=True handles token limits (512 tokens)
                result = classifier(safe_text, truncation=True, max_length=512)[0]
                label = result['label']
                score = result['score']

                # LOGIC: SST-2 is binary. If confidence is low (< 0.75), classify as NEUTRAL.
                if score < 0.75:
                    labels.append('NEUTRAL')
                else:
                    labels.append(label)
                scores.append(score)
                    
            except Exception:
                # Fail gracefully if an unexpected error occurs
                labels.append('NEUTRAL')
                scores.append(0.0)

        self.df['bert_label'] = labels
        self.df['bert_score'] = scores
        print("DistilBERT complete.")

    # -------------------------------
    # 5. Comparison and Aggregation
    # -------------------------------
    def compare_sentiments(self):
        """Calculates and prints the agreement rate."""
        self.df['agreement'] = self.df['vader_label'] == self.df['bert_label']
        agreement_rate = self.df['agreement'].mean()
        print(f"Agreement between VADER and BERT: {agreement_rate:.2%}")

    def aggregate_by_bank_rating(self):
        """Aggregates mean scores and most frequent labels by bank and rating."""
        print("Aggregating results...")
        summary = self.df.groupby(['bank_name', 'rating']).agg({
            'bert_score': 'mean',
            'vader_score': 'mean',
            'bert_label': lambda x: x.value_counts().idxmax(), 
            'vader_label': lambda x: x.value_counts().idxmax(),
            'review_text': 'count'
        }).rename(columns={'review_text': 'count'}).reset_index()
        return summary

    # -------------------------------
    # 6. Visualizations
    # -------------------------------
    def plot_results(self):
        """Generates distribution and comparison plots."""
        
        # Distribution Plots
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df['vader_score'], bins=20, kde=True, color='skyblue')
        plt.title("VADER Sentiment Score Distribution")
        
        plt.subplot(1, 2, 2)
        sns.histplot(self.df['bert_score'], bins=20, kde=True, color='orange')
        plt.title("BERT Sentiment Score Distribution")
        plt.show()

        # Agreement and Average Scores per Bank
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Agreement Plot
        sns.countplot(x='agreement', data=self.df, ax=axes[0], palette=['salmon', 'lightgreen'])
        axes[0].set_title("VADER vs BERT Agreement")

        # Avg VADER Score per Bank
        self.df.groupby('bank_name')['vader_score'].mean().sort_values().plot(kind='bar', color='skyblue', ax=axes[1])
        axes[1].set_title("Average VADER Score per Bank")
        axes[1].set_ylabel("VADER Score")
        axes[1].tick_params(axis='x', rotation=45)

        # Avg BERT Score per Bank
        self.df.groupby('bank_name')['bert_score'].mean().sort_values().plot(kind='bar', color='orange', ax=axes[2])
        axes[2].set_title("Average BERT Score per Bank")
        axes[2].set_ylabel("BERT Score")
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    # -------------------------------
    # 7. Save & Execute
    # -------------------------------
    def save_results(self):
        """Saves final sentiment results to CSV."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Select key sentiment columns to save
        cols = ['review_id', 'bank_name', 'rating', 'review_text', 'clean_text', 
                'vader_label', 'vader_score', 
                'bert_label', 'bert_score']
        
        # Only save columns that actually exist in the dataframe
        save_cols = [c for c in cols if c in self.df.columns]
        
        self.df[save_cols].to_csv(self.output_path, index=False)
        print(f"Sentiment results saved to {self.output_path}")

    def process(self):
        """Full pipeline execution for sentiment analysis."""
        self.load_data()
        self.run_preprocessing()
        self.run_vader()
        self.run_bert()
        self.compare_sentiments()
        summary = self.aggregate_by_bank_rating()
        self.plot_results()
        self.save_results()
        print("Sentiment analysis complete.")
        return self.df, summary

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    analyzer = SentimentNLP()
    
    # Run only if the input file is available
    if os.path.exists(analyzer.input_path):
        df, summary = analyzer.process()
        print("\n--- Summary of Mean Sentiment Scores per Bank/Rating ---")
        print(summary.head(10))
    else:
        print("ERROR: Input file 'processed_reviews.csv' not found. Please ensure your data is loaded.")
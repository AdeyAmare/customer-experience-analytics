"""
Data Preprocessing Script
Task: Clean and preprocess scraped reviews data

This script provides a complete preprocessing pipeline for Google Play
and other scraped reviews. It handles:
- Missing values
- Date normalization
- Text cleaning
- Duplicate removal
- Rating validation
- Summary reporting
"""

import sys
import os
import re
import pandas as pd
from datetime import datetime
from utils.config import DATA_PATHS


class ReviewPreprocessor:
    """Preprocessor for review datasets.

    Provides a complete pipeline to clean, validate, and save reviews
    collected from multiple sources.
    """

    def __init__(self, input_path: str = None, output_path: str = None):
        """Initialize the preprocessor with input/output paths.

        Args:
            input_path (str, optional): Path to raw reviews CSV.
                Defaults to DATA_PATHS['raw_reviews'].
            output_path (str, optional): Path to save processed CSV.
                Defaults to DATA_PATHS['processed_reviews'].
        """
        self.input_path = input_path or DATA_PATHS['raw_reviews']
        self.output_path = output_path or DATA_PATHS['processed_reviews']
        self.df: pd.DataFrame | None = None
        self.stats: dict = {}

    def load_data(self) -> bool:
        """Load raw review data from CSV into a DataFrame.

        Returns:
            bool: True if loading succeeded, False otherwise.
        """
        print("Loading raw review data...")
        try:
            self.df = pd.read_csv(self.input_path)
            self.stats['original_count'] = len(self.df)
            print(f"Loaded {len(self.df)} reviews")
            return True
        except FileNotFoundError:
            print(f"ERROR: File not found: {self.input_path}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load data: {e}")
            return False

    def check_missing_data(self) -> None:
        """Check for missing values and report critical column gaps."""
        print("\n[1/7] Checking for missing data...")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        for col in missing.index:
            if missing[col] > 0:
                print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
        self.stats['missing_before'] = missing.to_dict()

        critical_cols = ['review_text', 'rating', 'bank_name']
        missing_critical = self.df[critical_cols].isnull().sum()
        if missing_critical.sum() > 0:
            print("\nWARNING: Missing critical data:")
            print(missing_critical[missing_critical > 0])

    def handle_missing_values(self) -> None:
        """Handle missing values for critical and optional columns."""
        print("\n[2/7] Handling missing values...")
        critical_cols = ['review_text', 'rating', 'bank_name']
        before_count = len(self.df)
        self.df.dropna(subset=critical_cols, inplace=True)
        removed = before_count - len(self.df)
        print(f"Removed {removed} rows with missing critical values")

        self.df['user_name'] = self.df.get('user_name', pd.Series()).fillna('Anonymous')
        self.df['thumbs_up'] = self.df.get('thumbs_up', pd.Series()).fillna(0)
        self.df['reply_content'] = self.df.get('reply_content', pd.Series()).fillna('')

        self.stats['rows_removed_missing'] = removed
        self.stats['count_after_missing'] = len(self.df)

    def normalize_dates(self) -> None:
        """Normalize 'review_date' to datetime and extract year/month columns."""
        print("\n[3/7] Normalizing dates...")
        try:
            self.df['review_date'] = pd.to_datetime(self.df['review_date'], errors='coerce')
            self.df['review_date'] = self.df['review_date'].dt.date
            self.df['review_year'] = pd.to_datetime(self.df['review_date']).dt.year
            self.df['review_month'] = pd.to_datetime(self.df['review_date']).dt.month
            print(f"Date range: {self.df['review_date'].min()} to {self.df['review_date'].max()}")
        except Exception as e:
            print(f"WARNING: Failed to normalize dates: {e}")

    def clean_text(self) -> None:
        """Clean review text by removing non-English characters and short reviews."""
        print("\n[4/7] Cleaning text...")

        def clean(text: str) -> str:
            if pd.isna(text) or text == '':
                return ''
            text = re.sub(r'[^a-zA-Z ]+', ' ', str(text))
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        self.df['review_text'] = self.df['review_text'].apply(clean)
        before_count = len(self.df)
        self.df = self.df[self.df['review_text'].str.len() >= 3]
        removed = before_count - len(self.df)
        if removed > 0:
            print(f"Removed {removed} short or non-English reviews")
        self.df['text_length'] = self.df['review_text'].str.len()
        self.stats['empty_reviews_removed'] = removed
        self.stats['count_after_cleaning'] = len(self.df)

    def remove_duplicates(self) -> None:
        """Remove duplicate reviews based on 'review_text' and 'user_name'."""
        print("\n[5/7] Removing duplicates...")
        before_count = len(self.df)
        self.df.drop_duplicates(subset=['review_text', 'user_name'], inplace=True)
        removed = before_count - len(self.df)
        print(f"Removed {removed} duplicate reviews")
        self.stats['duplicates_removed'] = removed
        self.stats['count_after_duplicates'] = len(self.df)

    def validate_ratings(self) -> None:
        """Ensure ratings are between 1 and 5, removing invalid entries."""
        print("\n[6/7] Validating ratings...")
        invalid = self.df[(self.df['rating'] < 1) | (self.df['rating'] > 5)]
        if not invalid.empty:
            print(f"WARNING: Found {len(invalid)} invalid ratings")
            self.df = self.df[(self.df['rating'] >= 1) & (self.df['rating'] <= 5)]
        else:
            print("All ratings are valid (1-5)")
        self.stats['invalid_ratings_removed'] = len(invalid)

    def prepare_final_output(self) -> None:
        """Reorder columns, sort data, and reset index before saving."""
        print("\n[7/7] Preparing final output...")
        output_cols = [
            'review_id', 'review_text', 'rating', 'review_date',
            'review_year', 'review_month', 'bank_code', 'bank_name',
            'user_name', 'thumbs_up', 'text_length', 'source'
        ]
        output_cols = [col for col in output_cols if col in self.df.columns]
        self.df = self.df[output_cols]
        self.df.sort_values(['bank_code', 'review_date'], ascending=[True, False], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print(f"Final dataset: {len(self.df)} reviews")

    def save_data(self) -> bool:
        """Save processed reviews to CSV.

        Returns:
            bool: True if saving succeeded, False otherwise.
        """
        print("\nSaving processed data...")
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.df.to_csv(self.output_path, index=False)
            print(f"Data saved to {self.output_path}")
            self.stats['final_count'] = len(self.df)
            return True
        except Exception as e:
            print(f"ERROR: Failed to save data: {e}")
            return False

    def generate_report(self) -> None:
        """Print a detailed preprocessing report with statistics."""
        print("\n" + "=" * 60)
        print("PREPROCESSING REPORT")
        print("=" * 60)
        print(f"Original records: {self.stats.get('original_count', 0)}")
        print(f"Rows removed (missing critical data): {self.stats.get('rows_removed_missing', 0)}")
        print(f"Empty reviews removed: {self.stats.get('empty_reviews_removed', 0)}")
        print(f"Duplicate reviews removed: {self.stats.get('duplicates_removed', 0)}")
        print(f"Invalid ratings removed: {self.stats.get('invalid_ratings_removed', 0)}")
        print(f"Final records: {self.stats.get('final_count', 0)}")

        if self.stats.get('original_count', 0) > 0:
            retention_rate = (self.stats.get('final_count', 0) / self.stats.get('original_count', 1)) * 100
            print(f"Data retention rate: {retention_rate:.2f}%")

        if self.df is not None:
            print("\nReviews per bank:")
            for bank, count in self.df['bank_name'].value_counts().items():
                print(f"  {bank}: {count}")
            print("\nRating distribution:")
            for rating, count in self.df['rating'].value_counts().sort_index(ascending=False).items():
                pct = (count / len(self.df)) * 100
                print(f"  {'⭐' * int(rating)}: {count} ({pct:.1f}%)")
            print(f"\nDate range: {self.df['review_date'].min()} to {self.df['review_date'].max()}")
            print(f"\nText length (characters): Avg={self.df['text_length'].mean():.0f}, "
                  f"Median={self.df['text_length'].median():.0f}, Min={self.df['text_length'].min()}, "
                  f"Max={self.df['text_length'].max()}")

    def process(self) -> bool:
        """Run the full preprocessing pipeline from loading to report generation.

        Returns:
            bool: True if preprocessing completed successfully, False otherwise.
        """
        print("=" * 60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("=" * 60)

        if not self.load_data():
            return False

        self.check_missing_data()
        self.handle_missing_values()
        self.normalize_dates()
        self.clean_text()
        self.remove_duplicates()
        self.validate_ratings()
        self.prepare_final_output()

        if self.save_data():
            self.generate_report()
            return True
        return False

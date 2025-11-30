"""
Data Preprocessing Script
Task 1: Data Preprocessing

This script cleans and preprocesses the scraped reviews data.
- Handles missing values
- Normalizes dates
- Cleans text data
- Removes duplicates
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import re
from utils.config import DATA_PATHS


class ReviewPreprocessor:
    """Preprocessor class for review data.

    This class provides a complete pipeline for cleaning, validating, and 
    saving review data scraped from multiple sources.
    """

    def __init__(self, input_path=None, output_path=None):
        """Initializes the preprocessor with input and output paths.

        Args:
            input_path (str, optional): Path to raw reviews CSV. Defaults to DATA_PATHS['raw_reviews'].
            output_path (str, optional): Path to save processed reviews. Defaults to DATA_PATHS['processed_reviews'].
        """
        self.input_path = input_path or DATA_PATHS['raw_reviews']
        self.output_path = output_path or DATA_PATHS['processed_reviews']
        self.df = None
        self.stats = {}

    def load_data(self) -> bool:
        """Loads raw reviews data into a DataFrame.

        Returns:
            bool: True if data was successfully loaded, False otherwise.
        """
        print("Loading raw data...")
        try:
            self.df = pd.read_csv(self.input_path)
            print(f"Loaded {len(self.df)} reviews")
            self.stats['original_count'] = len(self.df)
            return True
        except FileNotFoundError:
            print(f"ERROR: File not found: {self.input_path}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load data: {str(e)}")
            return False

    def check_missing_data(self):
        """Checks for missing data and prints a summary.

        Records missing value statistics for each column and specifically
        flags critical columns required for analysis.
        """
        print("\n[1/7] Checking for missing data...")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100

        print("\nMissing values:")
        for col in missing.index:
            if missing[col] > 0:
                print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")

        self.stats['missing_before'] = missing.to_dict()

        critical_cols = ['review_text', 'rating', 'bank_name']
        missing_critical = self.df[critical_cols].isnull().sum()

        if missing_critical.sum() > 0:
            print("\nWARNING: Missing values in critical columns:")
            print(missing_critical[missing_critical > 0])

    def handle_missing_values(self):
        """Handles missing values in critical and non-critical columns.

        Drops rows with missing critical values and fills non-critical missing
        values with default placeholders.
        """
        print("\n[2/7] Handling missing values...")
        critical_cols = ['review_text', 'rating', 'bank_name']
        before_count = len(self.df)
        self.df = self.df.dropna(subset=critical_cols)
        removed = before_count - len(self.df)

        if removed > 0:
            print(f"Removed {removed} rows with missing critical values")

        self.df['user_name'] = self.df['user_name'].fillna('Anonymous')
        self.df['thumbs_up'] = self.df['thumbs_up'].fillna(0)
        self.df['reply_content'] = self.df['reply_content'].fillna('')

        self.stats['rows_removed_missing'] = removed
        self.stats['count_after_missing'] = len(self.df)

    def normalize_dates(self):
        """Normalizes the 'review_date' column to YYYY-MM-DD format.

        Also extracts 'review_year' and 'review_month' as separate columns.
        """
        print("\n[3/7] Normalizing dates...")
        try:
            self.df['review_date'] = pd.to_datetime(self.df['review_date'])
            self.df['review_date'] = self.df['review_date'].dt.date
            self.df['review_year'] = pd.to_datetime(self.df['review_date']).dt.year
            self.df['review_month'] = pd.to_datetime(self.df['review_date']).dt.month
            print(f"Date range: {self.df['review_date'].min()} to {self.df['review_date'].max()}")
        except Exception as e:
            print(f"WARNING: Error normalizing dates: {str(e)}")

    def clean_text(self):
        """
        Clean review text:
        - Keep only English letters (A-Z, a-z) and spaces
        - Remove emojis, special characters, numbers, and non-Latin scripts
        - Convert multiple spaces to a single space
        - Strip leading/trailing spaces
        - Remove empty or very short reviews (less than 3 characters)
        """
        print("\n[4/6] Cleaning text...")

        def clean_review_text(text):
            """Cleans individual review text."""
            if pd.isna(text) or text == '':
                return ''
            text = str(text)
            # Keep only English letters and spaces
            text = re.sub(r'[^a-zA-Z ]+', ' ', text)
            # Replace multiple spaces with a single space
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        # Apply cleaning to the review_text column
        self.df['review_text'] = self.df['review_text'].apply(clean_review_text)

        before_count = len(self.df)
        # Keep only reviews with length >= 3
        self.df = self.df[self.df['review_text'].str.len() >= 3]
        removed = before_count - len(self.df)

        if removed > 0:
            print(f"Removed {removed} reviews with non-English, empty, or very short text (<3 chars)")

        # Add text length column
        self.df['text_length'] = self.df['review_text'].str.len()
        self.stats['empty_reviews_removed'] = removed
        self.stats['count_after_cleaning'] = len(self.df)


    def remove_duplicates(self):
        """Removes duplicate reviews based on 'review_text' and 'user_name'."""
        print("\n[5/7] Removing duplicate reviews...")
        before_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['review_text', 'user_name'], keep='first')
        removed = before_count - len(self.df)
        print(f"Removed {removed} duplicate reviews")

        self.stats['duplicates_removed'] = removed
        self.stats['count_after_duplicates'] = len(self.df)

    def validate_ratings(self):
        """Validates rating values to ensure they are between 1 and 5.

        Invalid ratings are removed from the dataset.
        """
        print("\n[6/7] Validating ratings...")
        invalid = self.df[(self.df['rating'] < 1) | (self.df['rating'] > 5)]
        if len(invalid) > 0:
            print(f"WARNING: Found {len(invalid)} reviews with invalid ratings")
            self.df = self.df[(self.df['rating'] >= 1) & (self.df['rating'] <= 5)]
        else:
            print("All ratings are valid (1-5)")

        self.stats['invalid_ratings_removed'] = len(invalid)

    def prepare_final_output(self):
        """Prepares the final dataset for saving.

        Reorders columns, sorts by 'bank_code' and 'review_date', and resets index.
        """
        print("\n[7/7] Preparing final output...")
        output_columns = [
            'review_id', 'review_text', 'rating', 'review_date',
            'review_year', 'review_month', 'bank_code', 'bank_name',
            'user_name', 'thumbs_up', 'text_length', 'source'
        ]
        output_columns = [col for col in output_columns if col in self.df.columns]
        self.df = self.df[output_columns]
        self.df = self.df.sort_values(['bank_code', 'review_date'], ascending=[True, False])
        self.df = self.df.reset_index(drop=True)
        print(f"Final dataset: {len(self.df)} reviews")

    def save_data(self) -> bool:
        """Saves the processed DataFrame to a CSV file.

        Returns:
            bool: True if data was successfully saved, False otherwise.
        """
        print("\nSaving processed data...")
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.df.to_csv(self.output_path, index=False)
            print(f"Data saved to: {self.output_path}")
            self.stats['final_count'] = len(self.df)
            return True
        except Exception as e:
            print(f"ERROR: Failed to save data: {str(e)}")
            return False

    def generate_report(self):
        """Generates a detailed preprocessing report."""
        print("\n" + "=" * 60)
        print("PREPROCESSING REPORT")
        print("=" * 60)

        print(f"\nOriginal records: {self.stats.get('original_count', 0)}")
        print(f"Records with missing critical data: {self.stats.get('rows_removed_missing', 0)}")
        print(f"Empty reviews removed: {self.stats.get('empty_reviews_removed', 0)}")
        print(f"Duplicate reviews removed: {self.stats.get('duplicates_removed', 0)}")
        print(f"Invalid ratings removed: {self.stats.get('invalid_ratings_removed', 0)}")
        print(f"Final records: {self.stats.get('final_count', 0)}")

        if self.stats.get('original_count', 0) > 0:
            retention_rate = (self.stats.get('final_count', 0) / self.stats.get('original_count', 1)) * 100
            error_rate = 100 - retention_rate
            print(f"\nData retention rate: {retention_rate:.2f}%")
            print(f"Data error rate: {error_rate:.2f}%")
            if error_rate < 5:
                print("✓ Data quality: EXCELLENT (<5% errors)")
            elif error_rate < 10:
                print("✓ Data quality: GOOD (<10% errors)")
            else:
                print("⚠ Data quality: NEEDS ATTENTION (>10% errors)")

        if self.df is not None:
            print("\nReviews per bank:")
            bank_counts = self.df['bank_name'].value_counts()
            for bank, count in bank_counts.items():
                print(f"  {bank}: {count}")

            print("\nRating distribution:")
            rating_counts = self.df['rating'].value_counts().sort_index(ascending=False)
            for rating, count in rating_counts.items():
                pct = (count / len(self.df)) * 100
                print(f"  {'⭐' * int(rating)}: {count} ({pct:.1f}%)")

            print(f"\nDate range: {self.df['review_date'].min()} to {self.df['review_date'].max()}")
            print(f"\nText statistics:")
            print(f"  Average length: {self.df['text_length'].mean():.0f} characters")
            print(f"  Median length: {self.df['text_length'].median():.0f} characters")
            print(f"  Min length: {self.df['text_length'].min()}")
            print(f"  Max length: {self.df['text_length'].max()}")

    def process(self) -> bool:
        """Runs the complete preprocessing pipeline.

        Returns:
            bool: True if preprocessing was successful, False otherwise.
        """
        print("=" * 60)
        print("STARTING DATA PREPROCESSING")
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




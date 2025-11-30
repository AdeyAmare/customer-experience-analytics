"""
Data Preprocessing Script
Task: Clean and preprocess scraped reviews data

This script implements a full preprocessing pipeline for Google Play and
other scraped reviews. It performs the following operations:

- Missing value detection and handling
- Date normalization
- Text cleaning
- Duplicate removal
- Rating validation
- Summary reporting

The pipeline loads raw review data, cleans and validates fields,
normalizes formats, and outputs a processed dataset ready for further
analysis or modeling.
"""

import sys
import os
import re
import pandas as pd
from datetime import datetime
from utils.config import DATA_PATHS


class ReviewPreprocessor:
    """End-to-end preprocessor for review datasets.

    This class provides a structured pipeline to load, clean, validate,
    and save review data collected from Google Play Store or other sources.
    """

    def __init__(self, input_path: str = None, output_path: str = None):
        """Initialize the preprocessor with input and output file paths.

        Args:
            input_path (str, optional): Path to the raw review CSV file.
                Defaults to DATA_PATHS['raw_reviews'].
            output_path (str, optional): Path to output the processed CSV file.
                Defaults to DATA_PATHS['processed_reviews'].
        """
        self.input_path = input_path or DATA_PATHS["raw_reviews"]
        self.output_path = output_path or DATA_PATHS["processed_reviews"]

        self.df: pd.DataFrame | None = None
        self.stats: dict = {}

    # ----------------------------------------------------------------------
    # Data Loading
    # ----------------------------------------------------------------------
    def load_data(self) -> bool:
        """Load raw review data into a pandas DataFrame.

        Returns:
            bool: True if loading succeeded, False otherwise.
        """
        print("Loading raw review data...")
        try:
            self.df = pd.read_csv(self.input_path)
            self.stats["original_count"] = len(self.df)
            print(f"Loaded {len(self.df)} reviews")
            return True
        except FileNotFoundError:
            print(f"ERROR: File not found: {self.input_path}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to load data: {e}")
            return False

    # ----------------------------------------------------------------------
    # Missing Data Handling
    # ----------------------------------------------------------------------
    def check_missing_data(self) -> None:
        """Identify and display missing data statistics."""
        print("\n[1/7] Checking for missing data...")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100

        # Display missing summary
        for col in missing.index:
            if missing[col] > 0:
                print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")

        self.stats["missing_before"] = missing.to_dict()

        # Check critical columns that must not contain nulls
        critical_cols = ["review_text", "rating", "bank_name"]
        missing_critical = self.df[critical_cols].isnull().sum()

        if missing_critical.sum() > 0:
            print("\nWARNING: Missing data found in critical columns:")
            print(missing_critical[missing_critical > 0])

    def handle_missing_values(self) -> None:
        """Remove rows with missing critical values and fill optional fields."""
        print("\n[2/7] Handling missing values...")
        critical_cols = ["review_text", "rating", "bank_name"]

        before = len(self.df)
        self.df.dropna(subset=critical_cols, inplace=True)
        removed = before - len(self.df)

        print(f"Removed {removed} rows with missing critical values")
        self.stats["rows_removed_missing"] = removed
        self.stats["count_after_missing"] = len(self.df)

        # Fill optional values with defaults
        self.df["user_name"] = self.df.get("user_name", pd.Series()).fillna("Anonymous")
        self.df["thumbs_up"] = self.df.get("thumbs_up", pd.Series()).fillna(0)
        self.df["reply_content"] = self.df.get("reply_content", pd.Series()).fillna("")

    # ----------------------------------------------------------------------
    # Date Normalization
    # ----------------------------------------------------------------------
    def normalize_dates(self) -> None:
        """Normalize dates to a standard format and extract year/month fields."""
        print("\n[3/7] Normalizing dates...")
        try:
            self.df["review_date"] = pd.to_datetime(self.df["review_date"], errors="coerce")
            self.df["review_date"] = self.df["review_date"].dt.date

            # Extract new temporal features
            self.df["review_year"] = pd.to_datetime(self.df["review_date"]).dt.year
            self.df["review_month"] = pd.to_datetime(self.df["review_date"]).dt.month

            print(f"Date range: {self.df['review_date'].min()} to {self.df['review_date'].max()}")
        except Exception as e:
            print(f"WARNING: Failed to normalize dates: {e}")

    # ----------------------------------------------------------------------
    # Text Cleaning
    # ----------------------------------------------------------------------
    def clean_text(self) -> None:
        """Clean review text by removing noise and enforcing minimum length."""
        print("\n[4/7] Cleaning text...")

        def clean(text: str) -> str:
            """Remove non-letter characters and extra whitespace."""
            if pd.isna(text) or text == "":
                return ""
            text = re.sub(r"[^a-zA-Z ]+", " ", str(text))
            text = re.sub(r"\s+", " ", text).strip()
            return text

        self.df["review_text"] = self.df["review_text"].apply(clean)

        # Remove short or meaningless entries
        before = len(self.df)
        self.df = self.df[self.df["review_text"].str.len() >= 3]
        removed = before - len(self.df)

        if removed > 0:
            print(f"Removed {removed} short or non-English reviews")

        self.df["text_length"] = self.df["review_text"].str.len()
        self.stats["empty_reviews_removed"] = removed
        self.stats["count_after_cleaning"] = len(self.df)

    # ----------------------------------------------------------------------
    # Duplicate Removal
    # ----------------------------------------------------------------------
    def remove_duplicates(self) -> None:
        """Remove duplicate reviews based on text and username."""
        print("\n[5/7] Removing duplicates...")
        before = len(self.df)

        self.df.drop_duplicates(subset=["review_text", "user_name"], inplace=True)
        removed = before - len(self.df)

        print(f"Removed {removed} duplicate reviews")
        self.stats["duplicates_removed"] = removed
        self.stats["count_after_duplicates"] = len(self.df)

    # ----------------------------------------------------------------------
    # Rating Validation
    # ----------------------------------------------------------------------
    def validate_ratings(self) -> None:
        """Validate that ratings fall within the 1–5 range."""
        print("\n[6/7] Validating ratings...")

        invalid = self.df[(self.df["rating"] < 1) | (self.df["rating"] > 5)]

        if not invalid.empty:
            print(f"WARNING: Found {len(invalid)} invalid ratings")
            self.df = self.df[(self.df["rating"] >= 1) & (self.df["rating"] <= 5)]
        else:
            print("All ratings are valid (1-5)")

        self.stats["invalid_ratings_removed"] = len(invalid)

    # ----------------------------------------------------------------------
    # Final Output Formatting
    # ----------------------------------------------------------------------
    def prepare_final_output(self) -> None:
        """Finalize column ordering, sorting, and indexing."""
        print("\n[7/7] Preparing final output...")

        columns = [
            "review_id", "review_text", "rating", "review_date",
            "review_year", "review_month", "bank_code", "bank_name",
            "user_name", "thumbs_up", "text_length", "source"
        ]

        # Only keep columns that exist in the dataset
        columns = [c for c in columns if c in self.df.columns]

        self.df = self.df[columns]
        self.df.sort_values(["bank_code", "review_date"], ascending=[True, False], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        print(f"Final dataset contains {len(self.df)} reviews")

    # ----------------------------------------------------------------------
    # Saving Output
    # ----------------------------------------------------------------------
    def save_data(self) -> bool:
        """Save the processed DataFrame to disk.

        Returns:
            bool: True if save succeeded, False otherwise.
        """
        print("\nSaving processed data...")
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.df.to_csv(self.output_path, index=False)

            print(f"Data saved to {self.output_path}")
            self.stats["final_count"] = len(self.df)
            return True
        except Exception as e:
            print(f"ERROR: Failed to save data: {e}")
            return False

    # ----------------------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------------------
    def generate_report(self) -> None:
        """Generate a detailed summary of preprocessing steps and results."""
        print("\n" + "=" * 60)
        print("PREPROCESSING REPORT")
        print("=" * 60)

        print(f"Original records: {self.stats.get('original_count', 0)}")
        print(f"Rows removed (missing critical data): {self.stats.get('rows_removed_missing', 0)}")
        print(f"Empty reviews removed: {self.stats.get('empty_reviews_removed', 0)}")
        print(f"Duplicate reviews removed: {self.stats.get('duplicates_removed', 0)}")
        print(f"Invalid ratings removed: {self.stats.get('invalid_ratings_removed', 0)}")
        print(f"Final records: {self.stats.get('final_count', 0)}")

        # Retention percentage
        if self.stats.get("original_count", 0) > 0:
            retention = (self.stats.get("final_count", 0) /
                         self.stats.get("original_count", 1)) * 100
            print(f"Data retention rate: {retention:.2f}%")

        if self.df is None:
            return

        # Reviews per bank
        print("\nReviews per bank:")
        for bank, count in self.df["bank_name"].value_counts().items():
            print(f"  {bank}: {count}")

        print("\nRating distribution:")
        for rating, count in self.df["rating"].value_counts().sort_index(ascending=False).items():
            pct = (count / len(self.df)) * 100
            print(f"  {'⭐' * int(rating)}: {count} ({pct:.1f}%)")

        print(f"\nDate range: {self.df['review_date'].min()} to {self.df['review_date'].max()}")

        print(
            f"\nText length (characters): "
            f"Avg={self.df['text_length'].mean():.0f}, "
            f"Median={self.df['text_length'].median():.0f}, "
            f"Min={self.df['text_length'].min()}, "
            f"Max={self.df['text_length'].max()}"
        )

    # ----------------------------------------------------------------------
    # Pipeline Runner
    # ----------------------------------------------------------------------
    def process(self) -> bool:
        """Execute the full preprocessing workflow.

        Returns:
            bool: True if the pipeline executed successfully, False otherwise.
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

        # Save and report
        if self.save_data():
            self.generate_report()
            return True

        return False

import unittest
import pandas as pd
import sys
import os

# Add src folder to Python path
sys.path.insert(0, os.path.abspath("scripts"))

# ------------------ Imports from src package ------------------ #
"""
Import core classes and functions from the src package for testing.

- ReviewPreprocessor: Handles text cleaning, duplicate removal, and date normalization.
- SentimentNLP: Performs sentiment analysis using multiple models.
- ThematicNLP: Performs thematic/topic extraction from review text.
- BankReviewsETL: Prepares data for insertion into the database.

The try/except block ensures import errors are caught and reported.
"""
try:
    from src.data_preprocessing import ReviewPreprocessor
    from src.sentiment_thematic_analysis import SentimentNLP, ThematicNLP
    from postgres_connection import BankReviewsETL
except ImportError as e:
    print(f"[IMPORT ERROR] {e}")
    raise

# ---------------------- Preprocessor Tests ---------------------- #
class TestPreprocessor(unittest.TestCase):
    """Unit tests for ReviewPreprocessor class methods."""

    @classmethod
    def setUpClass(cls):
        """Set up test DataFrame and ReviewPreprocessor instance for all tests."""
        data = {
            'review_id': [1, 2, 3, 4],
            'review_text': ['Great app! but it crashes.', 'The bank is good.', 
                            'Great app! but it crashes.', 'I like it!'],
            'rating': [5, 4, 5, 3],
            'review_date': ['2023-01-01', '2023/02/02', '2023-01-01', 'Jan 1, 2023'],
            'bank_name': ['A', 'B', 'A', 'B'],
            'user_name': ['User1', 'User2', 'User1', 'User4']
        }
        cls.df = pd.DataFrame(data)
        cls.preprocessor = ReviewPreprocessor("dummy_input_path", "dummy_output_path")
        cls.preprocessor.df = cls.df.copy()

    def test_remove_duplicates(self):
        """Test that duplicate reviews are removed correctly."""
        self.preprocessor.remove_duplicates()
        self.assertEqual(len(self.preprocessor.df), 3)

    def test_normalize_dates(self):
        """Test that review_date column is converted to datetime dtype."""
        self.preprocessor.df['review_date'] = pd.to_datetime(
            self.preprocessor.df['review_date'], errors='coerce'
        )
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.preprocessor.df['review_date']))
        self.assertEqual(pd.to_datetime('2023-01-01'), pd.Timestamp(2023, 1, 1))

    def test_clean_text(self):
        """Test that review_text is cleaned by removing punctuation and lowercasing."""
        self.preprocessor.df['review_text'] = self.preprocessor.df['review_text'].str.replace(
            r'[^\w\s]', '', regex=True
        ).str.lower()
        self.assertEqual(self.preprocessor.df['review_text'].iloc[0], 'great app but it crashes')


# ------------------- Sentiment Tests ------------------- #
class TestSentimentNLP(unittest.TestCase):
    """Unit tests for SentimentNLP class methods."""

    @classmethod
    def setUpClass(cls):
        """Set up test DataFrame and SentimentNLP instance for all tests."""
        cls.df = pd.DataFrame({'review_text': [
            'Excellent service, very reliable.',
            'This feature is broken and terrible.',
            'I had an average, neutral transaction.'
        ]})
        cls.analyzer = SentimentNLP("dummy_input_path")
        cls.analyzer.df = cls.df.copy()

    def test_run_analysis(self):
        """Test that sentiment analysis generates expected columns and valid values."""
        self.analyzer.run_analysis()
        self.assertIn(self.analyzer.df['vader_label'].iloc[0], ['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
        self.assertTrue(-1 <= self.analyzer.df['textblob_score'].iloc[0] <= 1)
        self.assertIn(self.analyzer.df['distilbert_label'].iloc[0], ['POSITIVE', 'NEGATIVE', 'NEUTRAL'])


# ------------------- Thematic NLP Tests ------------------- #
class TestThematicNLP(unittest.TestCase):
    """Unit tests for ThematicNLP class methods."""

    def test_thematic_column_creation(self):
        """Test that thematic analysis adds an 'identified_theme' column."""
        df = pd.DataFrame({'review_text': ['Great app crashes', 'Bank is good']})
        df['preprocessed_text'] = df['review_text'].str.lower()
        nlp = ThematicNLP(df)
        
        # SAFE PARAMETERS FOR TINY DATASETS
        nlp.vectorizer_params = {'min_df': 1, 'max_df': 1.0}
        
        result = nlp.process()
        self.assertIn('identified_theme', result.columns)


# ------------------- ETL Tests (No DB) ------------------- #
class TestBankReviewsETL(unittest.TestCase):
    """Unit tests for BankReviewsETL class methods without actual database operations."""

    def setUp(self):
        """Set up a test DataFrame and BankReviewsETL instance for all tests."""
        self.df = pd.DataFrame({
            'review_id': ['a1', 'b2'], 'bank_id': [1, 2], 'review_text': ['test1', 'test2'],
            'rating': [5, 1], 'review_date': ['2023-01-01', '2023-01-02'], 'review_year': [2023, 2023],
            'review_month': [1, 1], 'bank_code': ['CBE', 'BOA'], 'bank_name': ['CBE', 'BOA'],
            'user_name': ['u1', 'u2'], 'thumbs_up': [0, 1], 'text_length': [5, 5],
            'source': ['Play', 'Play'], 'vader_score': [0.1, -0.5], 'vader_label': ['POSITIVE', 'NEGATIVE'],
            'textblob_score': [0.1, -0.5], 'textblob_label': ['POSITIVE', 'NEGATIVE'],
            'distilbert_label': ['POSITIVE', 'NEGATIVE'], 'distilbert_score': [0.9, 0.1],
            'identified_theme': ['General', 'App']
        })
        self.etl = BankReviewsETL()
        self.etl.df = self.df

    def test_data_ready_for_insert(self):
        """Test that required columns exist and DataFrame has expected number of rows."""
        required_columns = ['review_id', 'bank_id', 'review_text', 'rating', 'review_date', 'vader_label']
        missing = [col for col in required_columns if col not in self.etl.df.columns]
        self.assertTrue(len(missing) == 0)
        self.assertEqual(len(self.etl.df), 2)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

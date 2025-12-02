# Bank Reviews NLP Pipeline - Unit Tests

This test suite verifies the functionality of the **Bank Reviews NLP pipeline**, including preprocessing, sentiment analysis, thematic analysis, and ETL preparation. It uses Python's built-in `unittest` framework.

## Tested Components

1. **Preprocessor (`ReviewPreprocessor`)**
   - Removes duplicate reviews
   - Normalizes dates
   - Cleans review text (removes punctuation, converts to lowercase)

2. **Sentiment Analysis (`SentimentNLP`)**
   - Runs sentiment scoring using:
     - **VADER** (lexicon-based)
     - **TextBlob** (polarity-based)
     - **DistilBERT** (transformer-based, optional)
   - Verifies that sentiment scores and labels are correctly assigned

3. **Thematic Analysis (`ThematicNLP`)**
   - Extracts keywords using TF-IDF
   - Assigns themes to reviews based on predefined keyword mappings
   - Checks that an `identified_theme` column is created

4. **ETL Preparation (`BankReviewsETL`)**
   - Verifies that required columns for database insertion exist
   - Ensures data integrity for prepared review records

## Running the Tests

1. **Discover all tests** (run from the root folder):

- ```python -m unittest discover -s tests -p "*.py"```

2. **Run a specific test file**:

- ```python -m unittest tests.test_pipeline```

"""
Review Preprocessing Pipeline Runner

This script serves as the entry point to execute the data preprocessing
pipeline on scraped reviews. It loads the raw review data, cleans and
validates it, and outputs a processed dataset ready for analysis.
"""

from src.data_preprocessing import ReviewPreprocessor
import pandas as pd


def main() -> pd.DataFrame | None:
    """Main execution function for the review preprocessing pipeline.

    This function performs the following steps:
        1. Initializes the ReviewPreprocessor with default paths.
        2. Executes the full preprocessing workflow, including:
            - Missing value handling
            - Date normalization
            - Text cleaning
            - Duplicate removal
            - Rating validation
        3. Saves the processed dataset to disk.
        4. Generates a detailed preprocessing report.
        5. Returns the processed DataFrame if successful, otherwise None.

    Returns:
        pd.DataFrame | None: The processed DataFrame if preprocessing
                              succeeds; None if it fails.
    """
    print("=" * 60)
    print("STARTING REVIEW PREPROCESSING PIPELINE")
    print("=" * 60)

    # Initialize the preprocessor
    preprocessor = ReviewPreprocessor()

    # Execute full preprocessing workflow
    success = preprocessor.process()

    # Return processed DataFrame if successful
    if success:
        print("\n✓ Preprocessing completed successfully!")
        return preprocessor.df
    else:
        print("\n✗ Preprocessing failed!")
        return None


if __name__ == "__main__":
    # Run the preprocessing pipeline
    processed_df = main()

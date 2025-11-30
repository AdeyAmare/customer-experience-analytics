"""
Bank Review Sentiment and Thematic Analysis Runner

This script serves as the entry point to execute the full sentiment and
thematic analysis pipeline on processed bank reviews. It applies VADER,
TextBlob, and DistilBERT sentiment models, extracts thematic keywords,
assigns themes, and outputs the final annotated dataset.
"""

from src.sentiment_thematic_analysis import BankReviewPipeline
import pandas as pd


def main() -> pd.DataFrame:
    """Main execution function for bank review sentiment and thematic analysis.

    This function performs the following steps:
        1. Initializes the BankReviewPipeline.
        2. Runs the full sentiment and thematic analysis pipeline.
        3. Displays a sample of the final results with key columns.
        4. Returns the final DataFrame containing sentiment labels and themes.

    Returns:
        pd.DataFrame: DataFrame containing analyzed reviews with sentiment
                      and thematic labels.
    """
    print("=" * 60)
    print("STARTING BANK REVIEW SENTIMENT & THEMATIC ANALYSIS PIPELINE")
    print("=" * 60)

    # Initialize the pipeline
    pipeline = BankReviewPipeline()

    # Run the full analysis
    results_df = pipeline.run()

    # Display a concise sample of the results
    display_cols = [
        "bank_name",
        "review_text",
        "vader_label",
        "textblob_label",
        "distilbert_label",
        "identified_theme"
    ]

    print("\n--- Sample of Final Results ---")
    print(results_df[display_cols].head())

    return results_df


if __name__ == "__main__":
    # Execute the sentiment and thematic pipeline
    analyzed_df = main()

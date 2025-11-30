from src.sentiment_thematic_analysis import BankReviewPipeline

def main():
    pipeline = BankReviewPipeline()
    results_df = pipeline.run()
    display_cols = ["bank_name", "review_text", "vader_label", "textblob_label", "distilbert_label", "identified_theme"]
    print("\n--- Sample of Final Results ---")
    print(results_df[display_cols].head())

if __name__ == "__main__":
    main()

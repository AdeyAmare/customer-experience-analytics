from src.sentiment_thematic_analysis import BankReviewPipeline

def main():
    # Instantiate the full pipeline instead of just the sentiment class
    pipeline = BankReviewPipeline()
    
    final_df = pipeline.run()
    
    # Display columns to verify both Sentiment and Themes are present
    print(final_df[["bank_name", "vader_label", "identified_theme"]].head())

if __name__ == "__main__":
    main()

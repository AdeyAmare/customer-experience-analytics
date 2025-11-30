from src.data_preprocessing import ReviewPreprocessor

def main():
    """Main function to run the preprocessing pipeline.

    Returns:
        pd.DataFrame or None: The processed DataFrame if successful, else None.
    """
    preprocessor = ReviewPreprocessor()
    success = preprocessor.process()
    if success:
        print("\n✓ Preprocessing completed successfully!")
        return preprocessor.df
    else:
        print("\n✗ Preprocessing failed!")
        return None


if __name__ == "__main__":
    processed_df = main()
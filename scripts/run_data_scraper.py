"""
Google Play Store Review Scraper Runner

This script serves as the entry point to scrape user reviews from
Google Play Store for multiple Ethiopian banks. It initializes the
scraper, collects reviews, and displays sample reviews per bank.
"""

from src.data_scraper import PlayStoreScraper
import pandas as pd


def main() -> pd.DataFrame:
    """Main execution function for scraping Google Play Store reviews.

    This function performs the following steps:
        1. Initializes the PlayStoreScraper.
        2. Scrapes reviews for all configured banks.
        3. Displays sample reviews for each bank if data was collected.
        4. Returns the full DataFrame of scraped reviews.

    Returns:
        pd.DataFrame: DataFrame containing all scraped reviews.
                      Empty DataFrame if no reviews were collected.
    """
    print("=" * 60)
    print("STARTING GOOGLE PLAY STORE REVIEW SCRAPING")
    print("=" * 60)

    # Initialize scraper with configuration from utils.config
    scraper = PlayStoreScraper()

    # Scrape reviews for all banks
    df = scraper.scrape_all_banks()

    # Display sample reviews if any data was collected
    if not df.empty:
        scraper.display_sample_reviews(df, n=3)

    print("=" * 60)
    print("SCRAPING COMPLETED")
    print("=" * 60)

    return df


if __name__ == "__main__":
    # Run the scraper pipeline and collect reviews
    reviews_df = main()

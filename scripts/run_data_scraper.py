from src.data_scraper import PlayStoreScraper

def main():
    """Main execution function"""

    # Initialize scraper
    scraper = PlayStoreScraper()

    # Scrape all reviews
    df = scraper.scrape_all_banks()

    # Display samples if data was collected
    if not df.empty:
        scraper.display_sample_reviews(df)

    return df


if __name__ == "__main__":
    reviews_df = main()
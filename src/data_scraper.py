"""
Google Play Store Review Scraper
Task 1: Data Collection

This script scrapes user reviews from Google Play Store for Ethiopian banks.
Target: 400+ reviews per bank (1200 total minimum)
"""

import os
import time
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from google_play_scraper import app, Sort, reviews

from utils.config import APP_IDS, BANK_NAMES, SCRAPING_CONFIG, DATA_PATHS


class PlayStoreScraper:
    """Scraper for Google Play Store reviews for multiple banks."""

    def __init__(self):
        """Initialize the scraper with configuration from utils.config."""
        self.app_ids = APP_IDS
        self.bank_names = BANK_NAMES
        self.reviews_per_bank = SCRAPING_CONFIG.get('reviews_per_bank', 500)
        self.lang = SCRAPING_CONFIG.get('lang', 'en')
        self.country = SCRAPING_CONFIG.get('country', 'ET')
        self.max_retries = SCRAPING_CONFIG.get('max_retries', 3)

    def get_app_info(self, app_id: str) -> dict | None:
        """Fetch metadata about an app from Google Play Store.

        Args:
            app_id (str): Google Play app ID.

        Returns:
            dict | None: App info including title, rating, total reviews, installs, etc.
        """
        try:
            result = app(app_id, lang=self.lang, country=self.country)
            return {
                'app_id': app_id,
                'title': result.get('title', 'N/A'),
                'score': result.get('score', 0),
                'ratings': result.get('ratings', 0),
                'reviews': result.get('reviews', 0),
                'installs': result.get('installs', 'N/A')
            }
        except Exception as e:
            print(f"Error fetching app info for {app_id}: {e}")
            return None

    def scrape_reviews(self, app_id: str, count: int = 500) -> list[dict]:
        """Scrape reviews for a single app with retries.

        Args:
            app_id (str): Google Play app ID.
            count (int, optional): Number of reviews to fetch. Defaults to 500.

        Returns:
            list[dict]: List of raw review dictionaries.
        """
        print(f"\nScraping reviews for {app_id}...")
        for attempt in range(self.max_retries):
            try:
                result, _ = reviews(
                    app_id,
                    lang=self.lang,
                    country=self.country,
                    sort=Sort.NEWEST,
                    count=count,
                    filter_score_with=None
                )
                print(f"Successfully scraped {len(result)} reviews")
                return result
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("Max retries reached. Skipping this app.")
                    return []
        return []

    def process_reviews(self, reviews_data: list[dict], bank_code: str) -> list[dict]:
        """Convert raw reviews into a structured format.

        Args:
            reviews_data (list[dict]): Raw review data from scraper.
            bank_code (str): Bank code to annotate reviews with.

        Returns:
            list[dict]: Processed reviews.
        """
        processed = []
        for review in reviews_data:
            processed.append({
                'review_id': review.get('reviewId', ''),
                'review_text': review.get('content', ''),
                'rating': review.get('score', 0),
                'review_date': review.get('at', datetime.now()),
                'user_name': review.get('userName', 'Anonymous'),
                'thumbs_up': review.get('thumbsUpCount', 0),
                'reply_content': review.get('replyContent', None),
                'bank_code': bank_code,
                'bank_name': self.bank_names.get(bank_code, 'Unknown'),
                'app_id': review.get('reviewCreatedVersion', 'N/A'),
                'source': 'Google Play'
            })
        return processed

    def scrape_all_banks(self) -> pd.DataFrame:
        """Orchestrate the scraping process for all banks.

        Steps:
            1. Fetch app metadata
            2. Scrape reviews for each bank
            3. Save raw reviews and app info as CSV

        Returns:
            pd.DataFrame: Combined reviews for all banks.
        """
        all_reviews = []
        app_info_list = []

        print("=" * 60)
        print("Starting Google Play Store Review Scraper")
        print("=" * 60)

        # Fetch app metadata
        print("\n[1/2] Fetching app information...")
        for bank_code, app_id in self.app_ids.items():
            info = self.get_app_info(app_id)
            if info:
                info.update({'bank_code': bank_code, 'bank_name': self.bank_names[bank_code]})
                app_info_list.append(info)

        if app_info_list:
            os.makedirs(DATA_PATHS['raw'], exist_ok=True)
            pd.DataFrame(app_info_list).to_csv(f"{DATA_PATHS['raw']}/app_info.csv", index=False)
            print(f"App information saved to {DATA_PATHS['raw']}/app_info.csv")

        # Scrape reviews for each bank
        print("\n[2/2] Scraping reviews...")
        for bank_code, app_id in tqdm(self.app_ids.items(), desc="Banks"):
            reviews_data = self.scrape_reviews(app_id, self.reviews_per_bank)
            if reviews_data:
                processed = self.process_reviews(reviews_data, bank_code)
                all_reviews.extend(processed)
            time.sleep(2)  # polite delay

        # Save all reviews
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            os.makedirs(DATA_PATHS['raw'], exist_ok=True)
            df.to_csv(DATA_PATHS['raw_reviews'], index=False)
            print(f"\nScraping complete. Total reviews collected: {len(df)}")
            for bank_code in self.bank_names.keys():
                count = len(df[df['bank_code'] == bank_code])
                print(f"{self.bank_names[bank_code]}: {count} reviews")
            return df
        else:
            print("No reviews collected.")
            return pd.DataFrame()

    def display_sample_reviews(self, df: pd.DataFrame, n: int = 3) -> None:
        """Display sample reviews for each bank.

        Args:
            df (pd.DataFrame): DataFrame containing scraped reviews.
            n (int, optional): Number of reviews per bank to display. Defaults to 3.
        """
        print("\n" + "=" * 60)
        print("Sample Reviews")
        print("=" * 60)
        for bank_code in self.bank_names.keys():
            bank_df = df[df['bank_code'] == bank_code]
            if not bank_df.empty:
                print(f"\n{self.bank_names[bank_code]}:")
                print("-" * 60)
                for _, row in bank_df.head(n).iterrows():
                    print(f"\nRating: {'⭐' * row['rating']}")
                    print(f"Review: {row['review_text'][:200]}...")
                    print(f"Date: {row['review_date']}")

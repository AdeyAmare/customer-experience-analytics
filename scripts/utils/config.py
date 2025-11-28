"""
Configuration file for Bank Reviews Analysis Project
"""
import os
from dotenv import load_dotenv

# --- Project root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # main_folder

# --- Load .env ---
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path)

# --- Google Play Store App IDs ---
APP_IDS = {
    'CBE': os.getenv('CBE_APP_ID', 'com.combanketh.mobilebanking'),
    'BOA': os.getenv('BOA_APP_ID', 'com.boa.boaMobileBanking'),
    'Dashen': os.getenv('DASHEN_APP_ID', 'com.dashen.dashensuperapp'),
}

# --- Bank Names Mapping ---
BANK_NAMES = {
    'CBE': 'Commercial Bank of Ethiopia',
    'BOA': 'Bank of Abyssinia',
    'Dashen': 'Dashen Bank'
}

# --- Scraping Configuration ---
SCRAPING_CONFIG = {
    'reviews_per_bank': int(os.getenv('REVIEWS_PER_BANK', 500)),
    'max_retries': int(os.getenv('MAX_RETRIES', 3)),
    'lang': os.getenv('LANG', 'en'),
    'country': os.getenv('COUNTRY', 'et')
}

# --- File Paths (absolute) ---
DATA_PATHS = {
    'raw': os.path.join(PROJECT_ROOT, 'data/raw'),
    'processed': os.path.join(PROJECT_ROOT, 'data/processed'),
    'raw_reviews': os.path.join(PROJECT_ROOT, 'data/raw/reviews_raw.csv'),
    'processed_reviews': os.path.join(PROJECT_ROOT, 'data/processed/reviews_processed.csv'),
    'sentiment_results': os.path.join(PROJECT_ROOT, 'data/processed/reviews_with_sentiment.csv'),
    'final_results': os.path.join(PROJECT_ROOT, 'data/processed/reviews_final.csv')
}

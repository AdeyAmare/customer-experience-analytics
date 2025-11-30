# Project Configuration Overview

This file centralizes all configuration settings and parameters used across the project pipeline.

- Purpose: To store constants, paths, IDs, and other critical variables in a single location for easy maintenance and modification.

- Contents:

    - File Paths: Defines the DATA_PATHS for raw, processed, and final results.

    - Bank/App IDs: Maps bank codes to their names (BANK_NAMES) and Google Play Store IDs (APP_IDS).

    - Scraping Settings: Contains parameters like reviews_per_bank, lang, country, and max_retries (SCRAPING_CONFIG).

Benefit: Allows developers to quickly change core settings (like the number of reviews to scrape or a file location) without modifying the logic files in the src/ directory.
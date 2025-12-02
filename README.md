# Customer Experience Analytics for Ethiopian Banks via Google Play Store Reviews

This project provides a comprehensive data pipeline to collect, clean, and analyze user reviews for Ethiopian banks from the Google Play Store.

## Project Pipeline Stages

1. **Data Collection:** Scrapes raw user reviews from the Google Play Store.

2. **Data Preprocessing:** Cleans, validates, and prepares the raw data, handling duplicates, missing values, and text normalization.

3. **NLP Analysis:** Applies sentiment analysis (VADER, DistilBERT, TextBlob) and identifies key themes from reviews.

4. **ETL & Database Loading:** Loads the processed data into a PostgreSQL database, mapping banks and inserting reviews for querying and analysis.

## Folder Structure

- `scripts/`: Files to run the pipeline (`run_*.py` and `postgres_connection.py`).  
- `src/`: Core logic and class definitions (preprocessing, NLP, ETL helpers).  
- `utils/`: Configuration and settings (`config.py`).  
- `notebooks/`: Jupyter Notebooks for detailed EDA and model testing.  
- `database/`: Database schema and query results.  

## Scripts Overview

1. `run_data_scraper.py`  
   - **Purpose:** Initiates the data collection phase.  
   - **Effect:** Scrapes user reviews for configured banks and saves raw data.

2. `run_data_preprocesser.py`  
   - **Purpose:** Initiates the data cleaning phase.  
   - **Effect:** Cleans text, validates ratings, handles duplicates, and saves the cleaned dataset to `data/processed/`.

3. `run_sentiment_thematic_analysis.py`  
   - **Purpose:** Initiates NLP analysis.  
   - **Effect:** Runs sentiment models, extracts themes, aggregates results, and saves processed data.

4. `postgres_connection.py`  
   - **Purpose:** Loads processed review data into PostgreSQL.  
   - **Effect:** Inserts unique banks into the `banks` table, maps `bank_id`s, inserts reviews into the `reviews` table, and safely handles duplicates.  

## Database Directory

- `schema.sql`  
  - **Purpose:** Defines the database structure.  
  - **Contents:** SQL commands to create tables like `banks` and `reviews`, with constraints and indexes.  

- `query_results/`  
  - **Purpose:** Stores results from important or frequently run queries.  
  - **Contents:** CSV or SQL export files for reporting, validation, or testing purposes.

## How to Run

### 1. Execute the Pipeline (Scripts)
Run these files in sequence from the main directory:

```bash
python scripts/run_data_scraper.py
python scripts/run_data_preprocesser.py
python scripts/run_sentiment_thematic_analysis.py
python scripts/postgres_connection.py

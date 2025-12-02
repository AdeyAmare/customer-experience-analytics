# Scripts Directory

This folder contains the main execution files for the project's data pipeline. These scripts orchestrate the steps defined in the src/ directory.

Execution Scripts and Their Purpose
1. ```run_data_scraper.py```

- Purpose: Initiates the data collection phase.

- Effect: Scrapes user reviews from the Google Play Store for the configured banks and saves the raw data.

2. ```run_data_preprocesser.py```

- Purpose: Initiates the data cleaning and preparation phase.

- Effect: Loads the raw data, cleans text, handles missing values and duplicates, validates ratings, and saves the cleaned dataset to the ```data/processed/``` directory.

3. ```run_sentiment_thematic_analysis```.py

- Purpose: Initiates the Natural Language Processing (NLP) analysis phase.

- Effect: Runs multiple sentiment models (VADER, TextBlob, DistilBERT), performs TF-IDF keyword extraction, assigns themes, aggregates sentiment, and saves the final analytical results to ```data/processed/```.

4. ```postgres_connection.py```

Purpose: Initiates the ETL phase to load processed review data into PostgreSQL.

Effect: Reads the final processed review CSV, inserts unique banks into the banks table, maps bank_ids, inserts reviews into the reviews table, and manages database connections. Designed to handle duplicates safely and log progress.

## Pipeline Execution Order
The scripts must be executed sequentially to ensure each stage has the necessary input data:

Run the data collection: ```python run_data_scraper.py```

Run the data cleaning: ```python run_data_preprocesser.py```

Run the analysis: ```python run_sentiment_thematic_analysis.py```

Load the processed data into PostgreSQL: ```python postgres_connection.py```
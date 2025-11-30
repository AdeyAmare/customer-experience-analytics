# Customer Experience Analytics for Ethiopian Banks via Google Play Store Reviews

This project provides a comprehensive data pipeline to collect, clean, and analyze user reviews for Ethiopian banks from the Google Play Store.

## Project Pipeline Stages
1. Data Collection: Scrapes raw user reviews.

2. Data Preprocessing: Cleans, validates, and prepares the raw data.

3. NLP Analysis: Applies sentiment analysis (VADER, DistilBERT, Textblob) and identifies key themes.

## Folder Structure
- ```scripts/```: Files to run the pipeline (run_*.py).

- ```src/```: Core logic and class definitions.

- ```utils/```: Configuration and settings (config.py).

- ```notebooks/```: Jupyter Notebooks for detailed EDA and model testing.

## How to Run
There are two main parts to running the project:

1. Execute the Pipeline (Scripts)
Run these files in sequence from the main directory:

- ```python scripts/run_data_scraper.py```

- ```python scripts/run_data_preprocesser.py```

- ```python scripts/run_sentiment_thematic_analysis.py```

2. Explore the Analysis (Notebooks)

- ```preprocessing_eda.ipynb```: For data exploration and visualization.

- ```sentiment_analysis.ipynb```: For in-depth model testing and final results visualization.
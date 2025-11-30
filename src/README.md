# src/ - Core Modules and Logic

This folder contains the core reusable logic for the project. The modules here define the classes and functions responsible for data collection, cleaning, and natural language processing.

## Core Modules and Logic
1. ```data_scraper.py```

- Core Class: ```PlayStoreScraper```.

- Logic: Handles connecting to the Google Play Store, fetching app metadata, and collecting raw user reviews. It includes functions for retries and processing raw reviews into a structured format.

2. ```data_preprocessing.py```

- Core Class: ```ReviewPreprocessor```.

- Logic: Implements the full data cleaning pipeline, which includes loading raw data, checking and handling missing values, normalizing dates, cleaning review text (removing non-English characters and short reviews), removing duplicates, and validating ratings (1-5).

3. ```sentiment_thematic_analysis.py```

- Core Classes: ```SentimentNLP```, ```ThematicNLP```, and ```BankReviewPipeline```.

Logic:

- SentimentNLP runs three sentiment models: VADER, TextBlob, and a Hugging Face DistilBERT pipeline.

- ThematicNLP preprocesses text, uses TF-IDF for keyword extraction, and assigns predefined thematic categories (e.g., "Mobile App & Digital", "Customer Service") based on keywords.

- BankReviewPipeline orchestrates the sentiment and thematic analysis and saves the final output.
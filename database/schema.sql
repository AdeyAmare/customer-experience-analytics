CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name TEXT UNIQUE NOT NULL
);

CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    bank_id INT REFERENCES banks(bank_id),
    review_text TEXT,
    rating NUMERIC,
    review_date DATE,
    review_year INT,
    review_month INT,
    bank_code TEXT,
    bank_name TEXT,
    user_name TEXT,
    thumbs_up INT,
    text_length INT,
    source TEXT,
    vader_score NUMERIC,
    vader_label TEXT,
    textblob_score NUMERIC,
    textblob_label TEXT,
    distilbert_label TEXT,
    distilbert_score NUMERIC,
    identified_theme TEXT
);

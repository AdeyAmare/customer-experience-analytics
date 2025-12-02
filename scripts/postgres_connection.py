import os
import sys
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


# Add project root to path (for utils import)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import DATA_PATHS 

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")


class BankReviewsETL:
    """ETL class to load bank review data from CSV and insert into PostgreSQL."""

    def __init__(self):
        """Initialize database connection parameters and CSV path from .env."""
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_name = os.getenv("DB_NAME", "bank_reviews")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD")
        self.csv_path = DATA_PATHS['sentiment_results']
        self.conn = None
        self.cur = None
        self.df = None

    def load_csv(self) -> None:
        """Load review CSV into a pandas DataFrame."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.df['review_date'] = pd.to_datetime(self.df['review_date'], errors='coerce')
        print(f"Loaded {len(self.df)} rows from CSV.")

    def connect_db(self) -> None:
        """Connect to the PostgreSQL database."""
        self.conn = psycopg2.connect(
            host=self.db_host,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )
        self.cur = self.conn.cursor()
        print("Connected to PostgreSQL.")

    def insert_banks(self) -> None:
        """Insert unique banks into the banks table."""
        banks = self.df[['bank_code', 'bank_name']].drop_duplicates()

        for _, row in banks.iterrows():
            self.cur.execute(
                """
                INSERT INTO banks (bank_name)
                VALUES (%s)
                ON CONFLICT (bank_name) DO NOTHING;
                """,
                (row['bank_name'],)
            )
        self.conn.commit()
        print(f"Inserted/checked {len(banks)} banks.")


    def map_bank_ids(self) -> None:
        """Map bank_name to bank_id and add as a column in DataFrame."""
        self.cur.execute("SELECT bank_id, bank_name FROM banks")
        rows = self.cur.fetchall()
        bank_name_to_id = {name: bank_id for bank_id, name in rows}
        self.df['bank_id'] = self.df['bank_name'].map(bank_name_to_id)
        print("Mapped bank_ids.")

    def insert_reviews(self) -> None:
        """Insert reviews into the reviews table."""
        records = [
            (
                row['review_id'],
                row['bank_id'],
                row['review_text'],
                row['rating'],
                row['review_date'],
                row['review_year'],
                row['review_month'],
                row['bank_code'],
                row['bank_name'],
                row['user_name'],
                row['thumbs_up'],
                row['text_length'],
                row['source'],
                row['vader_score'],
                row['vader_label'],
                row['textblob_score'],
                row['textblob_label'],
                row['distilbert_label'],
                row['distilbert_score'],
                row['identified_theme']
            )
            for _, row in self.df.iterrows()
        ]

        sql = """
        INSERT INTO reviews (
            review_id, bank_id,
            review_text, rating, review_date,
            review_year, review_month,
            bank_code, bank_name, user_name,
            thumbs_up, text_length, source,
            vader_score, vader_label,
            textblob_score, textblob_label,
            distilbert_label, distilbert_score,
            identified_theme
        ) VALUES %s
        ON CONFLICT (review_id) DO NOTHING;
        """

        execute_values(self.cur, sql, records, page_size=500)
        self.conn.commit()
        print(f"Inserted {len(records)} reviews.")

    def close_connection(self) -> None:
        """Close the database connection."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        print("Database connection closed.")

    def run(self) -> None:
        """Run the full ETL process."""
        self.load_csv()
        self.connect_db()
        self.insert_banks()
        self.map_bank_ids()
        self.insert_reviews()
        self.close_connection()


if __name__ == "__main__":
    etl = BankReviewsETL()
    etl.run()

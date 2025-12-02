-- Sentiment distribution per bank (Vader, TextBlob, DistilBERT)
SELECT 
    bank_id,
    bank_code,
    bank_name,
    vader_label,
    textblob_label,
    distilbert_label,
    COUNT(*) AS review_count
FROM reviews
GROUP BY 
    bank_id,
    bank_code,
    bank_name,
    vader_label,
    textblob_label,
    distilbert_label
ORDER BY bank_id, review_count DESC;

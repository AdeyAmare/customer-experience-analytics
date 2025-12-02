-- Count of reviews per theme and sentiment (Vader, TextBlob, DistilBERT)
SELECT 
    identified_theme,
    vader_label,
    textblob_label,
    distilbert_label,
    COUNT(*) AS review_count
FROM reviews
GROUP BY 
    identified_theme,
    vader_label,
    textblob_label,
    distilbert_label
ORDER BY identified_theme, review_count DESC;

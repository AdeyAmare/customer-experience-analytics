-- Count of reviews per identified theme
SELECT identified_theme, COUNT(*) AS review_count
FROM reviews
GROUP BY identified_theme
ORDER BY review_count DESC;

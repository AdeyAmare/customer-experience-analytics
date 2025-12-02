-- Average rating per identified theme
SELECT identified_theme, AVG(rating) AS avg_rating
FROM reviews
GROUP BY identified_theme
ORDER BY avg_rating DESC;

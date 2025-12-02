-- Average rating per bank
SELECT bank_id, bank_code, bank_name, AVG(rating) AS avg_rating
FROM reviews
GROUP BY bank_id, bank_code, bank_name
ORDER BY avg_rating DESC;

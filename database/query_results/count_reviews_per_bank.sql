-- Count of reviews per bank
SELECT bank_id, bank_code, bank_name, COUNT(*) AS review_count
FROM reviews
GROUP BY bank_id, bank_code, bank_name
ORDER BY review_count DESC;

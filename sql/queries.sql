-- Q1: WHERE then HAVING
-- Completed orders > 500, customers whose total exceeds 8000
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: Revenue by city and product category (three-table join)
SELECT c.city, p.category, SUM(o.order_amount) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: Products priced above average catalogue price
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: Customers with no completed order (LEFT JOIN)
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: CTE - customers above average total spend
WITH customer_totals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total
    FROM customer_totals
)
SELECT ct.customer_id, ct.total_spend, at.avg_total
FROM customer_totals ct, avg_spend at
WHERE ct.total_spend > at.avg_total;

-- Q6: Ranking window functions per city
SELECT
    customer_id,
    city,
    total_spend,
    RANK()         OVER (PARTITION BY city ORDER BY total_spend DESC) AS rank_val,
    DENSE_RANK()   OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_val,
    ROW_NUMBER()   OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM (
    SELECT o.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
) t;

-- Q7: Running total of daily completed revenue
SELECT
    SUBSTR(order_ts, 1, 10) AS day,
    SUM(order_amount)       AS revenue,
    SUM(SUM(order_amount))  OVER (ORDER BY SUBSTR(order_ts, 1, 10)) AS cumulative_revenue
FROM orders
WHERE status = 'completed'
GROUP BY SUBSTR(order_ts, 1, 10)
ORDER BY day;

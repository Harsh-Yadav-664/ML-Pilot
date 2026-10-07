-- A shop small enough to check labels by hand (tests/unit/test_labels.py, #50).
-- Task: "no order in the next 30 days" at cutoffs 2024-03-01, 2024-04-01 and 2024-05-01.
-- Data ends 2024-07-01. Each customer is built to test one edge:
--   1  orders 02-10 and exactly 03-31 00:00 (the window end of the first cutoff: inside it)
--   2  orders 02-20, exactly 04-01 00:00 (the second cutoff: NOT in its window, but history
--      for the third) and 05-15
--   3  no order before any cutoff, one on 06-20 (the future must not make it eligible)
--   4  signs up 03-10 (not there at the first cutoff); orders 03-20 and 04-20
--   5  a cancelled order 03-15 (filtered out of the target) and an order 06-15 after every window
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, signup_at TIMESTAMP);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers (customer_id),
    ordered_at TIMESTAMP,
    status VARCHAR
);
INSERT INTO customers VALUES
    (1, '2024-01-01 00:00:00'), (2, '2024-01-01 00:00:00'), (3, '2024-01-01 00:00:00'),
    (4, '2024-03-10 00:00:00'), (5, '2024-01-05 00:00:00');
INSERT INTO orders VALUES
    (1, 1, '2024-02-10 12:00:00', 'completed'),
    (2, 1, '2024-03-31 00:00:00', 'completed'),
    (3, 2, '2024-02-20 09:00:00', 'completed'),
    (4, 2, '2024-04-01 00:00:00', 'completed'),
    (5, 2, '2024-05-15 10:00:00', 'completed'),
    (6, 3, '2024-06-20 10:00:00', 'completed'),
    (7, 4, '2024-03-20 10:00:00', 'completed'),
    (8, 4, '2024-04-20 10:00:00', 'completed'),
    (9, 5, '2024-02-01 10:00:00', 'completed'),
    (10, 5, '2024-03-15 10:00:00', 'cancelled'),
    (11, 5, '2024-06-15 10:00:00', 'completed');

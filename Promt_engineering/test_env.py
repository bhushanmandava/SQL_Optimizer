# test-env.py
test_queries = [
      "SELECT * FROM (SELECT * FROM orders) AS sub WHERE sub.order_date > '2023-01-01'"
]

CREATE DATABASE IF NOT EXISTS delivery_prediction_db;
USE delivery_prediction_db;

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id      INT AUTO_INCREMENT PRIMARY KEY,
    customer_state      VARCHAR(10),
    customer_city       VARCHAR(100),
    product_category    VARCHAR(100),
    predicted_days_mean FLOAT,
    predicted_days_min  FLOAT,
    predicted_days_max  FLOAT,
    prediction_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

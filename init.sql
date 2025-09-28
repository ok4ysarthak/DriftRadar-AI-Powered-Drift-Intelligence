CREATE TABLE IF NOT EXISTS drift_metrics (
    id SERIAL PRIMARY KEY,
    batch_id INT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    feature_name VARCHAR(255) NOT NULL,
    drift_score REAL,
    is_drifted BOOLEAN NOT NULL,
    alert_message TEXT,
    metric_type VARCHAR(50)
);
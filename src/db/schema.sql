-- Every time we run an evaluation, one row goes here
CREATE TABLE IF NOT EXISTS eval_runs (
    id          SERIAL PRIMARY KEY,
    run_name    VARCHAR(255) NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW(),
    config      JSONB,          -- stores the full config used for this run
    total_cost  DECIMAL(10,6),  -- total $ spent on this run
    notes       TEXT
);

-- One row per model per run
CREATE TABLE IF NOT EXISTS model_results (
    id          SERIAL PRIMARY KEY,
    run_id      INTEGER REFERENCES eval_runs(id) ON DELETE CASCADE,
    model_name  VARCHAR(100) NOT NULL,
    dataset     VARCHAR(100) NOT NULL,
    prompt      TEXT NOT NULL,
    response    TEXT,
    latency_ms  INTEGER,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd    DECIMAL(10,6),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- One row per metric per model result
CREATE TABLE IF NOT EXISTS metric_scores (
    id              SERIAL PRIMARY KEY,
    model_result_id INTEGER REFERENCES model_results(id) ON DELETE CASCADE,
    metric_name     VARCHAR(100) NOT NULL,
    score           DECIMAL(5,4),   -- normalized 0.0 to 1.0
    raw_value       JSONB,          -- stores the full metric output
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Indexes for the queries we'll run most
CREATE INDEX IF NOT EXISTS idx_model_results_run_id ON model_results(run_id);
CREATE INDEX IF NOT EXISTS idx_model_results_model_name ON model_results(model_name);
CREATE INDEX IF NOT EXISTS idx_metric_scores_metric_name ON metric_scores(metric_name);
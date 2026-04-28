# End-to-End MLOps Pipeline — Fraud Detection

```mermaid
flowchart TD
    %% ── DATA SOURCES ──────────────────────────────────────────
    subgraph SOURCES["📦 Data Sources"]
        CSV["🗂️ CSV Files"]
        PG_SRC["🐘 PostgreSQL\n(transactions table)"]
        BQ["☁️ BigQuery"]
    end

    %% ── INITIAL TRAINING ──────────────────────────────────────
    subgraph TRAIN["🏋️ Initial Training Pipeline  (main.py · Prefect)"]
        direction TB
        EXT["extraction.py\nCSVExtractor / PostgresExtractor / BigQueryExtractor"]
        FE["preprocessing.py\nFeatureEngineer\n(clean · log transform · time features · interaction terms)"]
        PRE["preprocessing.py\nPreprocessor\n(train/val/test split → SMOTE on train set only)"]
        TR["train.py\nTrainer\n(RandomForest · XGBoost · LogisticRegression)\nlogged to MLflow"]
        TUNE["model_tuning.py\nOptuna Tuner\n(50 trials · optional via do_tuning=True)"]
        VAL["validate.py\nValidator\n(ROC-AUC · F1 · Precision · Recall)"]
        PROMOTE["MlflowClient\nPromote best model\n→ 'production' alias"]
    end

    CSV & PG_SRC & BQ --> EXT
    EXT --> FE --> PRE --> TR --> TUNE --> VAL --> PROMOTE

    %% ── STORAGE LAYER ─────────────────────────────────────────
    subgraph STORAGE["🗄️ Storage Layer"]
        MINIO["🪣 MinIO  (S3-compatible)\n• models/{name}.pkl\n• preprocessor/preprocessor.pkl\n• reference/reference_data.parquet"]
        MLFLOW["📊 MLflow\n• Experiment tracking\n• Model registry + aliases\n  (production · previous-production)"]
        PGDB["🐘 PostgreSQL\n• transactions\n• predictions\n• fraud_feedback"]
    end

    TR -->|"save model + preprocessor"| MINIO
    TR -->|"log params · metrics · artifacts"| MLFLOW
    PROMOTE -->|"set alias 'production'"| MLFLOW

    %% ── SERVING ───────────────────────────────────────────────
    subgraph API["🚀 Serving  (app.py · FastAPI · uvicorn)"]
        direction TB
        STARTUP["Startup · lifespan()\nLoad model + preprocessor from MinIO\nLaunch background reload loop"]
        BG["⏱️ Background reload loop\nPoll MLflow every 5 min\nReload silently if new 'production' alias"]
        PREDICT["/predict\n→ FeatureEngineer on input\n→ model.predict + predict_proba\n→ Log to predictions table"]
        MONITOR_EP["/monitor  (on-demand)\n→ FeatureEngineer ref + current\n→ DriftMonitor.run()"]
        HEALTH["/health\nModel name · version · reload interval"]
    end

    MINIO -->|"load model + preprocessor"| STARTUP
    MLFLOW -->|"check alias"| BG
    STARTUP --> BG
    PREDICT -->|"write prediction\n+ confidence score"| PGDB

    %% ── DAILY MONITORING ──────────────────────────────────────
    subgraph MONITOR["📡 Daily Monitoring  (monitoring_schedule.py · Prefect · 9am daily)"]
        direction TB
        REF["load_reference()\nfrom MinIO"]
        CURR["load_current()\nfrom PostgreSQL\n+ FeatureEngineer applied"]
        SCORE["score_for_drift()\nRun prod model on both datasets\nAdd 'predicted_fraud' column"]
        PERF["load_performance_data()\nJoin predictions + fraud_feedback\n(Option B — self-activates on first feedback)"]
        DRIFT["DriftMonitor.run()\n━━━━━━━━━━━━━━━━━━━━\nLayer 1 · DataDriftPreset\n  (feature distribution shift)\nLayer 2 · ColumnDriftMetric\n  (prediction output drift)\nLayer 3 · ClassificationPreset\n  (recall · precision · F1 drift)"]
        SEV["Severity Classification\n🔴 serious   ≥50% features OR pred drift OR recall <0.70\n🟡 negligible 15–49% features, no pred drift\n🟢 none      <15% features"]
    end

    MINIO -->|"reference_data.parquet"| REF
    PGDB -->|"last 24h transactions"| CURR
    PGDB -->|"predictions + fraud_feedback join"| PERF
    MLFLOW -->|"load prod model"| SCORE
    MINIO -->|"load preprocessor"| SCORE
    REF & CURR --> SCORE --> DRIFT
    PERF --> DRIFT
    DRIFT --> SEV

    %% ── TIERED RESPONSE ───────────────────────────────────────
    subgraph RESPONSE["⚡ Tiered Retraining Response"]
        TRIG_NOW["🔴 Serious\ntrigger_retraining_if_needed()\n→ Prefect API: create_flow_run\n  trigger_reason: serious_drift"]
        WAIT["🟡 Negligible\nSkip immediate trigger\nSunday cron handles it"]
        OK["🟢 None\nNo action needed"]
        SLACK["📣 notify_slack()\nAlerts team for ALL severities\n(Slack Incoming Webhook)"]
    end

    SEV -->|"serious"| TRIG_NOW
    SEV -->|"negligible"| WAIT
    SEV -->|"none"| OK
    SEV --> SLACK

    %% ── RETRAINING ────────────────────────────────────────────
    subgraph RETRAIN["🔁 Retraining Pipeline  (retrain_flow.py · Prefect)"]
        direction TB
        R_EXT["retrain_extract()"]
        R_FE["retrain_feature_engineer()"]
        R_PRE["retrain_preprocess()\n(SMOTE on train only)"]
        R_TR["retrain_train()\n3 models · MLflow logged"]
        R_VAL["retrain_validate_and_promote()\nPromote best → 'production'\nSave previous → 'previous-production'"]
        R_REF["update_reference_data()\nSave new training data\nto MinIO reference baseline\n(only AFTER successful promotion)"]
    end

    TRIG_NOW -->|"immediate"| R_EXT
    WAIT -.->|"every Sunday midnight"| R_EXT
    R_EXT --> R_FE --> R_PRE --> R_TR --> R_VAL --> R_REF
    R_VAL -->|"new 'production' alias"| MLFLOW
    R_REF -->|"overwrite reference_data.parquet"| MINIO

    %% ── FEEDBACK LOOP ─────────────────────────────────────────
    subgraph FEEDBACK["🔄 Feedback Loop  (Option B)"]
        CHARGEBACK["💳 Chargebacks\n(payment processor webhook)"]
        DISPUTE["📱 Customer disputes\n(app / bank report)"]
        MANUAL["👤 Fraud analyst\nmanual review"]
    end

    CHARGEBACK & DISPUTE & MANUAL -->|"confirmed fraud label\n→ fraud_feedback table"| PGDB

    %% ── INFRASTRUCTURE ────────────────────────────────────────
    subgraph INFRA["🏗️ Infrastructure  (Docker Compose)"]
        direction LR
        PROM["📈 Prometheus\n/metrics scrape"]
        GRAF["📊 Grafana\ndashboards"]
        PREFECT_SRV["🔧 Prefect Server\n+ Worker"]
    end

    PROM --> GRAF
    PREFECT_SRV -->|"runs"| MONITOR & RETRAIN

    %% ── BACKGROUND RELOAD TRIGGER ─────────────────────────────
    BG -->|"new alias detected → silent reload"| STARTUP
```

---

## Summary

| Stage | Trigger | Runs On |
|---|---|---|
| Initial Training | Manual | Prefect (one-off) |
| Serving | On request | FastAPI (always on) |
| Model reload | Every 5 min (background) | FastAPI lifespan task |
| Drift Monitoring | Daily 9am | Prefect cron |
| Slack alert | Every monitor run | Prefect task |
| Retraining (serious) | Drift ≥50% or prediction drift | Prefect (immediate) |
| Retraining (scheduled) | Every Sunday midnight | Prefect cron |
| Reference baseline refresh | After successful promotion | End of retrain_pipeline |
| Option B performance check | Self-activates on first feedback row | Monitoring flow |

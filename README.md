# Fraud Detection ML System

> A production-grade machine learning system that catches fraudulent transactions in real time, monitors itself daily, and retrains automatically - so your fraud detection never goes stale.

---

## The Problem This Solves

Every financial platform faces the same uncomfortable truth: fraud is not a one-time problem. Fraudsters adapt. They find new patterns, exploit new channels, and evolve their tactics faster than any static rule-based system can keep up with.

Traditional fraud detection breaks in three predictable ways:

- **It misses novel fraud** - rules can only catch what you already know about
- **It ages silently** - a model trained six months ago is often already outdated, but nobody knows until customers start complaining
- **It can't explain itself** - when a regulator asks why a transaction was flagged, "the model said so" is not an answer

This system was built to fix all three.

---

## What It Does

**In plain language:** a transaction arrives, the system scores it for fraud risk within milliseconds, the result is returned to your platform, and the entire decision is logged for audit.

Behind the scenes, three things run continuously without any human intervention:

1. **Real-time scoring** - every transaction gets a fraud probability score the moment it arrives
2. **Daily health checks** - the system compares today's transactions to the patterns it was trained on and raises an alert if something has shifted
3. **Automatic retraining** - when drift is detected, a full retraining pipeline runs, a new model is evaluated, and only if it outperforms the current one does it take over

The result is a fraud detection system that gets better over time, not worse.

---

## Business Impact

| What used to happen | What happens now |
|---|---|
| Fraud patterns changed and the model silently degraded | Daily monitoring catches drift before customers feel it |
| Retraining required a data scientist to manually kick off a pipeline | Retraining triggers automatically when performance drops |
| A flagged transaction couldn't be explained to a regulator | Every prediction is traced to a specific model version, dataset, and confidence score |
| Scaling the fraud team meant hiring more analysts | The API handles any transaction volume without additional headcount |
| Rolling back a bad model was a manual emergency | Previous model versions are preserved and can be restored instantly |

---

## How It Works - Without the Jargon

Think of this system as having four jobs running at once:

**Job 1 - The Scorer**
Your platform sends a transaction to the system. Within milliseconds, it returns a fraud probability (e.g. 87% likely fraud) and a clear flag. Your platform decides what to do with that - block, review, or allow. The system's job is to be fast and accurate.

**Job 2 - The Watchdog**
Every morning at 9am, the system runs a silent health check. It compares the transactions coming in today against the patterns it learned during training. If the gap is too large - a sign that fraud tactics have shifted - it sends an alert to Slack and schedules an immediate retraining.

**Job 3 - The Learner**
Every Sunday at midnight, the system retrains itself on fresh data. It trains three candidate models, picks the best-performing one, and only promotes it to production if it genuinely outperforms what's already running. If the new model is worse, nothing changes.

**Job 4 - The Record Keeper**
Every prediction, every training run, every model version, and every metric is logged. Nothing is ever deleted. This means you can answer any compliance question - "which model flagged this transaction, and what was its accuracy that week?" - in seconds.

---

## Key Capabilities

### Real-Time Decisions
Fraud scoring happens in milliseconds via a secure API. Batch scoring is also supported for overnight processing of large transaction volumes.

### ️ Three Layers of Monitoring
The system doesn't just check if the model is running - it checks if the *data* has shifted, if the *model's outputs* have shifted, and if the *real-world accuracy* has dropped. These are three different failure modes, and all three are covered.

### Secure by Design
Every API call requires authentication. All requests are rate-limited to prevent abuse. Webhook callbacks are signature-verified. Access is audited and logged.

### Explainable Decisions
When the model flags a transaction, it can tell you *why* - which factors contributed most to the fraud score. This is critical for regulatory compliance and for building trust with customers who dispute a flag.

### ️ Self-Healing
The system doesn't need a data scientist on-call to stay healthy. Monitoring, retraining, model selection, and promotion all happen automatically. The team is notified via Slack at every step, but intervention is only needed if something genuinely unusual happens.

### Pluggable Data Sources
The system can ingest transactions from CSV files, PostgreSQL databases, or Google BigQuery with no code changes - just a configuration switch. New data sources can be added without touching the core pipeline.

---

## Alerts & Notifications

The system sends Slack alerts for every meaningful event:

| Event | Alert Type |
|---|---|
| Serious drift detected (≥50% of features shifted) | Critical - immediate retraining triggered |
| Moderate drift detected (15–49% of features shifted) | ️ Warning - Sunday retraining scheduled |
| Model recall dropped below 70% | Critical - too many fraud cases being missed |
| New model promoted to production | Info - model version and performance logged |
| Pipeline healthy | Info - daily confirmation, no action needed |

For critical alerts, PagerDuty can also be configured to page on-call engineers.

---

## What Makes This Different

Most ML projects are notebooks that became APIs. This is a system.

**Reproducibility** - every training run is versioned. You can reproduce any model ever trained, on any dataset, at any point in history.

**Reliability** - the API retries failed storage operations automatically, opens a circuit breaker to prevent cascading failures, and caches frequently-loaded models to keep response times low.

**Testability** - the test suite covers 120+ scenarios, prioritised by business risk. The most critical tests - the ones that catch fraud slipping through - run first.

**Adaptability** - the architecture is explicitly designed so that any component can be swapped without touching the rest. Replace the database, the storage layer, the scheduling tool, or any model - the pipeline doesn't change.

---

## Technology Overview

*For those who need to know what's under the hood:*

### A Note on Tool Choices

The tools used in this project are deliberately enterprise-grade. Fraud detection is a high-stakes, high-volume problem - the kind where a silent failure costs real money, and where regulators may ask questions that require a full audit trail going back months. The tools chosen reflect that reality: they are the same tools used by fintech companies, banks, and large-scale ML teams operating at production scale.

That said, this system was architected so that every tool is swappable. A startup or mid-size team that doesn't yet need - or can't yet justify - the cost and complexity of the full enterprise stack can replace each tool with a lighter, lower-cost alternative and the core pipeline logic stays exactly the same. Not a rewrite. A swap.

---

### Enterprise Stack - What This Project Uses

| What it does | Enterprise Tool | Why Enterprise |
|---|---|---|
| Serving fraud predictions via API | **FastAPI** | High-performance, async-ready, production standard |
| Training and comparing models | **Scikit-learn + XGBoost** | Industry-standard ML libraries, audit-friendly |
| Tracking experiments and model versions | **MLflow Server** | Full model registry, versioning, artifact storage, team UI |
| Storing models and data snapshots | **MinIO** | Self-hosted S3-compatible object storage, full data sovereignty |
| Scheduling and orchestrating pipelines | **Prefect** | Managed orchestration with retries, observability, and deployment UI |
| Monitoring data and model drift | **Evidently** | Production-grade drift reports with full statistical test suite |
| Explaining individual predictions | **SHAP** | Regulator-ready explainability for every decision |
| Storing transaction and prediction logs | **PostgreSQL** | Enterprise relational database, ACID-compliant |
| Running all services consistently | **Docker + Docker Compose** | Reproducible, portable, cloud-agnostic infrastructure |
| Provisioning cloud infrastructure | **Terraform** | Infrastructure as code — GCS bucket and BigQuery dataset version-controlled and reproducible |
| Cloud data lake & warehouse | **GCS + BigQuery** | Scalable object storage and serverless SQL analytics on GCP |
| Alerting and incident management | **Slack + PagerDuty** | Real-time team alerts and on-call escalation |

---

### Startup & Mid-Size Alternative Stack - Same System, Lower Cost

| What it does | Simpler Alternative | What You Save |
|---|---|---|
| Serving fraud predictions via API | **FastAPI** *(unchanged - it's already free and lightweight)* | - |
| Training and comparing models | **Scikit-learn + XGBoost** *(unchanged)* | - |
| Tracking experiments and model versions | **MLflow with SQLite** | No server to run - tracking stored in a local file |
| Storing models and data snapshots | **AWS S3** | No self-hosted storage - pay only for what you store |
| Scheduling and orchestrating pipelines | **Cron job** | No orchestration server - a scheduled Python script does the job |
| Monitoring data and model drift | **Evidently as a script** | No managed flow - run the same Evidently checks directly from a `.py` file |
| Explaining individual predictions | **SHAP** *(unchanged - it's already free)* | - |
| Storing transaction and prediction logs | **SQLite** | No database server - a local file handles early-stage volumes |
| Running all services consistently | **Docker** *(optional)* | Can run locally without containers at small scale |
| Provisioning cloud infrastructure | **Manual GCP Console setup** | Skip IaC entirely at early stage — create buckets and datasets by hand |
| Alerting | **Slack Webhook** | Free Slack alerts with no PagerDuty subscription needed |

The architecture was designed with this swap in mind from the start. The pipeline never depends directly on any specific tool - it depends on an interface, and the tool sits behind that interface. Swapping MinIO for S3, or Prefect for a cron job, is a one-file change. Nothing in the core fraud detection logic changes at all.

---

## Compliance & Audit Readiness

Every prediction is logged with:
- The exact model version that made it
- The confidence score
- The timestamp
- The transaction ID

Every training run is logged with:
- The dataset used
- All hyperparameters
- All evaluation metrics
- The git commit hash of the code

This means the system is audit-ready out of the box. Any regulator question about any decision at any point in time can be answered from the logs.

---

## Cloud Infrastructure (Terraform)

The GCP data landing zone — the GCS bucket and BigQuery dataset that feed the extraction pipeline — is fully provisioned via Terraform. Infrastructure is version-controlled, reproducible, and can be torn down and rebuilt in under a minute.

**What Terraform provisions:**

| Resource | Name | Purpose |
|---|---|---|
| GCS Bucket | `fraud-detection-489008-bucket` | Stores raw CSV uploads before BigQuery ingestion |
| GCS Object | `fraud-data/fraud_transactions.csv` | Uploads the transaction dataset directly from local path |
| BigQuery Dataset | `fraud_dataset` | Landing zone for transaction data queried by `extraction.py` |

**To provision the infrastructure:**

```bash
cd terraform/
terraform init
terraform apply
```

**To load the CSV from GCS into BigQuery after provisioning:**

```bash
bq load \
  --source_format=CSV \
  --skip_leading_rows=1 \
  --autodetect \
  fraud_dataset.transactions \
  gs://fraud-detection-489008-bucket/fraud-data/fraud_transactions.csv
```

Once loaded, the extraction pipeline can query `fraud_dataset.transactions` directly via the BigQuery extractor in `extraction.py` — no further configuration needed.

**Data flow through the infrastructure:**

```
fraud_transactions.csv
        ↓  (Terraform uploads)
GCS Bucket
        ↓  (bq load)
BigQuery — fraud_dataset.transactions
        ↓  (extraction.py BigQuery extractor)
Preprocessing → Training → MLflow → MinIO → /predict API
```

---

## Deployment

The entire system runs in Docker containers and can be deployed to any cloud provider (AWS, GCP, Azure) or on-premises environment. A single command starts all services:

```
docker-compose up -d
```

No specialised infrastructure required beyond a server that can run Docker.

---

## Roadmap

- [x] Real-time fraud scoring API
- [x] Automated daily drift monitoring
- [x] Automated weekly retraining
- [x] API authentication and rate limiting
- [x] Explainable predictions via SHAP
- [x] Full audit logging
- [x] Slack and PagerDuty alerting
- [x] 120+ test suite covering all critical failure modes
- [x] Cloud infrastructure provisioned via Terraform (GCS + BigQuery)
- [x] Load tested to 15,000 concurrent users via Locust
- [ ] Grafana dashboard for live fraud metrics
- [ ] CI/CD pipeline via GitHub Actions
- [ ] A/B testing framework for model comparison in production
- [ ] Simplified deployment config for smaller teams (S3 + cron)

---

## Frequently Asked Questions

**How do I know the model is still accurate?**
The daily monitoring job compares current transactions to the training baseline across three layers: data distribution, model output distribution, and real-world recall (via chargeback and dispute data). If any layer falls below threshold, the team is alerted before customers feel it.

**What happens if the new model performs worse than the current one?**
Nothing. The promotion logic requires the new model to outperform the current production model on ROC-AUC before it takes over. If it doesn't, the existing model stays in place and the team is notified.

**Can it handle our transaction volume?**
The scoring API is stateless and can be horizontally scaled behind a load balancer to handle any volume. The training and monitoring pipelines are completely separate from serving and don't impact API performance.

**What if we use a different database or cloud provider?**
The system is built around a port/adapter architecture - every external dependency (storage, database, data source) has a defined interface that can be swapped with a new implementation. Switching from MinIO to AWS S3, or from PostgreSQL to BigQuery, is a one-file change.

**Is it suitable for a team without an MLOps engineer?**
Yes. The system runs unattended. The only regular touchpoints are Slack alerts, which tell you what happened and whether any action is needed. Most of the time, the answer is "no action needed."

---

## About This Project

This system was designed and built as a complete production ML pipeline - not a proof of concept, not a research prototype. Every component - from data ingestion to model serving to monitoring - is production-ready, tested, and documented.

It reflects the kind of engineering that most ML teams build *after* their first silent model failure in production. This one was built right the first time.

---

*Built with Python 3.11 · Tested across 120+ scenarios · Infrastructure as Code via Terraform · Load tested to 15,000 concurrent users · Deployable in under 10 minutes*

---
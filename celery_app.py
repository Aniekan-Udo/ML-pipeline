"""
celery_app.py
-------------
Celery application for async task processing.

Architecture:
    RabbitMQ  — broker  (task queue, routes tasks to workers)
    Redis     — backend (stores task results and status)

Tasks:
    log_predictions — writes prediction records to PostgreSQL
                      fired via .delay() from /predict endpoint
                      so the API response is never blocked by DB writes

Environment variables (set in docker-compose.yml or .env):
    RABBITMQ_HOST     - defaults to localhost
    RABBITMQ_USER     - defaults to guest
    RABBITMQ_PASSWORD - defaults to guest
    REDIS_HOST        - defaults to localhost
    REDIS_PORT        - defaults to 6379
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import logging
from celery import Celery
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)

# ── Connection config ─────────────────────────────────────────────────────────

RABBITMQ_HOST     = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_USER     = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")
REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT        = os.getenv("REDIS_PORT", "6379")

BROKER_URL  = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:5672//"
BACKEND_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# ── Celery instance ───────────────────────────────────────────────────────────

celery = Celery(
    "ml_workflow",
    broker=BROKER_URL,
    backend=BACKEND_URL,
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,          # task acknowledged after completion, not on receipt
                                  # ensures no prediction is lost if worker crashes
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1, # one task at a time per worker — fair distribution
)


# ── Tasks ─────────────────────────────────────────────────────────────────────

@celery.task(
    name="log_predictions",
    bind=True,
    max_retries=3,
    default_retry_delay=5,        # retry after 5 seconds on failure
)
def log_predictions(
    self,
    transactions: list[dict],
    predictions: list,
    confidence_scores: list,
    model_name: str,
    model_version: str,
):
    """
    Async DB writer — logs prediction records to PostgreSQL.

    Fired via log_predictions.delay(...) from /predict endpoint.
    Never blocks the API response — runs in a separate Celery worker process.

    Retries up to 3 times on failure with 5 second delay.
    task_acks_late=True ensures no write is lost if the worker crashes mid-task.

    Parameters
    ----------
    transactions    : list of transaction dicts (from request.data)
    predictions     : list of int predictions (0 or 1)
    confidence_scores : list of float confidence scores
    model_name      : MLflow registered model name
    model_version   : MLflow model version
    """
    try:
        from database import get_engine, Prediction as PredictionRecord

        with Session(get_engine()) as session:
            for txn, pred, conf in zip(transactions, predictions, confidence_scores):
                session.add(PredictionRecord(
                    transaction_id=txn["transaction_id"],
                    predicted_label=int(pred),
                    confidence_score=float(conf),
                    model_name=model_name,
                    model_version=str(model_version),
                ))
            session.commit()

        logger.info(f"Logged {len(predictions)} predictions to DB")

    except Exception as e:
        logger.error(f"log_predictions task failed: {e}")
        raise self.retry(exc=e)   # retry up to max_retries times
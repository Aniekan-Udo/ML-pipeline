"""

locustfile.py — Load test for ML-Workflow Fraud Detection API

=============================================================

Endpoints tested:

  POST /predict   (weight 10 — dominant, core ML workload)

  GET  /health    (weight 1  — lightweight liveness probe)



Usage:

  # Web UI (recommended first run)

  locust -f locustfile.py --host http://localhost:8000



  # Headless — 15 k users, ramp 50/s, run 5 minutes, export CSV

  locust -f locustfile.py --headless -u 15000 -r 50 -t 5m \

         --host http://localhost:8000 --csv results/load_test



  # Distributed — 1 master + N workers (for 15k users use 3+ workers)

  locust -f locustfile.py --master

  locust -f locustfile.py --worker --master-host=<master-ip>

"""



import uuid

import random

import logging

from datetime import datetime, timedelta



from locust import HttpUser, task, between, events



logger = logging.getLogger(__name__)



# ── Realistic sample data pools ───────────────────────────────────────────────



TRANSACTION_TYPES = ["purchase", "withdrawal", "transfer", "refund", "payment"]

LOCATIONS         = ["Lagos", "Abuja", "Kano", "Port Harcourt", "Ibadan", "London", "New York"]

DEVICE_TYPES      = ["mobile", "desktop", "tablet", "pos_terminal"]



def _random_transaction() -> dict:

    """Generate a single realistic-looking transaction payload."""

    txn_time = datetime.utcnow() - timedelta(minutes=random.randint(0, 1440))

    return {

        "transaction_id":              str(uuid.uuid4()),

        "customer_id":                 f"CUST-{random.randint(1000, 99999)}",

        "transaction_amount":          round(random.uniform(10.0, 50000.0), 2),

        "transaction_type":            random.choice(TRANSACTION_TYPES),

        "transaction_time":            txn_time.isoformat(),

        "transaction_location":        random.choice(LOCATIONS),

        "device_type":                 random.choice(DEVICE_TYPES),

        "previous_transactions_count": random.randint(0, 500),

    }





def _predict_payload(batch_size: int = 1) -> dict:

    """Build a /predict request body with `batch_size` transactions."""

    return {"data": [_random_transaction() for _ in range(batch_size)]}





# ── User behaviour ────────────────────────────────────────────────────────────



class FraudDetectionUser(HttpUser):

    """

    Simulates a single API consumer of the fraud detection service.



    wait_time: each virtual user waits 1–3 s between tasks —

    realistic for a payment system sending async requests.

    Reduce to between(0.1, 0.5) for maximum throughput stress.

    """

    wait_time = between(1, 3)



    # ── /predict ──────────────────────────────────────────────────────────────



    @task(10)

    def predict_single(self):

        """Most common pattern: single transaction, real-time fraud check."""

        with self.client.post(

            "/predict",

            json=_predict_payload(batch_size=1),

            catch_response=True,

            name="/predict (single)",

        ) as resp:

            if resp.status_code == 200:

                body = resp.json()

                if "prediction" not in body:

                    resp.failure("Response missing 'prediction' field")

                elif "confidence" not in body:

                    resp.failure("Response missing 'confidence' field")

                else:

                    resp.success()

            elif resp.status_code == 503:

                # Model not loaded yet — don't count as a failure, just flag

                resp.failure("503 — model not loaded (startup warmup?)")

            else:

                resp.failure(f"Unexpected status {resp.status_code}: {resp.text[:200]}")



    @task(3)

    def predict_batch(self):

        """Batch scoring: 5 transactions at once (e.g. end-of-minute reconciliation)."""

        with self.client.post(

            "/predict",

            json=_predict_payload(batch_size=5),

            catch_response=True,

            name="/predict (batch-5)",

        ) as resp:

            if resp.status_code == 200:

                body = resp.json()

                preds = body.get("prediction", [])

                if len(preds) != 5:

                    resp.failure(f"Expected 5 predictions, got {len(preds)}")

                else:

                    resp.success()

            elif resp.status_code == 503:

                resp.failure("503 — model not loaded")

            else:

                resp.failure(f"Unexpected status {resp.status_code}")



    # ── /health ───────────────────────────────────────────────────────────────



    @task(1)

    def health_check(self):

        """Liveness probe — should always be fast (<50 ms)."""

        with self.client.get(

            "/health",

            catch_response=True,

            name="/health",

        ) as resp:

            if resp.status_code == 200:

                body = resp.json()

                if not body.get("model_loaded"):

                    resp.failure("Health OK but model_loaded=False")

                else:

                    resp.success()

            else:

                resp.failure(f"Health check failed: {resp.status_code}")



    # ── Startup warmup ────────────────────────────────────────────────────────



    def on_start(self):

        """

        Each virtual user fires a health check on spawn to confirm the

        service is up before hammering /predict.

        """

        resp = self.client.get("/health", name="/health (warmup)")

        if resp.status_code != 200:

            logger.warning(f"User {self.environment.runner} — warmup health check failed")





# ── CI/CD threshold hook ──────────────────────────────────────────────────────

# Fails the Locust process (exit code 1) if thresholds are breached.

# Remove or adjust thresholds for exploratory runs.



@events.quitting.add_listener

def _assert_thresholds(environment, **kwargs):

    stats = environment.runner.stats.total



    failures = [

        (stats.fail_ratio > 0.05,        f"Error rate {stats.fail_ratio:.1%} > 5%"),

        (stats.get_response_time_percentile(0.95) > 3000,

         f"p95 latency {stats.get_response_time_percentile(0.95):.0f}ms > 3000ms"),

    ]



    breached = [(ok, msg) for ok, msg in failures if ok]

    if breached:

        for _, msg in breached:

            logger.error(f"THRESHOLD BREACHED: {msg}")

        environment.process_exit_code = 1

    else:

        logger.info("All performance thresholds passed ✓")
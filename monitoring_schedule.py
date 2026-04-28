"""
monitoring_schedule.py
----------------------
Prefect deployment entrypoint for the daily drift monitoring schedule.
The full flow logic lives in monitoring.py. This file is the entry point
that prefect.yaml points to.
"""
from monitoring import daily_monitoring_flow  # noqa: F401 — re-exported as entrypoint

if __name__ == "__main__":
    daily_monitoring_flow.serve(
        name="daily-monitoring-deployment",
        cron="0 9 * * *",
    )
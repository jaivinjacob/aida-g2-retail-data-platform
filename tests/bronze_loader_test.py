import os
import sys
import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from databricks import sql

# Resolve project root so imports work from root or tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.simulator import OlistAPISimulator

# Load environment variables
env_path = find_dotenv(usecwd=True) or str(PROJECT_ROOT / ".env")
load_dotenv(env_path, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("bronze_loader_test")

CATALOG = "olist_dev"
SCHEMA = "bronze"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_landing"


def _get_databricks_config() -> tuple[str, str, str]:
    required_vars = ["DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {missing}")

    host = os.environ["DATABRICKS_HOST"].strip().rstrip("/")
    http_path = os.environ["DATABRICKS_HTTP_PATH"].strip()
    token = os.environ["DATABRICKS_TOKEN"].strip()
    return host, http_path, token


def upload_ndjson_direct(records: list[dict], target_volume_file: str, host: str, token: str):
    """Uploads NDJSON directly via Databricks Files REST API."""
    url = f"{host}/api/2.0/fs/files{target_volume_file}"
    headers = {"Authorization": f"Bearer {token}"}
    ndjson_bytes = "\n".join(json.dumps(rec, default=str) for rec in records).encode("utf-8")

    resp = requests.put(url, headers=headers, data=ndjson_bytes, params={"overwrite": "true"}, timeout=60)
    resp.raise_for_status()


def test_databricks_ingestion(test_start: str = "2017-11-24", test_end: str = "2017-11-24"):
    """End-to-end smoke test for simulator extraction, Volume staging, and Bronze Delta merge."""
    t0 = time.time()
    host, http_path, token = _get_databricks_config()

    logger.info("Step 1/5: Loading pre-indexed OlistAPISimulator...")
    sim = OlistAPISimulator(data_dir=PROJECT_ROOT / "olist_dataset")

    start_dt = datetime.strptime(test_start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(test_end, "%Y-%m-%d").date()
    window_days = (end_dt - start_dt).days + 1

    records = sim.get_orders_batch(target_date=test_start, window_days=window_days)
    assert len(records) > 0, f"Expected orders for {test_start}, got 0."
    logger.info(f"Step 2/5: Generated {len(records)} nested JSON records ({time.time() - t0:.1f}s)")

    target_volume_file = f"{VOLUME_PATH}/orders_batch_{test_start}_to_{test_end}.jsonl"
    logger.info(f"Step 3/5: Uploading NDJSON to {target_volume_file}...")
    upload_ndjson_direct(records, target_volume_file, host, token)
    logger.info(f"Upload complete ({time.time() - t0:.1f}s)")

    logger.info("Step 4/5: Connecting to Databricks SQL Warehouse...")
    with sql.connect(
        server_hostname=host.replace("https://", ""),
        http_path=http_path,
        access_token=token,
        _socket_timeout=60,
        retry_stop_after_attempts_count=2,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.ingestion_watermark (
                    pipeline_name STRING,
                    last_processed_date DATE,
                    records_loaded INT,
                    updated_at TIMESTAMP
                ) USING DELTA
            """)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.orders_raw_json (
                    order_id STRING,
                    batch_date DATE,
                    ingested_at TIMESTAMP,
                    raw_payload STRING
                ) USING DELTA
            """)

            logger.info("Step 5/5: Merging NDJSON into Bronze Delta table...")
            cursor.execute(f"""
                MERGE INTO {CATALOG}.{SCHEMA}.orders_raw_json AS target
                USING (
                    SELECT
                        order_id,
                        CAST('{test_end}' AS DATE) AS batch_date,
                        CURRENT_TIMESTAMP() AS ingested_at,
                        to_json(struct(*)) AS raw_payload
                    FROM read_files('{target_volume_file}', format => 'json')
                ) AS source
                ON target.order_id = source.order_id
                WHEN NOT MATCHED THEN INSERT *
            """)

            cursor.execute(f"""
                INSERT INTO {CATALOG}.{SCHEMA}.ingestion_watermark
                VALUES ('olist_orders_api', CAST('{test_end}' AS DATE), {len(records)}, CURRENT_TIMESTAMP())
            """)

            cursor.execute(f"SELECT COUNT(*) FROM {CATALOG}.{SCHEMA}.orders_raw_json")
            total_rows = cursor.fetchone()[0]
            assert total_rows >= len(records), f"Expected at least {len(records)} rows, found {total_rows}."
            logger.info(f"SUCCESS! Total rows in {CATALOG}.{SCHEMA}.orders_raw_json: {total_rows} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    test_databricks_ingestion()
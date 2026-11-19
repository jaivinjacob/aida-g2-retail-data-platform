import os
import io
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks import sql

# Load credentials from .env
load_dotenv()

# Configure Logging (Console + File)
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/ingestion.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

CATALOG = "olist_dev"
SCHEMA = "bronze"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_landing"


def get_sql_connection():
    return sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"]
    )


def initialize_bronze_tables(cursor):
    """Creates the watermark control table and raw Bronze Delta table if they don't exist."""
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


def get_next_batch_window(cursor, default_start="2016-09-04", initial_end="2017-12-31"):
    """
    Determines whether to run Initial Load or Incremental Load based on watermark.
    - If no watermark exists: loads initial baseline up to initial_end.
    - If watermark exists: loads the next 7 days (incremental batch).
    """
    cursor.execute(f"""
        SELECT MAX(last_processed_date)
        FROM {CATALOG}.{SCHEMA}.ingestion_watermark
        WHERE pipeline_name = 'olist_orders_api'
    """)
    row = cursor.fetchone()
    last_date = row[0] if row and row[0] else None

    if last_date is None:
        logger.info(f"No watermark found. Running INITIAL LOAD ({default_start} to {initial_end}).")
        start_dt = datetime.strptime(default_start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(initial_end, "%Y-%m-%d").date()
        window_days = (end_dt - start_dt).days + 1
        return str(start_dt), window_days, str(end_dt)
    else:
        next_start = last_date + timedelta(days=1)
        window_days = 7  # Incremental weekly batch
        next_end = next_start + timedelta(days=window_days - 1)
        logger.info(f"Watermark found ({last_date}). Running INCREMENTAL LOAD ({next_start} to {next_end}).")
        return str(next_start), window_days, str(next_end)


def upload_json_to_volume(records: list[dict], batch_label: str) -> str:
    """Uploads serialized JSON records directly to Databricks Unity Catalog Volume."""
    w = WorkspaceClient()
    file_name = f"orders_batch_{batch_label}.json"
    target_volume_file = f"{VOLUME_PATH}/{file_name}"

    json_bytes = json.dumps(records, default=str).encode("utf-8")
    w.files.upload(target_volume_file, io.BytesIO(json_bytes), overwrite=True)
    logger.info(f"Uploaded {len(records)} records to {target_volume_file}")
    return target_volume_file


def load_volume_to_bronze(cursor, volume_file_path: str, batch_end_date: str):
    """Reads staged JSON from Volume into Bronze Delta table and updates watermark."""
    # read_files() natively parses JSON arrays/lines from Unity Catalog Volumes
    cursor.execute(f"""
        MERGE INTO {CATALOG}.{SCHEMA}.orders_raw_json AS target
        USING (
            SELECT
                order_id,
                CAST('{batch_end_date}' AS DATE) AS batch_date,
                CURRENT_TIMESTAMP() AS ingested_at,
                to_json(struct(*)) AS raw_payload
            FROM read_files('{volume_file_path}', format => 'json')
        ) AS source
        ON target.order_id = source.order_id
        WHEN NOT MATCHED THEN INSERT *
    """)

    # Count records in the staged file for watermark logging
    cursor.execute(f"SELECT COUNT(*) FROM read_files('{volume_file_path}', format => 'json')")
    count = cursor.fetchone()[0]

    cursor.execute(f"""
        INSERT INTO {CATALOG}.{SCHEMA}.ingestion_watermark
        VALUES ('olist_orders_api', CAST('{batch_end_date}' AS DATE), {count}, CURRENT_TIMESTAMP())
    """)
    logger.info(f"Merged {count} records into {CATALOG}.{SCHEMA}.orders_raw_json and updated watermark to {batch_end_date}.")


def run_pipeline(simulator):
    try:
        with get_sql_connection() as conn:
            with conn.cursor() as cursor:
                initialize_bronze_tables(cursor)
                start_date, window_days, end_date = get_next_batch_window(cursor)

                # Fetch batch from your OlistAPISimulator (or future synthetic generator)
                records = simulator.get_orders_batch(target_date=start_date, window_days=window_days)

                if not records:
                    logger.warning(f"No orders found between {start_date} and {end_date}. Advancing watermark.")
                    cursor.execute(f"""
                        INSERT INTO {CATALOG}.{SCHEMA}.ingestion_watermark
                        VALUES ('olist_orders_api', CAST('{end_date}' AS DATE), 0, CURRENT_TIMESTAMP())
                    """)
                    return

                volume_file = upload_json_to_volume(records, batch_label=f"{start_date}_to_{end_date}")
                load_volume_to_bronze(cursor, volume_file, batch_end_date=end_date)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        from ingestion.simulator import OlistAPISimulator
    except ImportError:
        from simulator import OlistAPISimulator

    sim = OlistAPISimulator()
    run_pipeline(sim)
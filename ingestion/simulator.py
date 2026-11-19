import json
from datetime import timedelta
from pathlib import Path
import pandas as pd


class OlistAPISimulator:
    """Simulates a REST API serving batched Olist e-commerce records."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = self._resolve_data_dir(data_dir)
        self._load_and_index_datasets()

    @staticmethod
    def _resolve_data_dir(data_dir: str | Path | None) -> Path:
        if data_dir is not None:
            return Path(data_dir)
        cwd = Path.cwd()
        return cwd / "olist_dataset" if (cwd / "olist_dataset").exists() else cwd.parent / "olist_dataset"

    @staticmethod
    def _clean_records(df: pd.DataFrame) -> list[dict]:
        """Replaces pandas NaN/NaT with None for strict JSON compliance."""
        return df.where(pd.notna(df), None).to_dict(orient="records")

    def _load_and_index_datasets(self):
        """Loads CSVs and pre-indexes child tables into O(1) lookup maps."""
        self.orders = pd.read_csv(
            self.data_dir / "olist_orders_dataset.csv",
            parse_dates=["order_purchase_timestamp"],
        )

        # Pre-index customers by customer_id
        customers_df = pd.read_csv(self.data_dir / "olist_customers_dataset.csv")
        customers_clean = customers_df.where(pd.notna(customers_df), None)
        self._customer_map = customers_clean.set_index("customer_id").to_dict(orient="index")

        # Pre-group 1-to-many child tables by order_id for fast batch assembly
        self._items_map = self._group_by_order("olist_order_items_dataset.csv")
        self._payments_map = self._group_by_order("olist_order_payments_dataset.csv")
        self._reviews_map = self._group_by_order("olist_order_reviews_dataset.csv")

    def _group_by_order(self, filename: str) -> dict[str, list[dict]]:
        df = pd.read_csv(self.data_dir / filename)
        df = df.where(pd.notna(df), None)
        return {
            oid: group.drop(columns=["order_id"]).to_dict(orient="records")
            for oid, group in df.groupby("order_id", sort=False)
        }

    def get_orders_batch(self, target_date: str, window_days: int = 1) -> list[dict]:
        """
        Simulates GET /api/v1/orders?start_date=YYYY-MM-DD&days=N.
        Returns a list of nested order dictionaries within [target_date, target_date + window_days).
        """
        start_dt = pd.to_datetime(target_date)
        end_dt = start_dt + timedelta(days=window_days)

        mask = (
            (self.orders["order_purchase_timestamp"] >= start_dt)
            & (self.orders["order_purchase_timestamp"] < end_dt)
        )
        batch_orders = self.orders.loc[mask].where(pd.notna(self.orders.loc[mask]), None)

        if batch_orders.empty:
            return []

        api_response = []
        for row in batch_orders.itertuples(index=False):
            oid = row.order_id
            api_response.append({
                "order_id": oid,
                "order_status": row.order_status,
                "timestamps": {
                    "purchased_at": str(row.order_purchase_timestamp) if row.order_purchase_timestamp is not None else None,
                    "approved_at": str(row.order_approved_at) if row.order_approved_at is not None else None,
                    "delivered_carrier_at": str(row.order_delivered_carrier_date) if row.order_delivered_carrier_date is not None else None,
                    "delivered_customer_at": str(row.order_delivered_customer_date) if row.order_delivered_customer_date is not None else None,
                    "estimated_delivery_at": str(row.order_estimated_delivery_date) if row.order_estimated_delivery_date is not None else None,
                },
                "customer": self._customer_map.get(row.customer_id),
                "items": self._items_map.get(oid, []),
                "payments": self._payments_map.get(oid, []),
                "reviews": self._reviews_map.get(oid, []),
            })

        return api_response

    def get_catalog_reference(self) -> dict[str, list[dict]]:
        """Simulates GET /api/v1/catalog for static dimension tables."""
        products = pd.read_csv(self.data_dir / "olist_products_dataset.csv")
        translations = pd.read_csv(self.data_dir / "product_category_name_translation.csv")
        sellers = pd.read_csv(self.data_dir / "olist_sellers_dataset.csv")
        geo = pd.read_csv(self.data_dir / "olist_geolocation_dataset.csv")

        extra_translations = pd.DataFrame([
            {"product_category_name": "pc_gamer", "product_category_name_english": "pc_gamer"},
            {
                "product_category_name": "portateis_cozinha_e_preparadores_de_alimentos",
                "product_category_name_english": "kitchen_appliances_and_food_preparators",
            },
        ])
        full_translations = pd.concat([translations, extra_translations], ignore_index=True).drop_duplicates("product_category_name")
        products_enriched = products.merge(full_translations, on="product_category_name", how="left")

        geo_deduped = (
            geo.groupby("geolocation_zip_code_prefix", as_index=False)
            .agg(
                geolocation_lat=("geolocation_lat", "mean"),
                geolocation_lng=("geolocation_lng", "mean"),
                geolocation_city=("geolocation_city", "first"),
                geolocation_state=("geolocation_state", "first"),
            )
        )

        return {
            "products": self._clean_records(products_enriched),
            "sellers": self._clean_records(sellers),
            "geolocation": self._clean_records(geo_deduped),
        }


if __name__ == "__main__":
    sim = OlistAPISimulator()
    sample_orders = sim.get_orders_batch(target_date="2017-11-24", window_days=1)
    print(f"Fetched {len(sample_orders)} orders for 2017-11-24")
    if sample_orders:
        print(json.dumps(sample_orders[0], indent=2))
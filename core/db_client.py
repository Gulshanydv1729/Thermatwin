"""
InfluxDB query client for serving live SCADA data to the UI.

Provides:
  - get_historical_thermal_data(days=30) → long-term temperature decay
  - get_live_dyno_card() → most recent stroke cycle (load & position arrays)
"""

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient

# ── Configuration ──────────────────────────────────────────────────────────

INFLUX_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN", "thermatwin-token")
INFLUX_ORG = os.getenv("INFLUXDB_ORG", "thermatwin")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "scada_live")

_client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG,
)
_query_api = _client.query_api()


def get_historical_thermal_data(days: int = 30) -> pd.DataFrame:
    """
    Fetch the long-term temperature decay curve from InfluxDB.

    Returns DataFrame with columns: [time, wellhead_temp_c]
    """
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: -{days}d)
        |> filter(fn: (r) => r._measurement == "sensors")
        |> filter(fn: (r) => r._field == "wellhead_temp_c")
        |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
        |> yield(name: "mean")
    '''
    result = _query_api.query_data_frame(query)

    if result is None or result.empty:
        return pd.DataFrame(columns=["time", "wellhead_temp_c"])

    if isinstance(result, list):
        result = pd.concat(result, ignore_index=True)

    df = result[["_time", "_value"]].rename(columns={"_time": "time", "_value": "wellhead_temp_c"})
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_live_dyno_card() -> tuple[np.ndarray, np.ndarray] | None:
    """
    Fetch the most recent stroke cycle data (load and position) as NumPy arrays.

    Returns:
        (load_array, position_array) or None if no data available.
    """
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: -1m)
        |> filter(fn: (r) => r._measurement == "dyno_card")
        |> filter(fn: (r) => r._field == "load_n" or r._field == "pos_m")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> sort(columns: ["_time"], desc: true)
        |> limit(n: 200)
    '''
    result = _query_api.query_data_frame(query)

    if result is None or result.empty:
        return None

    if isinstance(result, list):
        result = pd.concat(result, ignore_index=True)

    result = result.sort_values("_time")

    load = result["load_n"].dropna().values
    pos = result["pos_m"].dropna().values

    if len(load) == 0 or len(pos) == 0:
        return None

    return load, pos


def get_latest_sensors() -> dict:
    """
    Fetch the most recent sensor readings.

    Returns:
        dict with keys: wellhead_temp_c, vfd_hz, timestamp
    """
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: -5m)
        |> filter(fn: (r) => r._measurement == "sensors")
        |> filter(fn: (r) => r._field == "wellhead_temp_c" or r._field == "vfd_hz")
        |> last()
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    result = _query_api.query_data_frame(query)

    if result is None or result.empty:
        return {}

    if isinstance(result, list):
        result = pd.concat(result, ignore_index=True)

    row = result.iloc[0]
    return {
        "wellhead_temp_c": float(row.get("wellhead_temp_c", 0)),
        "vfd_hz": float(row.get("vfd_hz", 0)),
        "timestamp": str(row.get("_time", "")),
    }

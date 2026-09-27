"""
MQTT → InfluxDB ingestion worker for live SCADA data.

Subscribes to:
  - baghewala/well_01/dyno/    (surface load & position arrays)
  - baghewala/well_01/sensors/ (wellhead temp, VFD Hz)

Writes to InfluxDB measurements:
  - dyno_card   (load_n, pos_m)
  - sensors     (wellhead_temp_c, vfd_hz)

Retries MQTT connection with exponential backoff.
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingestion_worker")

# ── Configuration ──────────────────────────────────────────────────────────

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPICS = [
    ("baghewala/well_01/dyno/", 1),
    ("baghewala/well_01/sensors/", 1),
]

INFLUX_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN", "thermatwin-token")
INFLUX_ORG = os.getenv("INFLUXDB_ORG", "thermatwin")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "scada_live")

RETRY_DELAY_INITIAL = 5  # seconds
RETRY_DELAY_MAX = 60  # seconds

# ── InfluxDB client ────────────────────────────────────────────────────────

influx_client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG,
)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)


def _write_dyno(payload: dict) -> None:
    """Write a dynamometer card data point to InfluxDB."""
    load = payload.get("load_n")
    pos = payload.get("pos_m")
    ts = payload.get("timestamp")

    point = Point("dyno_card").tag("well_id", "well_01")
    if load is not None:
        point = point.field("load_n", float(load))
    if pos is not None:
        point = point.field("pos_m", float(pos))
    if ts:
        point = point.time(ts)

    write_api.write(bucket=INFLUX_BUCKET, record=point)
    logger.debug(f"Wrote dyno point: load={load}, pos={pos}")


def _write_sensors(payload: dict) -> None:
    """Write sensor readings to InfluxDB."""
    temp = payload.get("wellhead_temp_c")
    vfd_hz = payload.get("vfd_hz")
    ts = payload.get("timestamp")

    point = Point("sensors").tag("well_id", "well_01")
    if temp is not None:
        point = point.field("wellhead_temp_c", float(temp))
    if vfd_hz is not None:
        point = point.field("vfd_hz", float(vfd_hz))
    if ts:
        point = point.time(ts)

    write_api.write(bucket=INFLUX_BUCKET, record=point)
    logger.debug(f"Wrote sensor point: temp={temp}, vfd_hz={vfd_hz}")


# ── MQTT callbacks ─────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc, properties=None):
    logger.info(f"Connected to MQTT broker at {MQTT_HOST}:{MQTT_PORT} (rc={rc})")
    for topic, qos in MQTT_TOPICS:
        client.subscribe(topic, qos=qos)
        logger.info(f"Subscribed to {topic}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to parse payload on {msg.topic}: {e}")
        return

    topic = msg.topic
    if topic.startswith("baghewala/well_01/dyno/"):
        _write_dyno(payload)
    elif topic.startswith("baghewala/well_01/sensors/"):
        _write_sensors(payload)
    else:
        logger.debug(f"Ignored message on unknown topic: {topic}")


def on_disconnect(client, userdata, rc, properties=None):
    logger.warning(f"Disconnected from MQTT broker (rc={rc})")


# ── Main loop with retry ───────────────────────────────────────────────────

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    # Graceful shutdown
    shutdown_requested = False

    def shutdown(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        logger.info("Shutting down ingestion worker...")
        client.disconnect()
        influx_client.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("Starting ingestion worker...")

    retry_delay = RETRY_DELAY_INITIAL
    while not shutdown_requested:
        try:
            client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            retry_delay = RETRY_DELAY_INITIAL  # Reset on successful connect
            client.loop_forever(retry_first_connection=True)
        except (ConnectionRefusedError, OSError) as e:
            if shutdown_requested:
                break
            logger.warning(f"MQTT connection failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, RETRY_DELAY_MAX)
        except Exception as e:
            if shutdown_requested:
                break
            logger.error(f"Unexpected error: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, RETRY_DELAY_MAX)


if __name__ == "__main__":
    main()

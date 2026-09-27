"""MQTT consumer — subscribes to SCADA topics and forwards to Kafka."""

import json
import logging

import paho.mqtt.client as mqtt

from backend.config import settings

logger = logging.getLogger(__name__)


def on_connect(client, userdata, flags, rc):
    logger.info(f"MQTT connected with result code {rc}")
    client.subscribe("scada/+/+")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        logger.debug(f"Received: {msg.topic} — {payload}")
        # Forward to Kafka (placeholder — Kafka producer injected in production)
        _forward_to_kafka(msg.topic, payload)
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")


def _forward_to_kafka(topic: str, payload: dict):
    """Forward parsed payload to Kafka scada.raw topic."""
    # In production, this uses aiokafka.AIOKafkaProducer
    logger.debug(f"Would forward to Kafka: {topic}")


def start_consumer():
    """Start the MQTT consumer (blocking)."""
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(settings.mqtt_host, settings.mqtt_port, 60)
    logger.info(f"MQTT consumer started, connecting to {settings.mqtt_host}:{settings.mqtt_port}")
    client.loop_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_consumer()

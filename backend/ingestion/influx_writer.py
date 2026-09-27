"""InfluxDB async writer for high-frequency VFD telemetry."""

import logging

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS

from backend.config import settings

logger = logging.getLogger(__name__)


class InfluxWriter:
    """Async InfluxDB writer for VFD high-frequency metrics."""

    def __init__(self):
        self.client = InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org,
        )
        self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)

    async def write_vfd_point(
        self,
        well_id: str,
        frequency_hz: float,
        current_a: float,
        voltage_v: float,
        power_kw: float | None = None,
        torque_nm: float | None = None,
    ):
        """Write a single VFD telemetry point."""
        point = (
            Point("vfd_telemetry")
            .tag("well_id", str(well_id))
            .field("frequency_hz", frequency_hz)
            .field("current_a", current_a)
            .field("voltage_v", voltage_v)
        )
        if power_kw is not None:
            point = point.field("power_kw", power_kw)
        if torque_nm is not None:
            point = point.field("torque_nm", torque_nm)

        self.write_api.write(bucket=settings.influxdb_bucket, record=point)

    def close(self):
        self.client.close()

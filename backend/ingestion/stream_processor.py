"""Kafka stream processor — consumes scada.raw, produces scada.processed."""

import json
import logging

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Processes raw SCADA messages from Kafka.
    
    Applies:
    - Unit conversion (psi → kPa, °F → °C)
    - Outlier filtering (load > 200 kN → drop)
    - Timestamp normalization
    """

    def process(self, raw_message: dict) -> dict | None:
        """Process a single raw SCADA message. Returns None if message should be dropped."""
        try:
            processed = {
                "time": raw_message.get("timestamp"),
                "well_id": raw_message.get("well_id"),
                "cycle_id": raw_message.get("cycle_id"),
                "surface_load_n": self._convert_load(raw_message.get("surface_load_lbf")),
                "surface_pos_m": raw_message.get("surface_pos_m"),
                "spm": raw_message.get("spm"),
                "vfd_frequency_hz": raw_message.get("vfd_freq_hz"),
                "motor_current_a": raw_message.get("motor_current_a"),
                "motor_voltage_v": raw_message.get("motor_voltage_v"),
                "wellhead_temp_c": self._convert_temp(raw_message.get("wellhead_temp_f")),
                "wellhead_pressure_kpa": self._convert_pressure(raw_message.get("wellhead_pressure_psi")),
            }

            # Outlier filter: drop if surface load exceeds 200 kN
            if processed["surface_load_n"] is not None and processed["surface_load_n"] > 200000:
                logger.warning(f"Outlier dropped: load={processed['surface_load_n']} N")
                return None

            return processed
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            return None

    def _convert_load(self, lbf: float | None) -> float | None:
        """Convert lbf to N."""
        if lbf is None:
            return None
        return lbf * 4.44822

    def _convert_temp(self, f: float | None) -> float | None:
        """Convert °F to °C."""
        if f is None:
            return None
        return (f - 32) * 5.0 / 9.0

    def _convert_pressure(self, psi: float | None) -> float | None:
        """Convert psi to kPa."""
        if psi is None:
            return None
        return psi * 6.89476

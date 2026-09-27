"""Environment-based configuration using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "thermatwin"
    postgres_password: str = "thermatwin"
    postgres_db: str = "thermatwin"

    # TimescaleDB
    timescaledb_host: str = "localhost"
    timescaledb_port: int = 5433
    timescaledb_user: str = "thermatwin"
    timescaledb_password: str = "thermatwin"
    timescaledb_db: str = "thermatwin_ts"

    # InfluxDB
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = "thermatwin-token"
    influxdb_org: str = "thermatwin"
    influxdb_bucket: str = "vfd_telemetry"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"

    # MQTT
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def timescaledb_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.timescaledb_user}:{self.timescaledb_password}"
            f"@{self.timescaledb_host}:{self.timescaledb_port}/{self.timescaledb_db}"
        )

    model_config = {"env_prefix": "THERMATWIN_"}


settings = Settings()

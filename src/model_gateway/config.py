import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    environment: str
    http_port: int
    grpc_port: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            environment=os.environ.get("CHORA_ENVIRONMENT", "development"),
            http_port=int(os.environ.get("CHORA_HTTP_PORT", "8080")),
            grpc_port=int(os.environ.get("CHORA_GRPC_PORT", "9090")),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        )

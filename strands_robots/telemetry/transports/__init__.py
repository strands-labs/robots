"""Telemetry transport backends."""

from strands_robots.telemetry.transports.local import LocalWALTransport
from strands_robots.telemetry.transports.stdout import StdoutTransport

__all__ = ["LocalWALTransport", "StdoutTransport"]

# OTel transport is optional (requires opentelemetry SDK)
try:
    from strands_robots.telemetry.transports.otel import OTelTransport  # noqa: F401

    __all__.append("OTelTransport")
except ImportError:
    pass

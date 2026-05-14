"""Frontend configuration (paths, colours, network endpoints)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    # Networking
    api_base_url: str = field(
        default_factory=lambda: os.getenv("API_BASE_URL", "http://localhost:8000")
    )
    request_timeout_s: int = 120

    # Local fallback: if the API is unreachable, run the model in-process.
    use_local_inference_fallback: bool = True

    # Window
    app_name: str = "Insole Foot Classification"
    app_version: str = "0.1.0"
    initial_width: int = 1440
    initial_height: int = 920

    # Theming
    use_dark_mode: bool = True

    # Data folder shown in the Training tab.
    default_data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "data"))
    )

    # Where uploaded images are stashed when the user picks them.
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".insole_app_cache")


APP_CONFIG = AppConfig()
APP_CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)

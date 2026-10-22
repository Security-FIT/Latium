"""Only data locations: dependency installation belongs to the small notebook."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Environment:
    work_dir: Path
    variant: str = "workshop"

    @classmethod
    def discover(cls, variant="workshop"):
        default = Path("/content/rome-workshop") if Path("/content").is_dir() else Path.home() / "rome-workshop"
        return cls(Path(os.environ.get("LATIUM_WORK_DIR", str(default))).expanduser().resolve(), variant)

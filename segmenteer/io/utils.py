from pathlib import Path
from datetime import datetime


def create_timestamped_output_dir(base_dir: str = "outputs") -> Path:
    timestamp = datetime.now().strftime("%d%m_%H%M")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
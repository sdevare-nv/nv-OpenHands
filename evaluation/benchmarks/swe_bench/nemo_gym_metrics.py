import json
import os
import uuid
from pathlib import Path
from typing import Mapping

from openhands.core.logger import openhands_logger as logger

MetricValue = float | int | str | bool | None


def update_nemo_gym_metrics(
    updates: Mapping[str, MetricValue],
    *,
    metrics_path: Path | None = None,
) -> bool:
    path = metrics_path
    if path is None:
        raw_path = os.environ.get('NEMO_GYM_METRICS_FPATH')
        if not raw_path:
            return False
        path = Path(raw_path)

    temporary: Path | None = None
    published = False
    try:
        current = json.loads(path.read_text() or '{}') if path.exists() else {}
        if not isinstance(current, dict):
            raise TypeError('metrics file must contain a JSON object')

        merged = {key: value for key, value in current.items() if value is not None}
        merged.update({key: value for key, value in updates.items() if value is not None})
        temporary = path.with_name(
            f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp'
        )
        temporary.write_text(json.dumps(merged, sort_keys=True))
        os.replace(temporary, path)
        published = True
        return True
    except (json.JSONDecodeError, OSError, TypeError, UnicodeDecodeError) as error:
        logger.warning('Could not update NeMo Gym metrics at %s: %s', path, error)
        return False
    finally:
        if temporary is not None and not published:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

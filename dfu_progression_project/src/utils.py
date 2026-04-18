from datetime import datetime
from zoneinfo import ZoneInfo


def generate_run_id(zone: ZoneInfo = ZoneInfo("UTC")) -> str:
    """Generate a unique run ID using current UTC date and time.

    Args:
        zone (ZoneInfo, optional): Timezone information. Defaults to UTC.

    Returns:
        str: A unique run ID in the format 'run-YYYY-MM-DD-HH-MM-SS'.
    """
    current_time = datetime.now(zone)
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    return f"run-{formatted_time}"

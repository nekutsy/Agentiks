from datetime import datetime

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Optional datetime format (default: ISO format)",
                    "enum": ["iso", "datetime", "date", "time"]
                }
            }
        }
    }
}

def execute(format: str = "iso"):
    now = datetime.now()
    if format == "iso":
        return now.isoformat()
    elif format == "datetime":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif format == "date":
        return now.strftime("%Y-%m-%d")
    elif format == "time":
        return now.strftime("%H:%M:%S")
    else:
        return now.isoformat()
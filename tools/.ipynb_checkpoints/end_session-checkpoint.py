TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "end_session",
        "description": "End the current session successfully. After calling this, no further messages will be processed.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}

def execute():
    return "__END_SESSION__"
import tool_loader

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "reload_tools",
        "description": "Reload all tools from the tools directory. Use this after adding, modifying, or removing tool files.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}

def execute():
    result = tool_loader.reload_tools()
    return f"Tools reloaded: {', '.join(result['tools'])}"
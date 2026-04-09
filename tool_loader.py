import importlib
import sys
from pathlib import Path
from typing import Dict, List, Any, Callable
from logger_setup import get_logger

TOOLS_DIR = Path(__file__).parent / "tools"
AVAILABLE_TOOLS = {}  # name -> {'definition': dict, 'execute': callable}

def load_tools():
    global AVAILABLE_TOOLS
    AVAILABLE_TOOLS.clear()
    
    if not TOOLS_DIR.exists():
        TOOLS_DIR.mkdir()
    
    for file in TOOLS_DIR.glob("*.py"):
        if file.name == "__init__.py":
            continue
        
        module_name = file.stem
        try:
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            if hasattr(module, 'TOOL_DEFINITION') and hasattr(module, 'execute'):
                tool_def = module.TOOL_DEFINITION
                tool_name = tool_def.get('function', {}).get('name')
                if tool_name:
                    AVAILABLE_TOOLS[tool_name] = {
                        'definition': tool_def,
                        'execute': module.execute
                    }
                    get_logger().info(f"Loaded tool: {tool_name}")
                else:
                    get_logger().warning(f"Tool {module_name}: missing name in TOOL_DEFINITION")
            else:
                get_logger().warning(f"Tool {module_name}: missing TOOL_DEFINITION or execute function")
        except Exception as e:
            get_logger().error(f"Failed to load tool {module_name}: {e}")

def reload_tools():
    global AVAILABLE_TOOLS
    for tool_name in list(AVAILABLE_TOOLS.keys()):
        for mod_name, mod in list(sys.modules.items()):
            if hasattr(mod, 'TOOL_DEFINITION'):
                mod_def = getattr(mod, 'TOOL_DEFINITION', {})
                if mod_def.get('function', {}).get('name') == tool_name:
                    del sys.modules[mod_name]
                    break
    AVAILABLE_TOOLS.clear()
    load_tools()
    get_logger().info(f"Tools reloaded. Available: {list(AVAILABLE_TOOLS.keys())}")
    return {"status": "reloaded", "tools": list(AVAILABLE_TOOLS.keys())}

def get_tools_for_ollama():
    return [info['definition'] for info in AVAILABLE_TOOLS.values()]

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    if tool_name not in AVAILABLE_TOOLS:
        return f"Error: Tool '{tool_name}' not found"
    
    try:
        result = AVAILABLE_TOOLS[tool_name]['execute'](**arguments)
        return str(result)
    except Exception as e:
        get_logger().error(f"Tool {tool_name} execution error: {e}")
        return f"Error executing tool {tool_name}: {e}"
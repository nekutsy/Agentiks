import subprocess
import tempfile
import os
import sys
from config import EXEC_TEMP_DIR

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute Python code and return its output (stdout and stderr). The code runs in a separate process with a timeout.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default: 5).",
                    "default": 5
                }
            },
            "required": ["code"]
        }
    }
}

def execute(code: str, timeout: int = 5) -> str:
    os.makedirs(EXEC_TEMP_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', dir=EXEC_TEMP_DIR) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=EXEC_TEMP_DIR
        )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append("STDERR:\n" + result.stderr)
        if result.returncode != 0 and not result.stderr:
            output_parts.append(f"Process exited with code {result.returncode}")

        return "\n".join(output_parts) if output_parts else "[no output]"

    except subprocess.TimeoutExpired:
        return f"Timeout: code execution exceeded {timeout} seconds."
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass
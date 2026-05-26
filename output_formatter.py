import json
import textwrap
from typing import Any, Dict, List


def format_argument_value(value: Any, indent: int = 2) -> str:
    if isinstance(value, str) and '\n' in value:
        return '\n'.join(' ' * indent + line for line in value.splitlines())
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=indent, ensure_ascii=False)
    return str(value)


def print_colorized_text(text: str, color_code: str = '\033[38;5;208m'):
    if not text:
        return
    lines = text.splitlines()
    if len(lines) <= 1:
        print(f"{color_code}{text}\033[0m")
    else:
        print(f"{color_code}{textwrap.indent(text, '  ')}\033[0m")


def print_tool_call(tool_name: str, arguments: dict):
    print(f"Calling tool: \033[96m{tool_name}\033[0m")
    if arguments:
        out = []
        for k, v in arguments.items():
            out.append(f"  {k}: {format_argument_value(v, indent=4)}")
        print_colorized_text('\n'.join(out), '\033[38;5;208m')
    else:
        print_colorized_text("{}", '\033[38;5;208m')
    print()


def print_tool_calls(tool_calls: List[Dict]):
    if not tool_calls:
        return
    print("\033[96m=== Tool Calls ===\033[0m")
    for i, tc in enumerate(tool_calls, 1):
        func = tc.get('function', {})
        name = func.get('name', 'unknown')
        args = func.get('arguments', {}) or {}
        print(f"\033[96mTool {i}: {name}\033[0m")
        if args:
            out = []
            for k, v in args.items():
                out.append(f"  {k}: {format_argument_value(v, indent=4)}")
            print_colorized_text('\n'.join(out), '\033[38;5;208m')
        else:
            print_colorized_text("{}", '\033[38;5;208m')
        if i < len(tool_calls):
            print()
    print("\033[96m=================\033[0m\n")


def print_tool_result(tool_name: str, result: str):
    print(f"\033[92mTool result ({tool_name}):\033[0m")
    r = str(result)
    if '\n' in r:
        print(f"\033[92m{textwrap.indent(r, '  ')}\033[0m")
    else:
        print(f"\033[92m{r}\033[0m")
    print()
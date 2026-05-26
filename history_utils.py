import json
from typing import List, Dict, Optional

from config import MAX_HISTORY_MESSAGES, LOG_CURRENT_INPUT, THINKING_HISTORY_MODE
from background_io import BackgroundIOManager


def build_initial_history(system_prompt: str, user_first: str) -> List[Dict]:
    history = []
    if system_prompt:
        history.append({'role': 'system', 'content': system_prompt})
    if user_first:
        history.append({'role': 'user', 'content': user_first})
    return history


def prune_history(history: List[Dict], max_messages: int = MAX_HISTORY_MESSAGES) -> List[Dict]:
    if len(history) <= max_messages + 1:
        return history

    pruned: List[Dict] = []
    if history and history[0].get('role') == 'system':
        pruned.append(history[0])
        rest = history[1:]
    else:
        rest = history

    if len(rest) <= max_messages:
        return pruned + rest

    old_part = rest[:-max_messages]
    recent_part = rest[-max_messages:]

    for msg in old_part:
        if msg.get('role') == 'tool':
            pruned.append({
                'role': 'tool',
                'tool_call_id': msg.get('tool_call_id', 'unknown'),
                'content': '[Old tool result omitted to save context]',
            })
        else:
            pruned.append(msg)
    return pruned + recent_part


def prepare_history_for_api(history: List[Dict], mode: str = THINKING_HISTORY_MODE) -> List[Dict]:
    formatted = []
    for i, msg in enumerate(history):
        new_msg = {k: v for k, v in msg.items() if k != 'thinking'}
        if msg.get('role') == 'assistant' and msg.get('thinking'):
            thinking = msg['thinking']
            content = msg.get('content', '')
            if mode == "all" or (mode == "last" and i == len(history) - 1):
                if content:
                    new_msg['content'] = f"<think>{thinking}</think>\n\n{content}"
                else:
                    new_msg['content'] = f"<think>{thinking}</think>"
        formatted.append(new_msg)
    return formatted


def build_dump_text(messages: List[Dict], session_num: int) -> str:
    input_text = f"=== Session {session_num} ===\n"
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        if role == 'system':
            input_text += f"[SYSTEM]\n{content}\n"
        elif role == 'user':
            input_text += f"[USER]\n{content}\n"
        elif role == 'assistant':
            if content:
                input_text += f"[ASSISTANT]\n{content}\n"
            if 'tool_calls' in msg:
                input_text += f"[TOOL_CALLS]\n{json.dumps(msg['tool_calls'], indent=2)}\n"
        elif role == 'tool':
            input_text += f"[TOOL_RESULT] {msg.get('tool_call_id')}\n{msg.get('content', '')}\n"
        else:
            input_text += f"[{role.upper()}]\n{content}\n"
    input_text += "=== END INPUT ===\n"
    return input_text


def write_current_input_async(io_manager: BackgroundIOManager, messages: List[Dict], session_num: int):
    if io_manager is None:
        return
    snapshot = [dict(m) for m in messages]
    text = build_dump_text(snapshot, session_num)
    io_manager.write_file(LOG_CURRENT_INPUT, text)
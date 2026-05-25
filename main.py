import json
import signal
import sys
import textwrap
from typing import Optional, List, Dict, Any

import ollama

from config import (
    MAX_SESSION_NUM, MODEL_NAME, OLLAMA_OPTIONS, STREAM_MESSAGES, LOG_THINKING,
    SYSTEM_PROMPT_FILE, USER_FIRST_MESSAGE_FILE, LOG_CURRENT_INPUT, THINKING_HISTORY_MODE,
    MAX_HISTORY_MESSAGES, MAX_TOOL_RESULT_LENGTH,
    OLLAMA_NUM_CTX, OLLAMA_NUM_BATCH, OLLAMA_NUM_GPU, OLLAMA_NUM_THREAD,
    OLLAMA_FLASH_ATTENTION, OLLAMA_KEEP_ALIVE,
)
from logger_setup import get_logger
from session_manager import SessionManager
import tool_loader
from tool_loader import get_tools_for_ollama, execute_tool
from message_generator import UserMessageGenerator
from background_io import BackgroundIOManager
from profiler import get_profiler

running = True
current_session_mgr: Optional[SessionManager] = None
io_manager: Optional[BackgroundIOManager] = None


def save_metrics_and_exit():
    """Сохраняет метрики и завершает программу."""
    global running
    running = False
    prof = get_profiler()
    try:
        prof.update_aggregated_metrics(MODEL_NAME)
    except Exception as e:
        get_logger().error(f"Failed to save metrics: {e}")
    if io_manager is not None:
        io_manager.shutdown(timeout=5.0)
    prof.report()
    sys.exit(0)


def signal_handler(sig, frame):
    get_logger().info("Received interrupt signal, shutting down gracefully...")
    if current_session_mgr and current_session_mgr.current_session:
        get_logger().info(
            f"Session {current_session_mgr.current_session['number']} interrupted, marking as completed."
        )
        current_session_mgr.complete_current_session()
    save_metrics_and_exit()


def build_ollama_options() -> dict:
    defaults = {
        'num_ctx': OLLAMA_NUM_CTX,
        'num_batch': OLLAMA_NUM_BATCH,
        'num_gpu': OLLAMA_NUM_GPU,
        'num_thread': OLLAMA_NUM_THREAD,
        'flash_attention': OLLAMA_FLASH_ATTENTION,
        'use_mlock': True,
    }
    merged = {**defaults, **(OLLAMA_OPTIONS or {})}
    if merged.get('num_thread') in (0, None):
        merged.pop('num_thread', None)
    return merged


OLLAMA_MERGED_OPTIONS = None


def get_ollama_options() -> dict:
    global OLLAMA_MERGED_OPTIONS
    if OLLAMA_MERGED_OPTIONS is None:
        OLLAMA_MERGED_OPTIONS = build_ollama_options()
        get_logger().info(f"Ollama options: {OLLAMA_MERGED_OPTIONS}")
        get_logger().info(f"Ollama keep_alive: {OLLAMA_KEEP_ALIVE}")
    return OLLAMA_MERGED_OPTIONS


def load_prompt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        get_logger().warning(f"Prompt file not found: {file_path}")
        return ""


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


def write_current_input_async(messages: List[Dict], session_num: int):
    if io_manager is None:
        return
    snapshot = [dict(m) for m in messages]
    text = build_dump_text(snapshot, session_num)
    io_manager.write_file(LOG_CURRENT_INPUT, text)


def log_message(session_num: int, msg_num: int, role: str, content: str = None, tool_calls: List = None):
    get_logger().info(f"Session {session_num}, msg {msg_num}: {role} message")


def normalize_tool_call(tc) -> Dict:
    if isinstance(tc, dict):
        return tc
    if hasattr(tc, 'model_dump'):
        return tc.model_dump()
    if hasattr(tc, 'dict'):
        return tc.dict()
    return {"raw": str(tc)}


def extract_thinking(msg: Dict) -> Optional[str]:
    for key in ('thinking', 'reasoning', 'reasoning_content'):
        if key in msg and msg[key]:
            return msg[key]
    return None


def _safe_parse_args(args: Any) -> Any:
    if args is None:
        return {}
    if isinstance(args, (dict, list)):
        return args
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            get_logger().warning(f"Failed to parse tool args: {s[:120]!r}")
            return {"_raw": args}
    return args


def finalize_tool_calls(raw_tool_calls: List[Dict]) -> List[Dict]:
    finalized = []
    for i, tc in enumerate(raw_tool_calls):
        func = tc.get('function', {}) or {}
        args = _safe_parse_args(func.get('arguments', ''))
        finalized.append({
            'id': tc.get('id') or f'tc_{i}',
            'type': 'function',
            'function': {
                'name': func.get('name', ''),
                'arguments': args,
            },
        })
    return finalized


def prepare_history_for_api(history: List[Dict], mode: str) -> List[Dict]:
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


def stream_chat(messages: List[Dict], session_num: int, tools=None):
    log = get_logger()
    prof = get_profiler()

    full_content = ""
    full_thinking = ""
    raw_tool_calls: List[Dict] = []
    token_count = 0

    try:
        with prof.measure('ollama_stream_total'):
            stream = ollama.chat(
                model=MODEL_NAME, messages=messages,
                options=get_ollama_options(),
                stream=True, tools=tools,
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
            print(f"\n--- Session {session_num}, generating response ---")

            for chunk in stream:
                prof.record_ollama_metrics(chunk)

                if 'message' not in chunk:
                    continue

                msg = chunk['message']

                if msg.get('content'):
                    print(msg['content'], end='', flush=True)
                    full_content += msg['content']

                if LOG_THINKING:
                    thinking_chunk = extract_thinking(msg)
                    if thinking_chunk:
                        print(f"\033[94m{thinking_chunk}\033[0m", end='', flush=True)
                        full_thinking += str(thinking_chunk)

                if msg.get('tool_calls'):
                    for i, tc_chunk in enumerate(msg['tool_calls']):
                        while len(raw_tool_calls) <= i:
                            raw_tool_calls.append({'id': None, 'function': {'name': '', 'arguments': ''}})
                        func_data = tc_chunk.get('function') or {}
                        if func_data.get('name'):
                            raw_tool_calls[i]['function']['name'] = func_data['name']
                        if tc_chunk.get('id'):
                            raw_tool_calls[i]['id'] = tc_chunk['id']
                        if 'arguments' in func_data and func_data['arguments'] is not None:
                            args_val = func_data['arguments']
                            if isinstance(args_val, str):
                                raw_tool_calls[i]['function']['arguments'] += args_val
                            else:
                                raw_tool_calls[i]['function']['arguments'] += json.dumps(args_val)

                if 'eval_count' in chunk:
                    token_count = chunk['eval_count']
                    prof.increment('ollama_stream_tokens', token_count)
                if 'prompt_eval_count' in chunk:
                    prof.increment('ollama_prompt_tokens', chunk['prompt_eval_count'])

        print("\n--- End of response ---\n")

        tool_calls = finalize_tool_calls(raw_tool_calls)
        if tool_calls:
            print_tool_calls(tool_calls)

        return full_content, full_thinking, tool_calls, token_count

    except Exception as e:
        log.error(f"Streaming error: {e}")
        return None, None, [], 0


def get_chat_response(messages: List[Dict], session_num: int, tools=None):
    log = get_logger()
    prof = get_profiler()

    try:
        with prof.measure('ollama_sync_total'):
            response = ollama.chat(
                model=MODEL_NAME, messages=messages,
                options=get_ollama_options(),
                stream=False, tools=tools,
                keep_alive=OLLAMA_KEEP_ALIVE,
            )

        content = response['message'].get('content', '')
        raw_tool_calls = response['message'].get('tool_calls', []) or []
        tool_calls = finalize_tool_calls([normalize_tool_call(tc) for tc in raw_tool_calls])

        prof.record_ollama_metrics(response)
        prof.increment('ollama_stream_tokens', response.get('eval_count', 0))
        prof.increment('ollama_prompt_tokens', response.get('prompt_eval_count', 0))

        thinking = None
        if LOG_THINKING:
            thinking = extract_thinking(response['message'])
            if thinking:
                print(f"\n--- Thinking (session {session_num}) ---")
                print(f"\033[90m{thinking}\033[0m")
                print("--- End thinking ---\n")

        if content:
            print(f"\n--- Session {session_num}, response ---")
            print(content)
            print("\n--- End response ---\n")

        if tool_calls:
            print_tool_calls(tool_calls)

        return content, thinking, tool_calls, response.get('eval_count', 0)

    except Exception as e:
        log.error(f"Ollama error: {e}")
        return None, None, [], 0


def _truncate(text: str, limit: int = MAX_TOOL_RESULT_LENGTH) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + f"\n\n... [TRUNCATED {len(text) - limit} chars] ...\n\n" + text[-tail:]


def process_tool_calls(
    tool_calls: List[Dict], session_num: int, current_msg_num: int, history: List[Dict]
) -> bool:
    tools_log = get_logger('tools')
    prof = get_profiler()
    end_session_requested = False

    for tc in tool_calls:
        func = tc.get('function', {}) or {}
        tool_name = func.get('name')
        arguments = func.get('arguments', {})
        if isinstance(arguments, str):
            arguments = _safe_parse_args(arguments)

        tools_log.info(f"Session {session_num}, msg {current_msg_num}: calling tool '{tool_name}'")
        print_tool_call(tool_name, arguments if isinstance(arguments, dict) else {})

        if tool_name == 'end_session':
            get_logger().info(f"Session {session_num}: 'end_session' tool invoked.")
            result = '__END_SESSION__'
            end_session_requested = True
        else:
            with prof.measure(f'tool:{tool_name}'):
                result = execute_tool(tool_name, arguments if isinstance(arguments, dict) else {})

        result = _truncate(result)
        prof.increment(f'tool_calls:{tool_name}')

        tools_log.info(
            f"Session {session_num}, msg {current_msg_num}: tool '{tool_name}' "
            f"returned length {len(result)}"
        )
        print_tool_result(tool_name, result)

        history.append({
            'role': 'tool',
            'tool_call_id': tc.get('id') or f'tc_{current_msg_num}',
            'content': result,
        })

        if end_session_requested:
            return True

    return False


def main():
    global current_session_mgr, io_manager, running
    signal.signal(signal.SIGINT, signal_handler)

    log = get_logger()
    log.info("Starting LLM environment with Ollama")

    io_manager = BackgroundIOManager()
    prof = get_profiler()

    tool_loader.load_tools()
    log.info(f"Loaded tools: {list(tool_loader.AVAILABLE_TOOLS.keys())}")

    system_prompt = load_prompt(SYSTEM_PROMPT_FILE)
    user_first = load_prompt(USER_FIRST_MESSAGE_FILE)
    initial_history = build_initial_history(system_prompt, user_first)

    session_mgr = SessionManager(io_manager=io_manager)
    current_session_mgr = session_mgr
    msg_generator = UserMessageGenerator(inactivity_threshold=3)

    get_ollama_options()

    for session_index in range(MAX_SESSION_NUM):
        if not running:
            break

        session = session_mgr.load_or_create_session(initial_history)
        history = session.get('history', [])
        session_num = session['number']
        log.info(
            f"Using session #{session_num}, status: {session['status']}, "
            f"history length: {len(history)}"
        )

        no_tool_streak = 0

        while running:
            session = session_mgr.current_session
            history = session.get('history', [])

            if history and history[-1]['role'] in ('assistant', 'tool'):
                next_msg_num = len(history) + 1
                user_msg_content = msg_generator.generate(session_num, next_msg_num, no_tool_streak)
                history.append({'role': 'user', 'content': user_msg_content})
                with prof.measure('session_save'):
                    session_mgr.update_current_session(history=history)
                log_message(session_num, len(history), 'user', user_msg_content)

            with prof.measure('prune_history'):
                pruned = prune_history(history, MAX_HISTORY_MESSAGES)
            with prof.measure('prepare_history'):
                api_history = prepare_history_for_api(pruned, THINKING_HISTORY_MODE)

            write_current_input_async(api_history, session_num)

            current_tools = get_tools_for_ollama()

            if STREAM_MESSAGES:
                assistant_msg, thinking, tool_calls, eval_count = stream_chat(
                    api_history, session_num, current_tools
                )
            else:
                assistant_msg, thinking, tool_calls, eval_count = get_chat_response(
                    api_history, session_num, current_tools
                )

            if assistant_msg is None and not tool_calls:
                log.error(f"Failed to get response for session {session_num}.")
                session_mgr.complete_current_session()
                break

            if assistant_msg or tool_calls:
                entry = {'role': 'assistant'}
                if assistant_msg:
                    entry['content'] = assistant_msg
                if thinking:
                    entry['thinking'] = thinking
                if tool_calls:
                    entry['tool_calls'] = tool_calls
                history.append(entry)
                with prof.measure('session_save'):
                    session_mgr.update_current_session(history=history)

            no_tool_streak = msg_generator.update_streak(tool_calls, no_tool_streak)

            end_session = False
            if tool_calls:
                end_session = process_tool_calls(tool_calls, session_num, len(history), history)
                with prof.measure('session_save'):
                    session_mgr.update_current_session(history=history)

            if end_session:
                log.info(f"Session {session_num}: end_session called, completing.")
                session_mgr.complete_current_session()
                break

            prof.step()

            if tool_calls:
                continue
            else:
                break

        if not running:
            break

    log.info("All sessions processed. Flushing background I/O...")
    save_metrics_and_exit()


if __name__ == "__main__":
    main()
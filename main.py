import signal
import sys
from typing import Optional, List, Dict

from config import (
    MAX_SESSION_NUM, STREAM_MESSAGES, LOG_THINKING,
    SYSTEM_PROMPT_FILE, USER_FIRST_MESSAGE_FILE,
    MAX_TOOL_RESULT_LENGTH,
)
from logger_setup import get_logger
from session_manager import SessionManager
import tool_loader
from tool_loader import get_tools_for_ollama, execute_tool
from message_generator import UserMessageGenerator
from background_io import BackgroundIOManager
from profiler import get_profiler
from ollama_client import stream_chat, sync_chat
from output_formatter import print_tool_calls, print_tool_call, print_tool_result
from history_utils import (
    build_initial_history, prune_history, prepare_history_for_api,
    write_current_input_async
)

running = True
current_session_mgr: Optional[SessionManager] = None
io_manager: Optional[BackgroundIOManager] = None


def save_metrics_and_exit():
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


def load_prompt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        get_logger().warning(f"Prompt file not found: {file_path}")
        return ""


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
            # simple fallback – already parsed in finalize_tool_calls, but keep safety
            try:
                import json
                arguments = json.loads(arguments) if arguments else {}
            except:
                arguments = {}

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

            with prof.measure('prune_history'):
                pruned = prune_history(history)
            with prof.measure('prepare_history'):
                api_history = prepare_history_for_api(pruned)

            write_current_input_async(io_manager, api_history, session_num)

            current_tools = get_tools_for_ollama()

            if STREAM_MESSAGES:
                assistant_msg, thinking, tool_calls, eval_count = stream_chat(
                    api_history, session_num, current_tools
                )
            else:
                assistant_msg, thinking, tool_calls, eval_count = sync_chat(
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
                if thinking and LOG_THINKING:
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
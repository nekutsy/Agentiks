import signal
import sys
import json
from typing import Optional, List, Dict
import ollama
from config import (MAX_SESSION_NUM, MODEL_NAME, OLLAMA_OPTIONS, STREAM_MESSAGES, LOG_THINKING,
                    SYSTEM_PROMPT_FILE, USER_FIRST_MESSAGE_FILE, LOG_CURRENT_INPUT, THINKING_HISTORY_MODE)
from logger_setup import get_logger
from session_manager import SessionManager
import tool_loader
from tool_loader import get_tools_for_ollama, execute_tool
from message_generator import UserMessageGenerator

running = True
current_session_mgr: Optional[SessionManager] = None

def signal_handler(sig, frame):
    global running
    get_logger().info("Received interrupt signal, shutting down gracefully...")
    running = False
    if current_session_mgr and current_session_mgr.current_session:
        get_logger().info(f"Session {current_session_mgr.current_session['number']} interrupted, marking as completed.")
        current_session_mgr.complete_current_session()
    sys.exit(0)

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

def get_next_user_message(session_num: int, msg_num: int) -> str:
    return f"Automatically generated message number {msg_num}. Go on with your business."

def write_current_input(messages: List[Dict], session_num: int):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        input_text = f"=== Session {session_num} ===\n{formatted}\n=== END INPUT ===\n"
    except Exception:
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
    with open(LOG_CURRENT_INPUT, 'w', encoding='utf-8') as f:
        f.write(input_text)

def log_message(session_num: int, msg_num: int, role: str, content: str = None, tool_calls: List = None):
    log = get_logger()
    text_log = get_logger('text')
    log.info(f"Session {session_num}, msg {msg_num}: {role} message")
    if role == 'assistant' and content:
        text_log.info(f"Session {session_num}, msg {msg_num} (response):\n{content}")
    elif role == 'tool' and content:
        text_log.info(f"Session {session_num}, msg {msg_num} (tool result):\n{content[:200]}")

def normalize_tool_call(tc):
    if isinstance(tc, dict):
        return tc
    if hasattr(tc, 'model_dump'):
        return tc.model_dump()
    if hasattr(tc, 'dict'):
        return tc.dict()
    return {"raw": str(tc)}

def extract_thinking(msg: Dict) -> Optional[str]:
    for key in ['thinking', 'reasoning', 'reasoning_content']:
        if key in msg and msg[key]:
            return msg[key]
    return None

def finalize_tool_calls(raw_tool_calls: List[Dict]) -> List[Dict]:
    finalized = []
    for i, tc in enumerate(raw_tool_calls):
        func = tc.get('function', {})
        args = func.get('arguments', '')
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        finalized.append({
            'id': tc.get('id', f'tc_{i}'),
            'type': 'function',
            'function': {
                'name': func.get('name', ''),
                'arguments': args
            }
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
                new_msg['content'] = f"<think>{thinking}</think>\n\n{content}" if content else f"<think>{thinking}</think>"
        formatted.append(new_msg)
    return formatted

def stream_chat(messages: List[Dict], session_num: int, tools=None):
    log = get_logger()
    text_log = get_logger('text')
    write_current_input(messages, session_num)

    full_content = ""
    full_thinking = ""
    raw_tool_calls = []
    token_count = 0

    try:
        stream = ollama.chat(model=MODEL_NAME, messages=messages, options=OLLAMA_OPTIONS, stream=True, tools=tools)
        print(f"\n--- Session {session_num}, generating response ---")

        for chunk in stream:
            if 'message' in chunk:
                msg = chunk['message']
                if 'content' in msg and msg['content']:
                    print(msg['content'], end='', flush=True)
                    full_content += msg['content']

                if LOG_THINKING:
                    thinking_chunk = extract_thinking(msg)
                    if thinking_chunk:
                        print(f"\033[94m{thinking_chunk}\033[0m", end='', flush=True)
                        full_thinking += str(thinking_chunk)

                if 'tool_calls' in msg and msg['tool_calls']:
                    for i, tc_chunk in enumerate(msg['tool_calls']):
                        while len(raw_tool_calls) <= i:
                            raw_tool_calls.append({'function': {'name': '', 'arguments': ''}})
                        func_data = tc_chunk.get('function', {})
                        if func_data.get('name'):
                            raw_tool_calls[i]['function']['name'] = func_data['name']
                        if 'arguments' in func_data and func_data['arguments']:
                            args_val = func_data['arguments']
                            args_str = args_val if isinstance(args_val, str) else json.dumps(args_val)
                            raw_tool_calls[i]['function']['arguments'] += args_str

            if 'eval_count' in chunk:
                token_count = chunk['eval_count']

        print("\n--- End of response ---\n")
        if full_thinking:
            text_log.info(f"Session {session_num} (thinking):\n{full_thinking}")

        tool_calls = finalize_tool_calls(raw_tool_calls)
        
        if tool_calls:
            print("\033[33m=== Tool Calls ===\033[0m")
            for tc in tool_calls:
                func = tc.get('function', {})
                name = func.get('name', 'unknown')
                args = func.get('arguments', {})
                print(f"\033[33mTool: {name}\033[0m")
                print(json.dumps(args, indent=2, ensure_ascii=False))
            print("\033[33m=================\033[0m\n")
        
        return full_content, full_thinking, tool_calls, token_count
    except Exception as e:
        log.error(f"Streaming error: {e}")
        return None, None, [], 0

def get_chat_response(messages: List[Dict], session_num: int, tools=None):
    log = get_logger()
    text_log = get_logger('text')
    write_current_input(messages, session_num)

    try:
        response = ollama.chat(model=MODEL_NAME, messages=messages, options=OLLAMA_OPTIONS, stream=False, tools=tools)
        content = response['message'].get('content', '')
        raw_tool_calls = response['message'].get('tool_calls', [])
        tool_calls = [normalize_tool_call(tc) for tc in raw_tool_calls]
        eval_count = response.get('eval_count', 0)

        thinking = None
        if LOG_THINKING:
            thinking = extract_thinking(response['message'])
            if thinking:
                print(f"\n--- Thinking (session {session_num}) ---")
                print(f"\033[90m{thinking}\033[0m")
                print("--- End thinking ---\n")
                text_log.info(f"Session {session_num} (thinking):\n{thinking}")

        if tool_calls:
            print("\033[33m=== Tool Calls ===\033[0m")
            for tc in tool_calls:
                func = tc.get('function', {})
                name = func.get('name', 'unknown')
                args = func.get('arguments', {})
                print(f"\033[33mTool: {name}\033[0m")
                print(json.dumps(args, indent=2, ensure_ascii=False))
            print("\033[33m=================\033[0m\n")

        return content, thinking, tool_calls, eval_count
    except Exception as e:
        log.error(f"Ollama error: {e}")
        return None, None, [], 0

def process_tool_calls(tool_calls: List[Dict], session_num: int, current_msg_num: int, history: List[Dict]) -> bool:
    tools_log = get_logger('tools')
    current_tool_log = get_logger('current_tool')
    for tc in tool_calls:
        func = tc.get('function', {})
        tool_name = func.get('name')
        arguments = func.get('arguments', {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except:
                arguments = {}

        tools_log.info(f"Session {session_num}, msg {current_msg_num}: calling tool '{tool_name}' with args {arguments}")
        current_tool_log.info(f"Session {session_num}, msg {current_msg_num}: executing tool '{tool_name}'")

        result = execute_tool(tool_name, arguments)
        tools_log.info(f"Session {session_num}, msg {current_msg_num}: tool '{tool_name}' returned: {result[:200]}")

        tool_msg = {'role': 'tool', 'tool_call_id': tc.get('id', 'unknown'), 'content': result}
        history.append(tool_msg)
        msg_num = len(history)
        log_message(session_num, msg_num, 'tool', result)

        if tool_name == 'end_session' and result == '__END_SESSION__':
            return True
    return False

def main():
    global current_session_mgr
    signal.signal(signal.SIGINT, signal_handler)
    log = get_logger()
    log.info("Starting LLM environment with Ollama")

    tool_loader.load_tools()
    log.info(f"Loaded tools: {list(tool_loader.AVAILABLE_TOOLS.keys())}")

    system_prompt = load_prompt(SYSTEM_PROMPT_FILE)
    user_first = load_prompt(USER_FIRST_MESSAGE_FILE)
    initial_history = build_initial_history(system_prompt, user_first)

    session_mgr = SessionManager()
    current_session_mgr = session_mgr
    msg_generator = UserMessageGenerator(inactivity_threshold=3)

    for session_index in range(MAX_SESSION_NUM):
        if not running:
            break

        session = session_mgr.load_or_create_session(initial_history)
        history = session.get('history', [])
        session_num = session['number']
        log.info(f"Using session #{session_num}, status: {session['status']}, history length: {len(history)}")

        no_tool_streak = 0

        while running and session_mgr.is_run_exists():
            session = session_mgr.current_session
            history = session.get('history', [])

            if history and history[-1]['role'] == 'assistant':
                next_msg_num = len(history) + 1
                user_msg_content = msg_generator.generate(session_num, next_msg_num, no_tool_streak)
                history.append({'role': 'user', 'content': user_msg_content})
                session_mgr.update_current_session(history=history)
                log_message(session_num, len(history), 'user', user_msg_content)

            current_tools = get_tools_for_ollama()
            api_history = prepare_history_for_api(history, THINKING_HISTORY_MODE)

            if STREAM_MESSAGES:
                assistant_msg, thinking, tool_calls, eval_count = stream_chat(api_history, session_num, current_tools)
            else:
                assistant_msg, thinking, tool_calls, eval_count = get_chat_response(api_history, session_num, current_tools)

            if assistant_msg is None and not tool_calls:
                log.error(f"Failed to get response for session {session_num}. Completing session.")
                session_mgr.complete_current_session()
                break

            if assistant_msg or tool_calls:
                assistant_entry = {'role': 'assistant'}
                if assistant_msg:
                    assistant_entry['content'] = assistant_msg
                if thinking:
                    assistant_entry['thinking'] = thinking
                if tool_calls:
                    assistant_entry['tool_calls'] = tool_calls
                history.append(assistant_entry)
                msg_num = len(history)
                log_message(session_num, msg_num, 'assistant', assistant_msg)
                session_mgr.update_current_session(history=history)

            no_tool_streak = msg_generator.update_streak(tool_calls, no_tool_streak)

            end_session = False
            if tool_calls:
                current_msg_num = len(history)
                end_session = process_tool_calls(tool_calls, session_num, current_msg_num, history)
                session_mgr.update_current_session(history=history)

            if end_session:
                log.info(f"Session {session_num}: end_session tool called, completing session.")
                session_mgr.complete_current_session()
                break

            if tool_calls:
                continue

        if not session_mgr.is_run_exists():
            log.info(f"Session {session_num}: is_run removed, session completed successfully.")
            session_mgr.complete_current_session()

        if not running:
            break

    log.info("All sessions processed or interrupted. Exiting.")

if __name__ == "__main__":
    main()
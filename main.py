import signal
import sys
import time
import json
from typing import Optional, List, Dict

import ollama

from config import (MAX_SESSION_NUM, MODEL_NAME, OLLAMA_OPTIONS, STREAM_MESSAGES, LOG_THINKING,
                    SYSTEM_PROMPT_FILE, USER_FIRST_MESSAGE_FILE, LOG_CURRENT_INPUT)
from logger_setup import get_logger
from session_manager import SessionManager
import tool_loader
from tool_loader import get_tools_for_ollama, execute_tool

running = True
current_session_mgr: Optional[SessionManager] = None

def signal_handler(sig, frame):
    global running
    get_logger().info("Received interrupt signal, shutting down gracefully...")
    running = False
    if current_session_mgr and current_session_mgr.current_session:
        get_logger().info(f"Session {current_session_mgr.current_session['number']} interrupted, will resume next time.")
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

def serialize_tool_call(tc):
    if isinstance(tc, dict):
        return tc
    if hasattr(tc, '__dict__'):
        d = {k: v for k, v in tc.__dict__.items() if not k.startswith('_')}
        if 'function' in d and hasattr(d['function'], '__dict__'):
            d['function'] = {k: v for k, v in d['function'].__dict__.items() if not k.startswith('_')}
        return d
    return {"raw": str(tc)}

def stream_chat(messages: List[Dict], session_num: int, tools=None):
    log = get_logger()
    text_log = get_logger('text')
    
    write_current_input(messages, session_num)
    
    full_content = ""
    full_thinking = ""
    tool_calls = []
    token_count = 0

    try:
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options=OLLAMA_OPTIONS,
            stream=True,
            tools=tools
        )
        print(f"\n--- Session {session_num}, generating response ---")
        
        for chunk in stream:
            if 'message' in chunk:
                msg = chunk['message']
                if 'content' in msg and msg['content']:
                    content = msg['content']
                    print(content, end='', flush=True)
                    full_content += content
                
                if LOG_THINKING:
                    thinking_parts = []
                    if 'thinking' in msg and msg['thinking']:
                        thinking_parts.append(msg['thinking'])
                    if 'reasoning' in msg and msg['reasoning']:
                        thinking_parts.append(msg['reasoning'])
                    if 'reasoning_content' in msg and msg['reasoning_content']:
                        thinking_parts.append(msg['reasoning_content'])
                    for part in thinking_parts:
                        if part:
                            print(f"\033[94m{part}\033[0m", end='', flush=True)
                            full_thinking += part
                
                if 'tool_calls' in msg and msg['tool_calls']:
                    for tc in msg['tool_calls']:
                        tool_calls.append(serialize_tool_call(tc))
            
            if 'eval_count' in chunk:
                token_count = chunk['eval_count']
        
        print("\n--- End of response ---\n")
        
        if full_thinking:
            text_log.info(f"Session {session_num} (thinking):\n{full_thinking}")
        
        return full_content, tool_calls, token_count
    
    except Exception as e:
        log.error(f"Streaming error: {e}")
        return None, [], 0

def get_chat_response(messages: List[Dict], session_num: int, tools=None):
    log = get_logger()
    text_log = get_logger('text')
    
    write_current_input(messages, session_num)
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options=OLLAMA_OPTIONS,
            stream=False,
            tools=tools
        )
        content = response['message']['content']
        tool_calls = [serialize_tool_call(tc) for tc in response['message'].get('tool_calls', [])]
        eval_count = response.get('eval_count', 0)
        
        if LOG_THINKING and 'message' in response:
            msg = response['message']
            thinking = None
            if 'thinking' in msg and msg['thinking']:
                thinking = msg['thinking']
            elif 'reasoning' in msg and msg['reasoning']:
                thinking = msg['reasoning']
            elif 'reasoning_content' in msg and msg['reasoning_content']:
                thinking = msg['reasoning_content']
            if thinking:
                print(f"\n--- Thinking (session {session_num}) ---")
                print(f"\033[90m{thinking}\033[0m")
                print("--- End thinking ---\n")
                text_log.info(f"Session {session_num} (thinking):\n{thinking}")
        
        return content, tool_calls, eval_count
    except Exception as e:
        log.error(f"Ollama error: {e}")
        return None, [], 0

def process_tool_calls(tool_calls: List[Dict], session_num: int, current_msg_num: int, history: List[Dict]) -> bool:
    tools_log = get_logger('tools')
    current_tool_log = get_logger('current_tool')
    
    for tc in tool_calls:
        if 'function' in tc:
            func = tc['function']
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
            
            tool_msg = {
                'role': 'tool',
                'tool_call_id': tc.get('id', 'unknown'),
                'content': result
            }
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

    log.info(f"System prompt loaded: {len(system_prompt)} chars")
    log.info(f"User first message loaded: {len(user_first)} chars")

    session_mgr = SessionManager()
    current_session_mgr = session_mgr

    for session_index in range(MAX_SESSION_NUM):
        if not running:
            break

        session = session_mgr.load_or_create_session(initial_history)
        history = session.get('history', [])
        session_num = session['number']
        log.info(f"Using session #{session_num}, status: {session['status']}, history length: {len(history)}")

        while running and session_mgr.is_run_exists():
            session = session_mgr.current_session
            history = session.get('history', [])

            if history and history[-1]['role'] == 'assistant':
                next_msg_num = len(history) + 1
                user_msg_content = get_next_user_message(session_num, next_msg_num)
                user_msg = {'role': 'user', 'content': user_msg_content}
                history.append(user_msg)
                session_mgr.update_current_session(history=history)
                log_message(session_num, len(history), 'user', user_msg_content)

            current_tools = get_tools_for_ollama()
            
            if STREAM_MESSAGES:
                assistant_msg, tool_calls, eval_count = stream_chat(history, session_num, current_tools)
            else:
                assistant_msg, tool_calls, eval_count = get_chat_response(history, session_num, current_tools)

            if assistant_msg is None and not tool_calls:
                log.error(f"Failed to get response for session {session_num}. Completing session.")
                session_mgr.complete_current_session()
                break

            if assistant_msg or tool_calls:
                assistant_entry = {'role': 'assistant'}
                if assistant_msg:
                    assistant_entry['content'] = assistant_msg
                if tool_calls:
                    assistant_entry['tool_calls'] = tool_calls
                history.append(assistant_entry)
                msg_num = len(history)
                log_message(session_num, msg_num, 'assistant', assistant_msg)
                session_mgr.update_current_session(history=history)
            
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
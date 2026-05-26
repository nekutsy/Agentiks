import json
from typing import List, Dict, Any, Optional, Tuple

import ollama

from config import (
    MODEL_NAME, OLLAMA_OPTIONS, LOG_THINKING, STREAM_MESSAGES,
    OLLAMA_NUM_CTX, OLLAMA_NUM_BATCH, OLLAMA_NUM_GPU, OLLAMA_NUM_THREAD,
    OLLAMA_FLASH_ATTENTION, OLLAMA_KEEP_ALIVE,
)
from logger_setup import get_logger
from profiler import get_profiler


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


def extract_thinking(msg: Dict) -> Optional[str]:
    for key in ('thinking', 'reasoning', 'reasoning_content'):
        if key in msg and msg[key]:
            return msg[key]
    return None


def normalize_tool_call(tc) -> Dict:
    if isinstance(tc, dict):
        return tc
    if hasattr(tc, 'model_dump'):
        return tc.model_dump()
    if hasattr(tc, 'dict'):
        return tc.dict()
    return {"raw": str(tc)}


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


def stream_chat(
    messages: List[Dict], session_num: int, tools=None
) -> Tuple[Optional[str], Optional[str], List[Dict], int]:
    log = get_logger()
    prof = get_profiler()

    full_content = ""
    full_thinking = ""
    raw_tool_calls: List[Dict] = []
    last_chunk = None

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
                last_chunk = chunk
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

            if last_chunk and isinstance(last_chunk, dict):
                eval_count = last_chunk.get('eval_count', 0)
                prompt_eval_count = last_chunk.get('prompt_eval_count', 0)
                if eval_count:
                    prof.increment('ollama_stream_tokens', eval_count)
                if prompt_eval_count:
                    prof.increment('ollama_prompt_tokens', prompt_eval_count)
                prof.record_ollama_metrics(last_chunk)

        print("\n--- End of response ---\n")

        tool_calls = finalize_tool_calls(raw_tool_calls)
        token_count = last_chunk.get('eval_count', 0) if last_chunk else 0
        return full_content, full_thinking, tool_calls, token_count

    except Exception as e:
        log.error(f"Streaming error: {e}")
        return None, None, [], 0


def sync_chat(
    messages: List[Dict], session_num: int, tools=None
) -> Tuple[Optional[str], Optional[str], List[Dict], int]:
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

        return content, thinking, tool_calls, response.get('eval_count', 0)

    except Exception as e:
        log.error(f"Ollama error: {e}")
        return None, None, [], 0
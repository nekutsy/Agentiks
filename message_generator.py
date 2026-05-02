from typing import List, Dict, Optional

class UserMessageGenerator:
    def __init__(self, inactivity_threshold: int = 3, warning_msg: str = None):
        self.inactivity_threshold = inactivity_threshold
        self.warning_msg = warning_msg or "WARNING: You haven't used any tools for several steps. If you continue without tool usage, the session will be terminated."

    def generate(self, session_num: int, msg_num: int, no_tool_streak: int) -> str:
        base_msg = f"Go on.."
        if no_tool_streak >= self.inactivity_threshold:
            return f"{base_msg}\n\n{self.warning_msg}"
        return base_msg

    def update_streak(self, tool_calls: List[Dict], current_streak: int) -> int:
        if tool_calls:
            return 0
        else:
            return current_streak + 1
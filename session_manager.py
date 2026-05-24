import json
import os
from typing import Optional, Dict, Any, List

from config import LOG_SESSIONS
from logger_setup import get_logger


class SessionManager:
    def __init__(self, io_manager=None):
        self.sessions_file = LOG_SESSIONS
        self.current_session: Optional[Dict[str, Any]] = None
        self._io = io_manager

    def _load_sessions(self) -> Dict[int, Dict]:
        if os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                return {int(k): v for k, v in json.load(f).items()}
        return {}

    def _save_sessions(self, sessions: Dict[int, Dict]):
        if self._io is not None:
            self._io.save_session(self.sessions_file, sessions)
        else:
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, indent=2, ensure_ascii=False)

    def get_last_session(self) -> Optional[Dict]:
        sessions = self._load_sessions()
        if not sessions:
            return None
        max_num = max(sessions.keys())
        session = sessions[max_num]
        session['number'] = max_num
        return session

    def is_session_completed(self, session: Dict) -> bool:
        return session.get('status') == 'completed'

    def create_new_session(self, session_num: int, initial_history: List[Dict]) -> Dict:
        session = {'number': session_num, 'status': 'active', 'history': initial_history}
        sessions = self._load_sessions()
        sessions[session_num] = session
        self._save_sessions(sessions)
        self.current_session = session
        return session

    def load_or_create_session(self, initial_history: List[Dict] = None) -> Dict:
        last = self.get_last_session()
        if last and not self.is_session_completed(last):
            if 'history' not in last:
                last['history'] = []
            self.current_session = last
            return last
        next_num = (last['number'] + 1) if last else 1
        return self.create_new_session(next_num, initial_history or [])

    def complete_current_session(self):
        if self.current_session:
            sessions = self._load_sessions()
            num = self.current_session['number']
            if num in sessions:
                sessions[num]['status'] = 'completed'
                self._save_sessions(sessions)
            self.current_session = None

    def update_current_session(self, **kwargs):
        if self.current_session:
            sessions = self._load_sessions()
            num = self.current_session['number']
            if num in sessions:
                sessions[num].update(kwargs)
                self._save_sessions(sessions)
                self.current_session.update(kwargs)
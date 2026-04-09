import json
import os
from typing import Optional, Dict, Any, List

from config import LOG_SESSIONS, IS_RUN_FILE

class SessionManager:
    def __init__(self):
        self.sessions_file = LOG_SESSIONS
        self.is_run_file = IS_RUN_FILE
        self.current_session: Optional[Dict[str, Any]] = None
        self._ensure_is_run()

    def _ensure_is_run(self):
        if not os.path.exists(self.is_run_file):
            open(self.is_run_file, 'w').close()

    def _load_sessions(self) -> Dict[int, Dict]:
        if os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                return {int(k): v for k, v in json.load(f).items()}
        return {}

    def _save_sessions(self, sessions: Dict[int, Dict]):
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
        session = {
            'number': session_num,
            'status': 'active',
            'history': initial_history
        }
        sessions = self._load_sessions()
        sessions[session_num] = session
        self._save_sessions(sessions)
        self.current_session = session
        return session

    def load_or_create_session(self, initial_history: List[Dict] = None) -> Dict:
        last = self.get_last_session()
        if last and not self.is_session_completed(last):
            # Убедимся, что в загруженной сессии есть поле history
            if 'history' not in last:
                last['history'] = []
            self.current_session = last
            return last
        else:
            next_num = (last['number'] + 1) if last else 1
            if initial_history is None:
                initial_history = []
            return self.create_new_session(next_num, initial_history)

    def complete_current_session(self):
        if self.current_session:
            sessions = self._load_sessions()
            if self.current_session['number'] in sessions:
                sessions[self.current_session['number']]['status'] = 'completed'
                self._save_sessions(sessions)
            self.current_session = None
        if os.path.exists(self.is_run_file):
            os.remove(self.is_run_file)

    def update_current_session(self, **kwargs):
        if self.current_session:
            sessions = self._load_sessions()
            num = self.current_session['number']
            if num in sessions:
                sessions[num].update(kwargs)
                self._save_sessions(sessions)
                self.current_session.update(kwargs)

    def is_run_exists(self) -> bool:
        return os.path.exists(self.is_run_file)
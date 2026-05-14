from dataclasses import dataclass
from typing import List
from config import Config


@dataclass
class Message:
    role: str    # "user" or "assistant"
    content: str


class ChatMemory:


    def __init__(self, window: int = Config.MEMORY_WINDOW):
        self.window = window
        self._messages: List[Message] = []

    def add(self, role: str, content: str) -> None:
        """Append a single message and enforce window limit."""
        self._messages.append(Message(role=role, content=content))
        # window turns = window * 2 messages (user + assistant each)
        max_messages = self.window * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def format_for_prompt(self) -> str:

        if not self._messages:
            return "No previous conversation."
        lines = []
        for msg in self._messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self._messages) == 0

    def clear(self) -> None:
        self._messages = []

from langchain_core.messages import HumanMessage, AIMessage
from typing import List, T


class ChatMemory:
    def __init__(self):
        self.history: List[HumanMessage | AIMessage] = []

    def add_user_message(self, message: str):
        self.history.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.history.append(AIMessage(content=message)) # for previous message

    def is_empty(self) -> bool:
        return len(self.history) == 0
    
    def get_history(self) -> List[HumanMessage | AIMessage]:
        return self.history
    
    def clear(self):
        self.history = []

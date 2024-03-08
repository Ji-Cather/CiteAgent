from langchain.schema import PromptValue as langchain_promptvalue
from typing import List
from LLMGraph.message import Message

# prompt value ： 为了适配langchain的get_buffer_string函数
def get_buffer_string(
    messages: List[Message]
) -> str:
    """Get buffer string of messages."""
    string_messages = []
    for m in messages:
        if isinstance(m, Message):
            role = m.sender
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        string_messages.append(f"{role}: {m.content}")
    return "\n".join(string_messages)


class PromptValue(langchain_promptvalue):
    
    messages: List[Message]
    
    def to_string(self) -> str:
        """Return prompt as string."""
        return get_buffer_string(self.messages)

    def to_messages(self) -> List[Message]:
        """Return prompt as messages."""
        return self.messages